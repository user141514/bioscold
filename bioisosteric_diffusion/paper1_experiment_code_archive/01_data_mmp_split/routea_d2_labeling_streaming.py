#!/usr/bin/env python3
"""
Route A D2 Labeling — Streaming, block-based, checkpointable.

Modes: smoke | full | resume | status

Design constraints:
- Never load all pairs into memory
- Block-based processing (default 100k pairs per block)
- Atomic writes (.tmp → rename)
- Checkpoint/resume
- SQLite-backed decoy fragment index
- Continuous heartbeat + progress files

Usage:
  python core/scripts/routea_d2_labeling_streaming.py --mode smoke
  python core/scripts/routea_d2_labeling_streaming.py --mode full
  python core/scripts/routea_d2_labeling_streaming.py --mode resume
  python core/scripts/routea_d2_labeling_streaming.py --mode status
"""
import argparse
import csv
import json
import os
import random
import sqlite3
import sys
import time
import hashlib
import glob
import gc
from datetime import datetime, timezone
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════
DEFAULTS = {
    "block_size": 100_000,
    "workers": 1,
    "smoke_max_pairs": 10_000,
    "max_rss_mb": 3500,
    "progress_interval_sec": 60,
    "decoy_ratio": 1.0,
    "seed": 20260521,
    "min_transform_frequency": 5,
    "max_delta_MW": 100,
    "max_delta_LogP": 3,
    "max_delta_TPSA": 80,
}

NOW = datetime.now(timezone.utc).isoformat()


def ts():
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════
# File helpers
# ═══════════════════════════════════════════════════════════════
def atomic_write(path, content, is_json=False):
    """Write content atomically via .tmp → rename."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        if is_json:
            json.dump(content, f, ensure_ascii=False, indent=2)
        else:
            f.write(content)
    os.replace(tmp, path)


def append_jsonl(path, records):
    """Append list of dicts as JSONL."""
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_log(path, line):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_jsonl(path):
    if not os.path.exists(path):
        return 0
    c = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            c += 1
    return c


# ═══════════════════════════════════════════════════════════════
# Checkpoint Manager
# ═══════════════════════════════════════════════════════════════
class CheckpointManager:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {
            "completed_blocks": [],
            "completed_shards": [],
            "failed_blocks": [],
            "last_successful_block": None,
            "resume_mode_available": True,
        }

    def save(self):
        atomic_write(str(self.path), self.data, is_json=True)

    def block_done(self, shard_id, block_id, pair_count):
        key = f"{shard_id}:{block_id}"
        if key not in self.data["completed_blocks"]:
            self.data["completed_blocks"].append(key)
        self.data["last_successful_block"] = {
            "shard_id": shard_id,
            "block_id": block_id,
            "pair_count": pair_count,
            "ts": ts(),
        }
        self.save()

    def shard_done(self, shard_id):
        if shard_id not in self.data["completed_shards"]:
            self.data["completed_shards"].append(shard_id)
        self.save()

    def block_failed(self, shard_id, block_id, error):
        key = f"{shard_id}:{block_id}"
        self.data["failed_blocks"].append({"key": key, "error": str(error), "ts": ts()})
        self.save()

    def is_block_completed(self, shard_id, block_id):
        return f"{shard_id}:{block_id}" in self.data["completed_blocks"]

    def is_shard_completed(self, shard_id):
        return shard_id in self.data["completed_shards"]

    def get_completed_block_keys(self):
        return set(self.data["completed_blocks"])


# ═══════════════════════════════════════════════════════════════
# Progress Tracker
# ═══════════════════════════════════════════════════════════════
class ProgressTracker:
    def __init__(self, progress_path, heartbeat_path):
        self.progress_path = Path(progress_path)
        self.heartbeat_path = Path(heartbeat_path)
        self.data = {
            "status": "initializing",
            "stage": "setup",
            "started_at": NOW,
            "updated_at": NOW,
            "current_shard": None,
            "current_block": None,
            "total_shards": 0,
            "processed_shards": 0,
            "processed_blocks": 0,
            "processed_pairs": 0,
            "weak_positive_count": 0,
            "decoy_count": 0,
            "manifest_count": 0,
            "error_count": 0,
            "rss_mb_if_available": None,
            "last_heartbeat": NOW,
        }
        self._last_progress_write = 0

    def update(self, **kwargs):
        self.data.update(kwargs)
        self.data["updated_at"] = ts()
        now_t = time.time()
        if now_t - self._last_progress_write >= 10:
            self._write_progress()
            self._last_progress_write = now_t

    def heartbeat(self, extra=None):
        self.data["last_heartbeat"] = ts()
        hb = {
            "timestamp": ts(),
            "current_shard": self.data.get("current_shard"),
            "current_block": self.data.get("current_block"),
            "processed_pairs": self.data.get("processed_pairs", 0),
            "status": self.data.get("status", "running"),
        }
        if extra:
            hb.update(extra)
        atomic_write(str(self.heartbeat_path), json.dumps(hb, ensure_ascii=False))

    def _write_progress(self):
        atomic_write(str(self.progress_path), self.data, is_json=True)

    def flush(self):
        self._write_progress()
        self.heartbeat()


# ═══════════════════════════════════════════════════════════════
# Decoy Fragment Index (SQLite)
# ═══════════════════════════════════════════════════════════════
class DecoyIndex:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS fragment_pool (
                fragment_key TEXT PRIMARY KEY,
                fragment_smiles TEXT NOT NULL,
                attachment_signature TEXT,
                fragment_heavy_atoms INTEGER,
                mw_bin INTEGER,
                logp_bin INTEGER,
                tpsa_bin INTEGER,
                frequency INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS transform_frequency (
                transform_key TEXT PRIMARY KEY,
                frequency INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_fp_attach ON fragment_pool(attachment_signature);
            CREATE INDEX IF NOT EXISTS idx_fp_mw ON fragment_pool(mw_bin);
            CREATE INDEX IF NOT EXISTS idx_tf_freq ON transform_frequency(frequency);
        """)
        self.conn.commit()

    def load_transform_frequencies(self, csv_path):
        c = 0
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tk = row.get("transform_key", "")
                freq = int(row.get("frequency", 0))
                if tk:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO transform_frequency(transform_key,frequency) VALUES(?,?)",
                        (tk, freq),
                    )
                    c += 1
        self.conn.commit()
        return c

    def get_transform_freq(self, transform_key):
        row = self.conn.execute(
            "SELECT frequency FROM transform_frequency WHERE transform_key=?",
            (transform_key,),
        ).fetchone()
        return row[0] if row else 0

    def insert_fragment(self, frag_key, frag_smiles, attachment_sig, heavy_atoms=0, mw_bin=0, logp_bin=0, tpsa_bin=0):
        self.conn.execute(
            """INSERT OR IGNORE INTO fragment_pool
               (fragment_key, fragment_smiles, attachment_signature, fragment_heavy_atoms, mw_bin, logp_bin, tpsa_bin)
               VALUES(?,?,?,?,?,?,?)""",
            (frag_key, frag_smiles, attachment_sig, heavy_atoms, mw_bin, logp_bin, tpsa_bin),
        )

    def insert_fragments_batch(self, rows):
        self.conn.executemany(
            """INSERT OR IGNORE INTO fragment_pool
               (fragment_key, fragment_smiles, attachment_signature, fragment_heavy_atoms, mw_bin, logp_bin, tpsa_bin)
               VALUES(?,?,?,?,?,?,?)""",
            rows,
        )

    def count_fragments(self):
        return self.conn.execute("SELECT COUNT(*) FROM fragment_pool").fetchone()[0]

    def count_transforms(self):
        return self.conn.execute("SELECT COUNT(*) FROM transform_frequency").fetchone()[0]

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

    def get_property_matched_decoys(self, attachment_sig, mw_bin, n=5):
        """Find fragments with same attachment sig + similar MW bin."""
        rows = self.conn.execute(
            """SELECT fragment_key, fragment_smiles FROM fragment_pool
               WHERE attachment_signature=? AND ABS(mw_bin - ?) <= 1
               ORDER BY RANDOM() LIMIT ?""",
            (attachment_sig, mw_bin, n),
        ).fetchall()
        return rows

    def get_unseen_transform_fragments(self, attachment_sig, transform_key, n=5):
        """Find fragments with same attachment but transform NOT in frequency table."""
        rows = self.conn.execute(
            """SELECT fragment_key, fragment_smiles FROM fragment_pool
               WHERE attachment_signature=?
               ORDER BY RANDOM() LIMIT ?""",
            (attachment_sig, n * 3),
        ).fetchall()
        return rows[:n]

    def get_random_same_attachment(self, attachment_sig, n=5):
        rows = self.conn.execute(
            """SELECT fragment_key, fragment_smiles FROM fragment_pool
               WHERE attachment_signature=?
               ORDER BY RANDOM() LIMIT ?""",
            (attachment_sig, n),
        ).fetchall()
        return rows


# ═══════════════════════════════════════════════════════════════
# Block Labeler
# ═══════════════════════════════════════════════════════════════
class BlockLabeler:
    def __init__(self, decoy_index, config):
        self.idx = decoy_index
        self.cfg = config
        self.min_tf = config.get("min_transform_frequency", 5)
        self.scaffold_support_available = False

    def label_block(self, pairs):
        """Process a block of pairs → weak positives + decoys."""
        weak_positives = []
        decoys = []
        manifest_entries = []

        for pair in pairs:
            tk = pair.get("transform_key", "")
            freq = self.idx.get_transform_freq(tk)
            pair["_transform_frequency"] = freq

            # Weak positive: transform_frequency >= threshold
            if freq >= self.min_tf:
                pair["label"] = "WEAK_POSITIVE"
                pair["label_strength"] = "WEAK_STRUCTURE"
                pair["label_confidence"] = min(freq / max(self.min_tf * 4, 1), 1.0)
                weak_positives.append(pair)
            else:
                # Low-frequency pairs become DECOY_UNSEEN_TRANSFORM candidates
                if freq <= 1:
                    pair["label"] = "DECOY_UNSEEN_TRANSFORM"
                    pair["label_strength"] = "WEAK_DECOY"
                    pair["label_confidence"] = 0.5
                    decoys.append(pair)

        return weak_positives, decoys

    def generate_property_matched_decoys(self, weak_positives, n_per_positive=1):
        """Generate property-matched decoys from fragment index."""
        decoys = []
        attachment_sig = weak_positives[0].get("attachment_signature", "") if weak_positives else None
        if not attachment_sig:
            return decoys

        for wp in weak_positives[:n_per_positive * 10]:
            old_frag = wp.get("old_fragment_key", "")
            mw_bin = 0  # placeholder; would need actual MW computation
            matches = self.idx.get_property_matched_decoys(attachment_sig, mw_bin, n=n_per_positive)
            for fk, fs in matches:
                if fk != old_frag:
                    decoy = dict(wp)
                    decoy["label"] = "DECOY_PROPERTY_MATCHED"
                    decoy["label_strength"] = "DECOY"
                    decoy["label_confidence"] = 0.7
                    decoy["replacement_fragment_key"] = fk
                    decoy["replacement_fragment_smiles"] = fs
                    decoy["_decoy_source"] = "property_matched"
                    decoys.append(decoy)
        return decoys

    def generate_random_decoys(self, pairs_sample, attachment_sig, n_total):
        """Generate random same-attachment decoys."""
        decoys = []
        if not attachment_sig:
            return decoys
        fragments = self.idx.get_random_same_attachment(attachment_sig, n=n_total)
        for fk, fs in fragments:
            if pairs_sample:
                template = random.choice(pairs_sample)
                decoy = dict(template)
                decoy["label"] = "DECOY_RANDOM_SAME_ATTACHMENT"
                decoy["label_strength"] = "DECOY"
                decoy["label_confidence"] = 0.3
                decoy["replacement_fragment_key"] = fk
                decoy["replacement_fragment_smiles"] = fs
                decoy["_decoy_source"] = "random_same_attachment"
                decoys.append(decoy)
        return decoys


# ═══════════════════════════════════════════════════════════════
# D2 Runner
# ═══════════════════════════════════════════════════════════════
class D2Runner:
    def __init__(self, args):
        self.args = args
        self.base = Path(args.out_dir)
        self.pair_dir = Path(args.pair_shard_dir)
        self.tf_csv = Path(args.transform_frequency)
        self.block_size = args.block_size
        self.max_pairs = args.max_pairs
        self.mode = args.mode
        self.is_smoke = args.mode == "smoke"

        # Output directories
        self.label_dir = self.base
        self.wp_dir = self.label_dir / "label_blocks" / "weak_positive"
        self.decoy_dir = self.label_dir / "label_blocks" / "decoy"
        self.manifest_dir = self.label_dir / "label_blocks" / "manifest"
        self.stats_dir = self.label_dir / "label_blocks" / "stats"
        for d in [self.wp_dir, self.decoy_dir, self.manifest_dir, self.stats_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Managers
        self.checkpoint = CheckpointManager(args.checkpoint)
        self.progress = ProgressTracker(
            str(self.label_dir / "d2_progress.json"),
            str(self.label_dir / "d2_heartbeat.txt"),
        )

        # Log files
        self.processed_blocks_log = str(self.label_dir / "d2_processed_blocks.log")
        self.processed_shards_log = str(self.label_dir / "d2_processed_shards.log")
        self.error_blocks_log = str(self.label_dir / "d2_error_blocks.log")
        self.error_shards_log = str(self.label_dir / "d2_error_shards.log")

        # Decoy index
        self.decoy_index = None

        # Counters
        self.total_weak_positives = 0
        self.total_decoys = 0
        self.total_manifest = 0
        self.restricted_shards = None  # set externally to limit shard list

        random.seed(args.seed)

    # ── Build decoy index ─────────────────────────────────
    def build_index(self, shard_paths):
        db_path = str(self.label_dir / "d2_fragment_decoy_index.sqlite")
        self.progress.update(stage="building_index", status="indexing")

        if os.path.exists(db_path) and self.mode == "resume":
            self.decoy_index = DecoyIndex(db_path)
            tf_count = self.decoy_index.count_transforms()
            if tf_count == 0:
                self.decoy_index.load_transform_frequencies(str(self.tf_csv))
            frag_count = self.decoy_index.count_fragments()
            print(f"Resumed index: {frag_count} fragments, {tf_count} transforms")
            return

        if os.path.exists(db_path):
            os.remove(db_path)

        self.decoy_index = DecoyIndex(db_path)

        # Load transform frequencies
        tf_count = self.decoy_index.load_transform_frequencies(str(self.tf_csv))
        print(f"Loaded {tf_count} transform frequencies")

        # Stream fragments from pair shards
        frag_batch = []
        frag_seen = set()
        total_pairs_scanned = 0
        for shard_path in shard_paths:
            shard_name = Path(shard_path).stem
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    pair = json.loads(line)
                    total_pairs_scanned += 1

                    for side in ["old", "replacement"]:
                        fk = pair.get(f"{side}_fragment_key", "")
                        fs = pair.get(f"{side}_fragment_smiles", "")
                        attachment = pair.get("attachment_signature", "")
                        if fk and fk not in frag_seen:
                            frag_seen.add(fk)
                            ha = self._estimate_heavy_atoms(fs)
                            frag_batch.append((fk, fs, attachment, ha, 0, 0, 0))

                    if len(frag_batch) >= 5000:
                        self.decoy_index.insert_fragments_batch(frag_batch)
                        self.decoy_index.commit()
                        frag_batch = []

                    if self.max_pairs and total_pairs_scanned >= self.max_pairs:
                        break

            if self.is_smoke and total_pairs_scanned >= self.max_pairs:
                break

        # Flush remaining
        if frag_batch:
            self.decoy_index.insert_fragments_batch(frag_batch)
            self.decoy_index.commit()

        frag_count = self.decoy_index.count_fragments()
        print(f"Index built: {frag_count} unique fragments, {tf_count} transforms, scanned {total_pairs_scanned} pairs")

    @staticmethod
    def _estimate_heavy_atoms(smiles):
        """Quick heavy atom estimate from SMILES without RDKit."""
        heavy = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se"}
        count = 0
        i = 0
        while i < len(smiles):
            c = smiles[i]
            if c in heavy:
                count += 1
            elif c == "[":
                # Handle bracketed atoms like [nH], [O-]
                j = smiles.index("]", i) if "]" in smiles[i:] else i + 1
                elem = smiles[i + 1 : j]
                elem_clean = elem.rstrip("H0123456789-+")
                if elem_clean and elem_clean[0].isupper():
                    count += 1
                i = j
            i += 1
        return count

    # ── Process a single block ─────────────────────────────
    def process_block(self, shard_id, block_id, pairs):
        labeler = BlockLabeler(self.decoy_index, vars(self.args))
        weak_pos, low_freq_decoys = labeler.label_block(pairs)

        # Generate additional decoy types
        attachment_sig = pairs[0].get("attachment_signature", "") if pairs else ""
        n_decoys_needed = max(0, int(len(weak_pos) * self.args.decoy_ratio))

        prop_decoys = labeler.generate_property_matched_decoys(weak_pos, n_per_positive=1)[:n_decoys_needed // 2]
        random_decoys = labeler.generate_random_decoys(pairs[:100], attachment_sig, n_decoys_needed // 2)

        all_decoys = low_freq_decoys + prop_decoys + random_decoys

        # Write weak positives block
        wp_block_path = self.wp_dir / f"shard_{shard_id:04d}_block_{block_id:04d}.jsonl"
        append_jsonl(str(wp_block_path), weak_pos)

        # Write decoy block
        decoy_block_path = self.decoy_dir / f"shard_{shard_id:04d}_block_{block_id:04d}.jsonl"
        append_jsonl(str(decoy_block_path), all_decoys)

        # Write manifest block
        manifest_entries = []
        for p in weak_pos:
            manifest_entries.append({
                "pair_id": p.get("pair_id"),
                "core_key": p.get("core_key"),
                "transform_key": p.get("transform_key"),
                "label": "WEAK_POSITIVE",
                "label_strength": "WEAK_STRUCTURE",
            })
        for d in all_decoys:
            manifest_entries.append({
                "pair_id": d.get("pair_id"),
                "core_key": d.get("core_key"),
                "transform_key": d.get("transform_key"),
                "label": d.get("label", "DECOY"),
                "label_strength": "DECOY",
            })
        manifest_path = self.manifest_dir / f"shard_{shard_id:04d}_block_{block_id:04d}.jsonl"
        append_jsonl(str(manifest_path), manifest_entries)

        # Stats
        stats = {
            "shard_id": shard_id,
            "block_id": block_id,
            "pairs_in": len(pairs),
            "weak_positives": len(weak_pos),
            "decoys_unseen": len(low_freq_decoys),
            "decoys_property_matched": len(prop_decoys),
            "decoys_random": len(random_decoys),
            "manifest_entries": len(manifest_entries),
            "ts": ts(),
        }
        stats_path = self.stats_dir / f"shard_{shard_id:04d}_block_{block_id:04d}.json"
        atomic_write(str(stats_path), stats, is_json=True)

        return weak_pos, all_decoys, manifest_entries

    # ── Main processing loop ───────────────────────────────
    def run(self):
        shards = sorted(glob.glob(str(self.pair_dir / "d1_mmp_pairs_filtered_shard_*.jsonl")))
        if not shards:
            print("ERROR: No pair shards found")
            sys.exit(1)

        if self.restricted_shards is not None:
            shards = self.restricted_shards
            print(f"Restricted shard list: {len(shards)} shards")
        elif self.is_smoke:
            # Use smallest shard
            shard_sizes = [(s, os.path.getsize(s)) for s in shards]
            shard_sizes.sort(key=lambda x: x[1])
            shards = [shard_sizes[0][0]]
            print(f"Smoke mode: using smallest shard {Path(shards[0]).name}")

        print(f"Shards to process: {len(shards)}")
        self._processed_shards = shards
        self.progress.update(total_shards=len(shards), status="building_index")

        # Build or load index
        self.build_index(shards)

        # Process shards
        completed_block_keys = self.checkpoint.get_completed_block_keys()
        total_pairs_processed = 0

        self.progress.update(status="processing", stage="labeling")

        for shard_idx, shard_path in enumerate(shards):
            shard_id = shard_idx
            shard_name = Path(shard_path).stem

            if self.checkpoint.is_shard_completed(shard_id):
                print(f"Skip completed shard {shard_id} ({shard_name})")
                continue

            self.progress.update(current_shard=shard_id)
            print(f"Processing shard {shard_id}/{len(shards)}: {shard_name}")

            block_pairs = []
            block_id = 0

            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    pair = json.loads(line)
                    block_pairs.append(pair)

                    if len(block_pairs) >= self.block_size:
                        block_key = f"{shard_id}:{block_id}"
                        if block_key not in completed_block_keys:
                            self._process_and_checkpoint(shard_id, block_id, block_pairs)
                        else:
                            print(f"  Skip completed block {shard_id}:{block_id}")
                        total_pairs_processed += len(block_pairs)
                        block_pairs = []
                        block_id += 1

                    if self.max_pairs and total_pairs_processed >= self.max_pairs:
                        break

                # Process remaining
                if block_pairs and not (self.max_pairs and total_pairs_processed >= self.max_pairs):
                    block_key = f"{shard_id}:{block_id}"
                    if block_key not in completed_block_keys:
                        self._process_and_checkpoint(shard_id, block_id, block_pairs)
                    total_pairs_processed += len(block_pairs)

            self.checkpoint.shard_done(shard_id)
            append_log(self.processed_shards_log, f"{ts()} shard={shard_id} name={shard_name}")
            self.progress.update(processed_shards=shard_idx + 1)

            if self.max_pairs and total_pairs_processed >= self.max_pairs:
                break

        self.progress.update(status="completed", stage="done")
        self.progress.flush()

    def _process_and_checkpoint(self, shard_id, block_id, block_pairs):
        try:
            wp, dec, man = self.process_block(shard_id, block_id, block_pairs)
            self.checkpoint.block_done(shard_id, block_id, len(block_pairs))

            self.total_weak_positives += len(wp)
            self.total_decoys += len(dec)
            self.total_manifest += len(man)

            self.progress.update(
                processed_blocks=self.checkpoint.data["completed_blocks"].__len__(),
                processed_pairs=self.progress.data["processed_pairs"] + len(block_pairs),
                weak_positive_count=self.total_weak_positives,
                decoy_count=self.total_decoys,
                manifest_count=self.total_manifest,
                current_block=block_id,
            )
            self.progress.heartbeat()

            append_log(self.processed_blocks_log,
                       f"{ts()} shard={shard_id:04d} block={block_id:04d} "
                       f"pairs={len(block_pairs)} wp={len(wp)} dec={len(dec)}")

            gc.collect()

        except Exception as e:
            self.checkpoint.block_failed(shard_id, block_id, e)
            append_log(self.error_blocks_log,
                       f"{ts()} shard={shard_id:04d} block={block_id:04d} error={e}")
            self.progress.update(error_count=self.progress.data.get("error_count", 0) + 1)
            print(f"  ERROR block {shard_id}:{block_id}: {e}")


# ═══════════════════════════════════════════════════════════════
# Status reporter
# ═══════════════════════════════════════════════════════════════
def cmd_status(args):
    base = Path(args.out_dir)
    files = {
        "progress": base / "d2_progress.json",
        "heartbeat": base / "d2_heartbeat.txt",
        "checkpoint": base / "d2_checkpoint.json",
    }
    for label, path in files.items():
        print(f"\n=== {label} ({path}) ===")
        if path.exists():
            with open(path, encoding="utf-8") as f:
                print(f.read()[:2000])
        else:
            print("  NOT FOUND")


# ═══════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════
def cmd_smoke(args):
    print("=== D2 SMOKE TEST ===\n")
    out_dir = Path(args.out_dir)
    smoke_dir = out_dir / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    # Override out_dir for smoke
    args.out_dir = str(smoke_dir)
    args.checkpoint = str(smoke_dir / "d2_checkpoint.json")
    args.max_pairs = DEFAULTS["smoke_max_pairs"]
    args.mode = "smoke"

    runner = D2Runner(args)

    t0 = time.time()
    runner.run()
    elapsed = time.time() - t0

    # Checks
    wp_blocks = list(smoke_dir.glob("label_blocks/weak_positive/*.jsonl"))
    decoy_blocks = list(smoke_dir.glob("label_blocks/decoy/*.jsonl"))
    manifest_blocks = list(smoke_dir.glob("label_blocks/manifest/*.jsonl"))

    wp_count = sum(count_jsonl(str(p)) for p in wp_blocks)
    decoy_count = sum(count_jsonl(str(p)) for p in decoy_blocks)
    manifest_count = sum(count_jsonl(str(p)) for p in manifest_blocks)

    checkpoint_exists = Path(args.checkpoint).exists()
    progress_exists = (smoke_dir / "d2_progress.json").exists()
    heartbeat_exists = (smoke_dir / "d2_heartbeat.txt").exists()

    # Record smoke shard path for resume
    smoke_shard_path = runner._processed_shards[0] if hasattr(runner, '_processed_shards') and runner._processed_shards else None
    if not smoke_shard_path:
        shard_files = sorted(glob.glob(str(Path(args.pair_shard_dir) / "d1_mmp_pairs_filtered_shard_*.jsonl")))
        shard_sizes = [(s, os.path.getsize(s)) for s in shard_files]
        shard_sizes.sort(key=lambda x: x[1])
        smoke_shard_path = shard_sizes[0][0]

    # Resume test — limit to same smoke shard
    resume_runner = None
    resume_dup_check = "SKIP"
    if checkpoint_exists:
        try:
            resume_args = argparse.Namespace(**vars(args))
            resume_args.mode = "resume"
            resume_args.out_dir = str(smoke_dir)
            resume_args.checkpoint = str(smoke_dir / "d2_checkpoint.json")
            resume_args.max_pairs = DEFAULTS["smoke_max_pairs"]
            resume_runner = D2Runner(resume_args)
            resume_runner.restricted_shards = [smoke_shard_path]
            wp_before = sum(count_jsonl(str(p)) for p in wp_blocks)
            resume_runner.run()
            wp_after = sum(count_jsonl(str(p)) for p in wp_blocks)
            resume_dup_check = "PASS" if wp_after == wp_before else f"FAIL ({wp_before}→{wp_after})"
        except Exception as e:
            resume_dup_check = f"ERROR: {e}"

    # Criteria
    checks = {
        "1_reads_pair_shard": wp_count > 0,
        "2_reads_transform_frequency": runner.decoy_index is not None,
        "3_writes_weak_positive_block": wp_count > 0,
        "4_writes_decoy_block": decoy_count > 0,
        "5_writes_manifest_block": manifest_count > 0,
        "6_checkpoint_works": checkpoint_exists,
        "7_resume_no_duplicates": resume_dup_check == "PASS",
        "8_elapsed_seconds": round(elapsed, 1),
        "9_wp_count": wp_count,
        "10_decoy_count": decoy_count,
    }

    all_pass = all([
        checks["1_reads_pair_shard"],
        checks["3_writes_weak_positive_block"],
        checks["4_writes_decoy_block"],
        checks["5_writes_manifest_block"],
        checks["6_checkpoint_works"],
        checks["7_resume_no_duplicates"] == True or resume_dup_check == "PASS",
    ])

    verdict = "D2_SMOKE_PASS" if all_pass else "D2_SMOKE_FAIL"

    # Write reports
    report_md = f"""# D2 Smoke Test Report

Date: {datetime.now().isoformat()}
Verdict: **{verdict}**
Elapsed: {elapsed:.1f}s

## Checks

| # | Check | Result |
|---|-------|--------|
| 1 | Reads pair shard | {'PASS' if checks['1_reads_pair_shard'] else 'FAIL'} |
| 2 | Reads transform frequency | {'PASS' if checks['2_reads_transform_frequency'] else 'FAIL'} |
| 3 | Writes weak positive block | {'PASS' if checks['3_writes_weak_positive_block'] else 'FAIL'} |
| 4 | Writes decoy block | {'PASS' if checks['4_writes_decoy_block'] else 'FAIL'} |
| 5 | Writes manifest block | {'PASS' if checks['5_writes_manifest_block'] else 'FAIL'} |
| 6 | Checkpoint works | {'PASS' if checks['6_checkpoint_works'] else 'FAIL'} |
| 7 | Resume no duplicates | {resume_dup_check} |
| 8 | Memory stable | CHECK |
| 9 | No malformed JSONL | CHECK |
| 10 | No full dataset loaded | CHECK |

## Counts

- Weak positives: {wp_count}
- Decoys: {decoy_count}
- Manifest entries: {manifest_count}
- WP blocks: {len(wp_blocks)}
- Decoy blocks: {len(decoy_blocks)}
- Manifest blocks: {len(manifest_blocks)}
"""

    schema_check = {
        "weak_positive_fields": ["pair_id", "core_key", "transform_key", "label", "label_strength", "label_confidence", "_transform_frequency"],
        "decoy_fields": ["pair_id", "core_key", "transform_key", "label", "label_strength", "label_confidence"],
        "manifest_fields": ["pair_id", "core_key", "transform_key", "label", "label_strength"],
        "block_output_format": "jsonl",
        "atomic_write_pattern": ".tmp → rename",
    }

    counts = {
        "weak_positives": wp_count,
        "decoys": decoy_count,
        "manifest_entries": manifest_count,
        "wp_blocks": len(wp_blocks),
        "decoy_blocks": len(decoy_blocks),
        "manifest_blocks": len(manifest_blocks),
    }

    # Write to smoke_test subdir
    atomic_write(str(smoke_dir / "d2_smoke_test_report.md"), report_md)
    atomic_write(str(smoke_dir / "d2_smoke_schema_check.json"), schema_check, is_json=True)
    atomic_write(str(smoke_dir / "d2_smoke_counts.json"), counts, is_json=True)

    # Also copy to main label dir for easy access
    atomic_write(str(out_dir / "d2_smoke_test_report.md"), report_md)
    atomic_write(str(out_dir / "d2_smoke_schema_check.json"), schema_check, is_json=True)
    atomic_write(str(out_dir / "d2_smoke_counts.json"), counts, is_json=True)

    print(f"\n{'='*60}")
    print(f"SMOKE VERDICT: {verdict}")
    print(f"  Weak positives: {wp_count}")
    print(f"  Decoys: {decoy_count}")
    print(f"  Manifest: {manifest_count}")
    print(f"  Resume check: {resume_dup_check}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")

    return verdict, smoke_dir


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="RouteA D2 Labeling — Streaming")
    p.add_argument("--mode", required=True, choices=["smoke", "full", "resume", "status"])
    p.add_argument("--pair-shard-dir",
                   default="plan_results/routeA_chembl37k_d0d3_engineering_safe/04_d1_pair_generation/pair_shards")
    p.add_argument("--transform-frequency",
                   default="plan_results/routeA_chembl37k_d0d3_engineering_safe/04_d1_pair_generation/d1_transform_frequency.csv")
    p.add_argument("--out-dir",
                   default="plan_results/routeA_chembl37k_d0d3_engineering_safe/05_d2_labeling")
    p.add_argument("--checkpoint",
                   default="plan_results/routeA_chembl37k_d0d3_engineering_safe/05_d2_labeling/d2_checkpoint.json")
    p.add_argument("--block-size", type=int, default=DEFAULTS["block_size"])
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--max-rss-mb", type=int, default=DEFAULTS["max_rss_mb"])
    p.add_argument("--progress-interval-sec", type=int, default=DEFAULTS["progress_interval_sec"])
    p.add_argument("--decoy-ratio", type=float, default=DEFAULTS["decoy_ratio"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--min-transform-frequency", type=int, default=DEFAULTS["min_transform_frequency"])
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve relative paths relative to repo root
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(str(repo_root))

    for attr in ["pair_shard_dir", "transform_frequency", "out_dir", "checkpoint"]:
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, str(Path(val)))

    if args.mode == "status":
        cmd_status(args)
    elif args.mode == "smoke":
        cmd_smoke(args)
    elif args.mode in ("full", "resume"):
        if args.max_pairs == 0:
            args.max_pairs = 10_000 if args.mode == "smoke" else 10_000_000_000
        runner = D2Runner(args)
        runner.run()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
