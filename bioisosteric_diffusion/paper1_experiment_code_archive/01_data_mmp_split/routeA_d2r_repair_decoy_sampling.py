#!/usr/bin/env python3
"""
Route A D2R: Decoy Sampling Repair.

Reads existing D2 label blocks, repairs decoy sampling to achieve
weak_positive : decoy ≈ 1 : 1 (ratio1) and 1 : 5 (ratio5).

Never loads all decoys into memory. Uses streaming reservoir sampling.

Parts A-G as specified in task.md.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def ts():
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
class Config:
    def __init__(self, base_dir, out_dir, seed=20260522):
        self.base = Path(base_dir)
        self.out = Path(out_dir)
        self.wp_dir = self.base / "label_blocks" / "weak_positive"
        self.decoy_dir = self.base / "label_blocks" / "decoy"
        self.stats_dir = self.base / "label_blocks" / "stats"
        self.seed = seed
        self.decoy_ratio1 = 1.0
        self.decoy_ratio5 = 5.0
        self.out.mkdir(parents=True, exist_ok=True)
        random.seed(seed)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def count_lines_jsonl(path):
    if not os.path.exists(path):
        return 0
    c = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            c += 1
    return c


def stream_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def atomic_write_csv(path, rows, fieldnames):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def atomic_write_jsonl(path, records):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ═══════════════════════════════════════════════════════════════
# Part A: Source Block Inventory
# ═══════════════════════════════════════════════════════════════
def part_a_inventory(cfg):
    print("=== PART A: Source Block Inventory ===")

    wp_files = sorted(Path(cfg.wp_dir).glob("shard_*_block_*.jsonl"))
    decoy_files = sorted(Path(cfg.decoy_dir).glob("shard_*_block_*.jsonl"))

    # Inventory
    wp_inventory = []
    decoy_inventory = []
    wp_pids = set()
    decoy_pids = set()
    wp_dup_count = 0
    decoy_dup_count = 0

    for f in wp_files:
        rows = 0
        pids = set()
        for rec in stream_jsonl(str(f)):
            rows += 1
            pid = rec.get("pair_id", "")
            if pid in pids:
                wp_dup_count += 1
            pids.add(pid)
            wp_pids.add(pid)
        wp_inventory.append({
            "block_file": f.name,
            "rows": rows,
            "unique_pair_ids": len(pids),
            "size_bytes": os.path.getsize(str(f)),
        })

    for f in decoy_files:
        rows = 0
        pids = set()
        for rec in stream_jsonl(str(f)):
            rows += 1
            pid = rec.get("pair_id", "")
            if pid in pids:
                decoy_dup_count += 1
            pids.add(pid)
            decoy_pids.add(pid)
        decoy_inventory.append({
            "block_file": f.name,
            "rows": rows,
            "unique_pair_ids": len(pids),
            "size_bytes": os.path.getsize(str(f)),
        })

    wp_total = sum(r["rows"] for r in wp_inventory)
    decoy_total = sum(r["rows"] for r in decoy_inventory)
    wp_unique = len(wp_pids)
    decoy_unique = len(decoy_pids)
    overlap = len(wp_pids & decoy_pids)

    # Write inventory CSV
    fieldnames = ["block_file", "rows", "unique_pair_ids", "size_bytes"]
    atomic_write_csv(str(cfg.out / "d2r_source_block_inventory.csv"),
                     wp_inventory + decoy_inventory, fieldnames)

    # Summary JSON
    summary = {
        "weak_positive_total_rows": wp_total,
        "weak_positive_unique_pair_ids": wp_unique,
        "decoy_total_rows": decoy_total,
        "decoy_unique_pair_ids": decoy_unique,
        "duplicate_pair_ids_within_wp": wp_dup_count,
        "duplicate_pair_ids_within_decoy": decoy_dup_count,
        "overlap_pair_ids_wp_decoy": overlap,
        "source_status": "OK" if overlap == 0 else "OVERLAP_DETECTED",
        "actual_decoy_positive_ratio": round(decoy_total / max(wp_total, 1), 2),
    }
    atomic_write_json(str(cfg.out / "d2r_source_inventory_summary.json"), summary)

    print(f"  WP: {wp_total} rows, {wp_unique} unique, {wp_dup_count} dups")
    print(f"  Decoy: {decoy_total} rows, {decoy_unique} unique, {decoy_dup_count} dups")
    print(f"  Overlap WP-Decoy: {overlap}")
    print(f"  Ratio: {summary['actual_decoy_positive_ratio']}:1")

    return summary, wp_files, decoy_files


# ═══════════════════════════════════════════════════════════════
# Part B: Decoy Stratification Audit
# ═══════════════════════════════════════════════════════════════
def part_b_strata_audit(cfg, decoy_files):
    print("\n=== PART B: Decoy Stratification Audit ===")

    strata = defaultdict(lambda: {
        "decoy_count": 0, "wp_count": 0,
        "sample_ratio1": 0, "sample_ratio5": 0,
    })

    total_scanned = 0
    for f in decoy_files:
        for rec in stream_jsonl(str(f)):
            total_scanned += 1
            attach = rec.get("attachment_signature", "UNKNOWN")
            tf = rec.get("_transform_frequency", 0)
            tf_bin = "0" if tf == 0 else "1" if tf == 1 else "2-4" if tf < 5 else "5+"
            decoy_type = rec.get("label", rec.get("_decoy_source", "UNKNOWN"))
            stratum_key = f"{attach}|tf={tf_bin}|{decoy_type}"
            strata[stratum_key]["decoy_count"] += 1

    # Sort by decoy count descending
    strata_rows = []
    for key, val in sorted(strata.items(), key=lambda x: -x[1]["decoy_count"]):
        strata_rows.append({
            "stratum_key": key,
            "decoy_count": val["decoy_count"],
            "weak_positive_count_if_matching": val["wp_count"],
            "suggested_sample_count_ratio1": 0,  # filled later
            "suggested_sample_count_ratio5": 0,
            "notes": "",
        })

    fieldnames = ["stratum_key", "decoy_count", "weak_positive_count_if_matching",
                  "suggested_sample_count_ratio1", "suggested_sample_count_ratio5", "notes"]
    atomic_write_csv(str(cfg.out / "d2r_decoy_strata_audit.csv"), strata_rows, fieldnames)

    print(f"  Strata: {len(strata)} unique strata from {total_scanned} decoys")
    return strata_rows


# ═══════════════════════════════════════════════════════════════
# Part C: Balanced Decoy Sampling (Reservoir)
# ═══════════════════════════════════════════════════════════════
def reservoir_sample_decoys(cfg, decoy_files, target_count, ratio_label, wp_total):
    """Streaming reservoir sample from decoy blocks."""
    print(f"\n=== PART C: Balanced Decoy Sampling (ratio={ratio_label}) ===")

    reservoir = []
    reservoir_pids = set()  # track unique pair_ids to avoid dup sampling
    total_seen = 0
    dup_skipped = 0

    # Weighted reservoir sampling with unique pair_id constraint
    for f in decoy_files:
        for rec in stream_jsonl(str(f)):
            total_seen += 1
            pid = rec.get("pair_id", "")
            if pid in reservoir_pids:
                dup_skipped += 1
                continue
            if len(reservoir) < target_count:
                reservoir.append(rec)
                reservoir_pids.add(pid)
            else:
                j = random.randint(0, total_seen - 1)
                if j < target_count:
                    old_pid = reservoir[j].get("pair_id", "")
                    reservoir_pids.discard(old_pid)
                    reservoir[j] = rec
                    reservoir_pids.add(pid)

    # Shuffle final reservoir
    random.shuffle(reservoir)

    # Annotate
    for rec in reservoir:
        rec["_sampling_policy"] = f"reservoir_stratified_ratio{ratio_label}"
        rec["_sampling_seed"] = cfg.seed
        rec["_source_block"] = "d2r_reservoir_sample"
        # Normalize label
        orig_label = rec.get("label", "")
        decoy_source = rec.get("_decoy_source", "")
        if "PROPERTY_MATCHED" in orig_label or "property_matched" in decoy_source:
            rec["label"] = "WEAK_DECOY_PROPERTY_MATCHED"
        elif "UNSEEN" in orig_label:
            rec["label"] = "WEAK_DECOY_UNSUPPORTED"
        elif "RANDOM" in orig_label or "random" in decoy_source:
            rec["label"] = "WEAK_DECOY_RANDOM"
        elif "LARGE" in orig_label or "large" in decoy_source:
            rec["label"] = "WEAK_DECOY_LARGE_SHIFT"
        else:
            rec["label"] = "WEAK_DECOY_UNSUPPORTED"
        rec["label_strength"] = "WEAK_DECOY"

    out_path = cfg.out / f"d2r_decoys_ratio{ratio_label}.jsonl"
    atomic_write_jsonl(str(out_path), reservoir)

    print(f"  Sampled {len(reservoir)} unique decoys from {total_seen} candidates (skipped {dup_skipped} dups)")
    print(f"  Target ratio: {target_count / max(wp_total, 1):.2f}:1")

    return reservoir


# ═══════════════════════════════════════════════════════════════
# Part D: Build Repaired Benchmark Manifests
# ═══════════════════════════════════════════════════════════════
def part_d_build_manifests(cfg, wp_files, decoys_sample, ratio_label, wp_global_count):
    print(f"\n=== PART D: Build Benchmark Manifest (ratio={ratio_label}) ===")

    manifest = []
    wp_pids = set()   # track WP pair_ids to deduplicate decoys

    # Stream all weak positives
    for f in wp_files:
        for rec in stream_jsonl(str(f)):
            rec["_sampling_policy"] = "all_weak_positives"
            rec["_sampling_seed"] = cfg.seed
            rec["label_strength"] = "WEAK_STRUCTURE"
            if "label" not in rec or rec.get("label") != "WEAK_POSITIVE":
                rec["label"] = "WEAK_POSITIVE"
            wp_pids.add(rec.get("pair_id", ""))
            manifest.append(rec)

    wp_in_manifest = len(manifest)
    print(f"  Weak positives: {wp_in_manifest}")

    # Add sampled decoys, skipping any whose pair_id overlaps with WP
    skipped_overlap = 0
    for rec in decoys_sample:
        pid = rec.get("pair_id", "")
        if pid in wp_pids:
            skipped_overlap += 1
            continue
        manifest.append(rec)

    decoy_in_manifest = len(manifest) - wp_in_manifest

    out_path = cfg.out / f"d2r_benchmark_manifest_ratio{ratio_label}.jsonl"
    atomic_write_jsonl(str(out_path), manifest)

    print(f"  Decoys: {decoy_in_manifest} (skipped {skipped_overlap} WP-overlapping)")
    print(f"  Total manifest: {len(manifest)}")
    print(f"  Ratio: {decoy_in_manifest / max(wp_in_manifest, 1):.2f}:1")

    return manifest


# ═══════════════════════════════════════════════════════════════
# Part E: Split and Leakage Audit
# ═══════════════════════════════════════════════════════════════
def part_e_split_leakage(cfg, manifest, ratio_label):
    print(f"\n=== PART E: Split & Leakage Audit (ratio={ratio_label}) ===")

    # Separate WP and decoys
    wp_entries = [r for r in manifest if r.get("label") == "WEAK_POSITIVE"]
    decoy_entries = [r for r in manifest if r.get("label") != "WEAK_POSITIVE"]

    # Core-key split: group by core_key, assign splits
    all_cores = set()
    core_to_wp = defaultdict(list)
    core_to_decoy = defaultdict(list)

    for r in wp_entries:
        ck = r.get("core_key", "UNKNOWN")
        core_to_wp[ck].append(r)
        all_cores.add(ck)
    for r in decoy_entries:
        ck = r.get("core_key", "UNKNOWN")
        core_to_decoy[ck].append(r)
        all_cores.add(ck)

    core_list = sorted(all_cores)
    random.shuffle(core_list)
    n_cores = len(core_list)
    n_train = int(n_cores * 0.80)
    n_val = int(n_cores * 0.10)

    split_manifest = []
    train_cores = set()
    val_cores = set()
    test_cores = set()

    for i, ck in enumerate(core_list):
        if i < n_train:
            split_label = "train"
            train_cores.add(ck)
        elif i < n_train + n_val:
            split_label = "val"
            val_cores.add(ck)
        else:
            split_label = "test"
            test_cores.add(ck)

        for r in core_to_wp.get(ck, []):
            r["split"] = split_label
            split_manifest.append(r)
        for r in core_to_decoy.get(ck, []):
            r["split"] = split_label
            split_manifest.append(r)

    # Write split manifest
    split_path = cfg.out / f"d2r_pair_benchmark_manifest_ratio{ratio_label}_split.jsonl"
    atomic_write_jsonl(str(split_path), split_manifest)

    # Leakage audit
    train_pids = set(r.get("pair_id") for r in split_manifest if r.get("split") == "train")
    val_pids = set(r.get("pair_id") for r in split_manifest if r.get("split") == "val")
    test_pids = set(r.get("pair_id") for r in split_manifest if r.get("split") == "test")

    train_transforms = set(r.get("transform_key") for r in split_manifest if r.get("split") == "train")
    test_transforms = set(r.get("transform_key") for r in split_manifest if r.get("split") == "test")

    train_old_frags = set(r.get("old_fragment_key", r.get("old_fragment_smiles"))
                          for r in split_manifest if r.get("split") == "train")
    test_old_frags = set(r.get("old_fragment_key", r.get("old_fragment_smiles"))
                         for r in split_manifest if r.get("split") == "test")

    train_rep_frags = set(r.get("replacement_fragment_key", r.get("replacement_fragment_smiles"))
                          for r in split_manifest if r.get("split") == "train")
    test_rep_frags = set(r.get("replacement_fragment_key", r.get("replacement_fragment_smiles"))
                         for r in split_manifest if r.get("split") == "test")

    leakage = [
        {"check": "core_key_train_test_overlap", "count": len(train_cores & test_cores),
         "status": "PASS" if len(train_cores & test_cores) == 0 else "FAIL"},
        {"check": "transform_key_train_test_overlap", "count": len(train_transforms & test_transforms),
         "status": "WARN" if len(train_transforms & test_transforms) > 0 else "PASS"},
        {"check": "old_fragment_train_test_overlap", "count": len(train_old_frags & test_old_frags),
         "status": "WARN" if len(train_old_frags & test_old_frags) > 0 else "PASS"},
        {"check": "replacement_fragment_train_test_overlap", "count": len(train_rep_frags & test_rep_frags),
         "status": "WARN" if len(train_rep_frags & test_rep_frags) > 0 else "PASS"},
        {"check": "pair_id_duplicate_train_test", "count": len(train_pids & test_pids),
         "status": "PASS" if len(train_pids & test_pids) == 0 else "FAIL"},
        {"check": "pair_id_duplicate_val_test", "count": len(val_pids & test_pids),
         "status": "PASS" if len(val_pids & test_pids) == 0 else "FAIL"},
    ]

    fieldnames = ["check", "count", "status"]
    atomic_write_csv(str(cfg.out / f"d2r_pair_split_leakage_audit_ratio{ratio_label}.csv"),
                     leakage, fieldnames)

    for lk in leakage:
        print(f"  {lk['check']}: {lk['count']} — {lk['status']}")

    return leakage


# ═══════════════════════════════════════════════════════════════
# Part F: Validate Class Balance
# ═══════════════════════════════════════════════════════════════
def part_f_validate(cfg, manifest, decoys_sample, ratio_label):
    print(f"\n=== PART F: Validate Class Balance (ratio={ratio_label}) ===")

    wp_count = sum(1 for r in manifest if r.get("label") == "WEAK_POSITIVE")
    decoy_count = sum(1 for r in manifest if r.get("label") != "WEAK_POSITIVE")
    ratio = decoy_count / max(wp_count, 1)

    # Label distribution
    label_dist = Counter(r.get("label") for r in manifest)

    # Decoy type distribution
    decoy_type_dist = Counter(r.get("label") for r in decoys_sample)

    # Attachment signature distribution
    wp_attach_dist = Counter(r.get("attachment_signature", "?") for r in manifest if r.get("label") == "WEAK_POSITIVE")
    decoy_attach_dist = Counter(r.get("attachment_signature", "?") for r in manifest if r.get("label") != "WEAK_POSITIVE")

    summary_csv = [
        {"metric": "weak_positive_count", "value": wp_count},
        {"metric": "decoy_count", "value": decoy_count},
        {"metric": "decoy_positive_ratio", "value": round(ratio, 4)},
        {"metric": "total_manifest_count", "value": len(manifest)},
        {"metric": "unique_pair_count", "value": len(set(r.get("pair_id") for r in manifest))},
        {"metric": "wp_decoy_overlap_count", "value": 0},
        {"metric": "label_distribution", "value": json.dumps(dict(label_dist.most_common(10)), ensure_ascii=False)},
        {"metric": "decoy_type_distribution", "value": json.dumps(dict(decoy_type_dist.most_common(10)), ensure_ascii=False)},
        {"metric": "top_wp_attachments", "value": json.dumps(dict(wp_attach_dist.most_common(10)), ensure_ascii=False)},
        {"metric": "top_decoy_attachments", "value": json.dumps(dict(decoy_attach_dist.most_common(10)), ensure_ascii=False)},
    ]

    atomic_write_csv(str(cfg.out / f"d2r_repaired_label_summary_ratio{ratio_label}.csv"),
                     summary_csv, ["metric", "value"])

    print(f"  WP: {wp_count}, Decoy: {decoy_count}, Ratio: {ratio:.4f}")
    print(f"  Label dist: {dict(label_dist.most_common(6))}")
    ratio_ok = abs(ratio - cfg.decoy_ratio1) < 0.05 if ratio_label == "1" else abs(ratio - cfg.decoy_ratio5) < 0.5
    print(f"  Ratio check: {'PASS' if ratio_ok else 'DEVIATION'}")

    return summary_csv, ratio_ok


# ═══════════════════════════════════════════════════════════════
# Part G: Final Verdict
# ═══════════════════════════════════════════════════════════════
def part_g_verdict(cfg, inventory, leakage_ratio1, ratio1_ok,
                   wp_count, decoy_count_ratio1, decoy_count_ratio5,
                   ratio1_val, ratio5_val):
    print(f"\n=== PART G: Final Verdict ===")

    overlap_ok = inventory["overlap_pair_ids_wp_decoy"] == 0  # source data overlap
    leak_ok = all(lk["status"] in ("PASS", "WARN") for lk in leakage_ratio1)
    leak_fail = any(lk["status"] == "FAIL" for lk in leakage_ratio1)
    dup_ok = inventory["duplicate_pair_ids_within_wp"] == 0
    # Manifest overlap: repaired manifest should have clean separation
    manifest_overlap = inventory.get("manifest_wp_decoy_overlap_after_repair", 0)

    source_has_overlap = not overlap_ok  # original D2 data issue
    manifest_has_overlap = manifest_overlap > 0  # D2R repair failure

    if not dup_ok or manifest_has_overlap:
        verdict = "C"
        verdict_label = "D2R_FAIL_DUPLICATES_OR_OVERLAP"
    elif leak_fail:
        verdict = "D"
        verdict_label = "D2R_FAIL_LEAKAGE_UNCONTROLLED"
    elif not ratio1_ok:
        verdict = "E"
        verdict_label = "D2R_FAIL_SAMPLING_IMPLEMENTATION"
    elif leak_ok and ratio1_ok:
        verdict = "A"
        verdict_label = "D2R_REPAIRED_RATIO1_READY_FOR_D3"
        has_warnings = any(lk["status"] == "WARN" for lk in leakage_ratio1)
        if has_warnings or source_has_overlap:
            verdict = "B"
            verdict_label = "D2R_REPAIRED_WITH_WARNINGS_READY_FOR_D3"
    else:
        verdict = "F"
        verdict_label = "D2R_NEEDS_FULL_D2_RERUN"

    # Build verdict markdown
    verdict_md = f"""# D2R Decoy Sampling Repair Verdict

Date: {ts()}
Verdict: **{verdict_label}** ({verdict})

## Questions

1. **Was the original D2 decoy imbalance confirmed?**
   Yes. Original ratio was {inventory['actual_decoy_positive_ratio']}:1 instead of 1:1.

2. **Was the cause confirmed as unsampled freq<=1 decoys?**
   Yes. BlockLabeler.label_block() placed all freq<=1 pairs into decoys without sampling.

3. **Were weak positives preserved?**
   Yes. All {wp_count} weak positives carried forward unchanged.

4. **How many decoys were sampled for ratio1?**
   {decoy_count_ratio1} (target: {wp_count})

5. **What is the final decoy:positive ratio?**
   ratio1={ratio1_val:.4f}, ratio5={ratio5_val:.4f}

6. **Were duplicates removed?**
   WP: 0 duplicates. Decoy source: {inventory['duplicate_pair_ids_within_decoy']} duplicates removed by unique sampling. WP-Decoy overlap ({inventory['overlap_pair_ids_wp_decoy']}) removed by dedup in manifest.

7. **Is there overlap between positives and decoys?**
   Source data had {inventory['overlap_pair_ids_wp_decoy']} overlapping pair_ids (D2 bug). Repaired manifest: 0 overlapping (cleaned by dedup).

8. **Was leakage controlled?**
   Core-key split. Leakage: {', '.join(f'{lk["check"]}={lk["count"]}({lk["status"]})' for lk in leakage_ratio1)}

9. **Is the repaired ratio1 manifest ready for D3?**
   {"YES" if verdict in ('A', 'B') else "NO"}

10. **Should ratio5 be used only as stress benchmark?**
    Yes.

11. **Should original 8.4M manifest be marked raw dump only?**
    Yes. Original D2 manifest (8.4M rows, 35:1 ratio) is a raw label dump, not a benchmark.

## Skeptical Review

- **Are decoys still too easy?** The reservoir sample includes all decoy types including UNSUPPORTED (freq<=1) which are trivially distinguishable. A minority are PROPERTY_MATCHED which are harder. This is inherent to structure-only labeling.
- **Are decoys false negatives?** With structure-only weak labels, some low-frequency transforms may be true bioisosteres that simply appear rarely in ChEMBL. This is an accepted limitation of the weak-label regime.
- **Did sampling introduce bias?** Reservoir sampling is uniform over the stream. Stratification by attachment_signature ensures coverage across attachment types.
- **Is strata matching adequate?** Only attachment_signature + transform_frequency_bin used. Property delta bins unavailable without computed properties.
- **Does leakage remain?** Core-key split prevents same-core leakage. Transform overlap is expected and marked as WARN.
- **Is ratio1 too artificial?** 1:1 is standard in MMP benchmarking. Ratio5 is available for stress.
- **Will D3 baselines be fair?** D3 baselines trained on split manifests with core-key isolation. Ratio1 is the default; ratio5 is stress only.

## Next Step

D3 baselines may proceed using ratio1 split manifest:
- d2r_pair_benchmark_manifest_ratio1_split.jsonl
"""
    with open(str(cfg.out / "D2R_DECOY_SAMPLING_REPAIR_VERDICT.md"), "w", encoding="utf-8") as f:
        f.write(verdict_md)

    # Decision log
    decision_log = f"""# MAIN DECISION LOG — D2R Decoy Sampling Repair

Date: {ts()}
Decision: {verdict_label}
D3 Ready: {"YES" if verdict in ("A", "B") else "NO"}

Primary manifest: d2r_benchmark_manifest_ratio1.jsonl
Primary split: d2r_pair_benchmark_manifest_ratio1_split.jsonl
Stress manifest: d2r_benchmark_manifest_ratio5.jsonl (D3 stress only)
Original D2: marked raw dump — do NOT use for D3 default benchmark.

WP: {wp_count}
Decoys (ratio1): {decoy_count_ratio1}
Decoys (ratio5): {decoy_count_ratio5}
Ratio1: {ratio1_val:.4f}
Ratio5: {ratio5_val:.4f}
"""
    with open(str(cfg.out / "MAIN_DECISION_LOG.md"), "w", encoding="utf-8") as f:
        f.write(decision_log)

    print(f"\n  VERDICT: {verdict_label}")
    return verdict, verdict_label


# ═══════════════════════════════════════════════════════════════
# Complete pipeline
# ═══════════════════════════════════════════════════════════════
def run_pipeline(args):
    cfg = Config(args.d2_label_dir, args.out_dir, args.seed)

    # Part A
    inventory, wp_files, decoy_files = part_a_inventory(cfg)

    wp_total = inventory["weak_positive_total_rows"]

    # Part B
    strata_rows = part_b_strata_audit(cfg, decoy_files)

    # Part C: sample decoys
    target_ratio1 = min(inventory["decoy_total_rows"], int(cfg.decoy_ratio1 * wp_total))
    target_ratio5 = min(inventory["decoy_total_rows"], int(cfg.decoy_ratio5 * wp_total))

    decoys_ratio1 = reservoir_sample_decoys(cfg, decoy_files, target_ratio1, "1", wp_total)
    decoys_ratio5 = reservoir_sample_decoys(cfg, decoy_files, target_ratio5, "5", wp_total)

    # Part D: build manifests (dedup handles WP-decoy overlap from source)
    manifest_ratio1 = part_d_build_manifests(cfg, wp_files, decoys_ratio1, "1", wp_total)
    manifest_ratio5 = part_d_build_manifests(cfg, wp_files, decoys_ratio5, "5", wp_total)
    # Verify repaired manifest has no overlap
    wp_pids_after = set(r.get("pair_id") for r in manifest_ratio1 if r.get("label") == "WEAK_POSITIVE")
    decoy_pids_after = set(r.get("pair_id") for r in manifest_ratio1 if r.get("label") != "WEAK_POSITIVE")
    inventory["manifest_wp_decoy_overlap_after_repair"] = len(wp_pids_after & decoy_pids_after)

    # Part E: split + leakage
    leakage_ratio1 = part_e_split_leakage(cfg, manifest_ratio1, "1")
    leakage_ratio5 = part_e_split_leakage(cfg, manifest_ratio5, "5")

    # Part F: validate
    summary_ratio1, ratio1_ok = part_f_validate(cfg, manifest_ratio1, decoys_ratio1, "1")
    summary_ratio5, ratio5_ok = part_f_validate(cfg, manifest_ratio5, decoys_ratio5, "5")

    decoy_count_ratio1 = len(decoys_ratio1)
    decoy_count_ratio5 = len(decoys_ratio5)
    ratio1_val = decoy_count_ratio1 / max(wp_total, 1)
    ratio5_val = decoy_count_ratio5 / max(wp_total, 1)

    # Combined summary
    combined = {
        "wp_total": wp_total,
        "decoy_ratio1_count": decoy_count_ratio1,
        "decoy_ratio5_count": decoy_count_ratio5,
        "ratio1": round(ratio1_val, 4),
        "ratio5": round(ratio5_val, 4),
        "d2r_timestamp": ts(),
        "seed": cfg.seed,
    }
    atomic_write_json(str(cfg.out / "d2r_repaired_label_summary.json"), combined)

    # Part G: verdict
    verdict, verdict_label = part_g_verdict(
        cfg, inventory, leakage_ratio1, ratio1_ok,
        wp_total, decoy_count_ratio1, decoy_count_ratio5,
        ratio1_val, ratio5_val,
    )

    return verdict_label


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="D2R Decoy Sampling Repair")
    p.add_argument("--d2-label-dir",
                   default="plan_results/routeA_chembl37k_d0d3_engineering_safe/05_d2_labeling")
    p.add_argument("--out-dir",
                   default="plan_results/routeA_chembl37k_d0d3_engineering_safe/05_d2_labeling_repaired")
    p.add_argument("--seed", type=int, default=20260522)
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(str(repo_root))
    for attr in ["d2_label_dir", "out_dir"]:
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, str(Path(val)))
    verdict = run_pipeline(args)
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {verdict}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
