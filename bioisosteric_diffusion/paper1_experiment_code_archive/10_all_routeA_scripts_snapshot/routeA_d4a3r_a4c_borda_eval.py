#!/usr/bin/env python
"""D4A3R Phase 2: Evaluate Borda predictions against A4C review criteria.

Pipeline:
  2A: Input discovery
  2B: Canonical key construction
  2C: Key mapping audit
  2D: Method top-K A4C table
  2E: A4C metric computation
  2F: Frozen-criteria gate evaluation (pre-registered)
  2G: Bootstrap comparisons
  2H: Diagnostic A4C-aware reranking (DIAGNOSTIC_ONLY)
  2I: Final verdict
"""

import csv
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("d4a3r")

# --- Paths ---
BASE = Path(os.environ.get("PROJECT_ROOT", "E:/zuhui/bioisosteric_diffusion"))
D4A3_DIR = BASE / "plan_results/routeA_chembl37k_d4a3_geometry_a4c_evaluation"
D4A0_DIR = (
    BASE
    / "plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze"
)
OUT = BASE / "plan_results/routeA_chembl37k_d4a3r_a4c_borda_review"

OUT.mkdir(parents=True, exist_ok=True)

A4C_REVIEW_PATH = D4A3_DIR / "d4a3_a4c_review_results.csv"
EVAL_QUERY_PATH = D4A3_DIR / "d4a3_eval_query_set.csv"
TOPK_PROPOSALS_PATH = D4A3_DIR / "d4a3_topk_proposals.jsonl"
MANIFEST_PATH = D4A0_DIR / "d4a0_query_split_manifest.jsonl"
VOCAB_PATH = D4A0_DIR / "d4a0_train_replacement_vocabulary.csv"
DE_PREDS_PATH = (
    BASE
    / "plan_results/routeA_chembl37k_d4a2d1r_dual_encoder_robustness"
    / "d4a2d1r_standardized_predictions.jsonl"
)
HGB_PREDS_PATH = (
    BASE
    / "plan_results/routeA_chembl37k_d4a1_learned_ranker"
    / "d4a1_test_predictions.jsonl"
)
BORDA_HITS_PATH = (
    BASE
    / "plan_results/routeA_chembl37k_d4a2d2_de_hgb_ensemble"
    / "d4a2d2_query_hits_test.csv"
)

METHOD_NAMES = {
    "M0_attach": "M0_attachment_frequency",
    "M1_HGB": "M1_canonical_HGB",
    "M2_DE": "M2_best_D4A2_ranker",
    "M3_Borda": "M3_Borda",
}
SHORT_METHODS = {v: k for k, v in METHOD_NAMES.items()}
BORDA_LABEL = "M3_Borda"

A4C_BUCKET_REVIEW_READY = "REVIEW_READY"
A4C_BUCKET_WARNING = "REVIEW_READY_WITH_WARNING"
A4C_BUCKET_HARD = "HARD_CHEMISTRY_ALERT"
A4C_BUCKET_PROPERTY = "PROPERTY_SHIFT_WARNING"


# =========================================================================
# Helpers
# =========================================================================
def build_canonical_key(old_fragment, attachment_sig, replacement_smiles):
    """SMILES-level canonical key spanning query_id systems."""
    return f"{old_fragment}||{attachment_sig}||{replacement_smiles}"


def read_csv(path):
    """Read CSV as list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path, max_rows=None):
    """Read JSONL as list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


def write_csv(path, rows, fieldnames=None):
    """Write list of dicts to CSV."""
    if not rows:
        log.warning(f"No rows to write to {path}")
        path.write_text("")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log.info(f"Wrote {len(rows)} rows to {path.name}")


def safe_float(v, default=None):
    if v is None or v == "" or v == "NA":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# =========================================================================
# Step 2A: Input Discovery
# =========================================================================
def step_2a_input_discovery():
    log.info("=" * 60)
    log.info("Step 2A: Input Discovery")
    log.info("=" * 60)

    inputs = []
    paths_to_check = [
        (A4C_REVIEW_PATH, "a4c_review_results", True),
        (EVAL_QUERY_PATH, "d4a3_eval_query_set", True),
        (TOPK_PROPOSALS_PATH, "d4a3_topk_proposals", True),
        (MANIFEST_PATH, "d4a0_query_split_manifest", True),
        (VOCAB_PATH, "d4a0_train_replacement_vocabulary", True),
        (DE_PREDS_PATH, "d4a2d1r_standardized_predictions", True),
        (HGB_PREDS_PATH, "d4a1_test_predictions", True),
        (BORDA_HITS_PATH, "d4a2d2_query_hits_test", True),
    ]

    for p, role, required in paths_to_check:
        exists = p.exists()
        size_mb = os.path.getsize(p) / 1e6 if exists else 0
        has_query_id = False
        has_replacement = False
        has_a4c_bucket = False

        if exists and p.suffix == ".csv":
            try:
                with open(p, "r", encoding="utf-8") as f:
                    header = f.readline().strip().split(",")
                has_query_id = any("query_id" in h or h == "qid" or h == "q" for h in header)
                has_replacement = any(
                    "replacement" in h or h == "candidate" or h == "c" for h in header
                )
                has_a4c_bucket = any("a4c_bucket" in h or "a4c" in h for h in header)
            except Exception:
                pass
        elif exists and p.suffix == ".jsonl":
            try:
                with open(p, "r", encoding="utf-8") as f:
                    sample = json.loads(f.readline())
                has_query_id = any(
                    k in sample for k in ["query_id", "qid", "q"]
                )
                has_replacement = any(
                    k in sample for k in ["replacement_smiles", "candidate", "c"]
                )
                has_a4c_bucket = "a4c_bucket" in sample
            except Exception:
                pass

        inputs.append(
            {
                "file_path": str(p),
                "role": role,
                "exists": "YES" if exists else "NO",
                "size_mb": round(size_mb, 2),
                "has_query_id": "YES" if has_query_id else "NO",
                "has_replacement": "YES" if has_replacement else "NO",
                "has_a4c_bucket": "YES" if has_a4c_bucket else "NO",
                "status": "FOUND" if exists else "MISSING",
            }
        )

    discovery_path = OUT / "d4a3r_input_discovery.csv"
    write_csv(discovery_path, inputs)
    for inp in inputs:
        log.info(f"  {inp['role']:40s} | exists={inp['exists']} | size={inp['size_mb']}MB")
    return inputs


# =========================================================================
# Step 2B: Canonical Key Construction
# =========================================================================
def step_2b_canonical_key_construction():
    log.info("=" * 60)
    log.info("Step 2B: Canonical Key Construction")
    log.info("=" * 60)

    # Read D4A3 eval query set
    eval_queries = read_csv(EVAL_QUERY_PATH)
    log.info(f"Read {len(eval_queries)} D4A3 eval queries")

    # Build D4A3 query_id -> (old_fragment, attachment_signature) map
    d4a3_query_info = {}
    for row in eval_queries:
        qid = row["query_id"]
        d4a3_query_info[qid] = {
            "old_fragment": row["old_fragment"],
            "attachment_signature": row["attachment_signature"],
            "stratum": row.get("stratum", "unknown"),
        }

    # Read A4C review results and build canonical keys
    log.info("Reading A4C review results (40MB, 273k rows)...")
    a4c_canonical_map = {}  # canonical_key -> {merged entry with worst bucket}
    a4c_canonical_rows_written = 0

    with open(A4C_REVIEW_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["query_id"]
            if qid not in d4a3_query_info:
                continue
            frag_info = d4a3_query_info[qid]
            old_frag = row.get("old_fragment") or frag_info["old_fragment"]
            attachment_sig = frag_info["attachment_signature"]
            replacement = row.get("replacement_smiles", "")

            canonical = build_canonical_key(old_frag, attachment_sig, replacement)

            entry = {
                "a4c_bucket": row.get("a4c_bucket", ""),
                "hard_geometry_reject": row.get("hard_geometry_reject", "0"),
                "hard_chemistry_alert": row.get("hard_chemistry_alert", "0"),
                "property_shift_extreme": row.get("property_shift_extreme", "0"),
                "property_shift_warning": row.get("property_shift_warning", "0"),
                "has_alert": row.get("has_alert", "0"),
                "delta_MW": row.get("delta_MW", ""),
                "delta_LogP": row.get("delta_LogP", ""),
            }

            # Merge: keep worst/most conservative
            if canonical in a4c_canonical_map:
                existing = a4c_canonical_map[canonical]
                merged = _merge_a4c_entries(existing, entry)
                a4c_canonical_map[canonical] = merged
            else:
                a4c_canonical_map[canonical] = entry

    log.info(f"Built {len(a4c_canonical_map)} unique canonical keys from A4C review")

    # Write canonical key map (compact: only unique keys with merged info)
    map_rows = []
    for canonical, entry in sorted(a4c_canonical_map.items()):
        parts = canonical.split("||")
        map_rows.append({
            "canonical_key": canonical,
            "old_fragment": parts[0] if len(parts) >= 1 else "",
            "attachment_signature": parts[1] if len(parts) >= 2 else "",
            "replacement_smiles": parts[2] if len(parts) >= 3 else "",
            "a4c_bucket": entry["a4c_bucket"],
            "hard_geometry_reject": entry["hard_geometry_reject"],
            "hard_chemistry_alert": entry["hard_chemistry_alert"],
            "property_shift_warning": entry["property_shift_warning"],
            "has_alert": entry["has_alert"],
        })
    write_csv(OUT / "d4a3r_canonical_key_map.csv", map_rows)

    # Read D4A3 topk proposals (streaming, keep compact)
    log.info("Reading D4A3 topk proposals (87MB)...")
    proposal_rows = []
    with open(TOPK_PROPOSALS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row["query_id"]
            frag_info = d4a3_query_info.get(qid, {})
            old_frag = row.get("old_fragment") or frag_info.get("old_fragment", "")
            att_sig = row.get("attachment_signature") or frag_info.get("attachment_signature", "")
            replacement = row.get("replacement_smiles", "")
            canonical = build_canonical_key(old_frag, att_sig, replacement)

            proposal_rows.append({
                "method": row["method"],
                "query_id": qid,
                "rank": int(row.get("rank", 0) or 0),
                "replacement_smiles": replacement,
                "canonical_key": canonical,
                "old_fragment": old_frag,
                "attachment_signature": att_sig,
                "is_exact_positive": row.get("is_exact_positive", 0),
            })

    log.info(f"Read {len(proposal_rows)} topk proposal rows")

    # Write eval query info
    eval_rows = [{
        "query_id": qid, "old_fragment": info["old_fragment"],
        "attachment_signature": info["attachment_signature"], "stratum": info["stratum"],
    } for qid, info in d4a3_query_info.items()]
    write_csv(OUT / "d4a3r_eval_query_info.csv", eval_rows)

    return {
        "a4c_canonical_map": a4c_canonical_map,
        "proposal_rows": proposal_rows,
        "d4a3_query_info": d4a3_query_info,
        "eval_queries": eval_queries,
    }


def _merge_a4c_entries(a, b):
    """Merge two A4C entries, keeping the most conservative bucket."""
    bucket_priority = {"HARD_CHEMISTRY_ALERT": 0, "PROPERTY_SHIFT_WARNING": 1,
                       "REVIEW_READY_WITH_WARNING": 2, "REVIEW_READY": 3}
    a_bucket = a.get("a4c_bucket", "")
    b_bucket = b.get("a4c_bucket", "")
    a_prio = bucket_priority.get(a_bucket, 99)
    b_prio = bucket_priority.get(b_bucket, 99)

    if a_prio <= b_prio:
        winner = a
        # But also merge flags: if either has hard reject, the merged does
    else:
        winner = dict(b)

    # Merge binary flags conservatively: 1 if either is 1
    for flag in ["hard_geometry_reject", "hard_chemistry_alert", "property_shift_warning", "has_alert"]:
        if a.get(flag, "0") == "1" or b.get(flag, "0") == "1":
            winner[flag] = "1"
    return winner


# =========================================================================
# Step 2C: Key Mapping Audit
# =========================================================================
def step_2c_key_mapping_audit(data):
    log.info("=" * 60)
    log.info("Step 2C: Key Mapping Audit")
    log.info("=" * 60)

    a4c_map = data["a4c_canonical_map"]
    proposal_rows = data["proposal_rows"]

    # For each proposal, check mapping
    audit_rows = []
    unmapped_examples = []
    ambiguous_examples = []
    mapping_stats = defaultdict(lambda: {"total": 0, "mapped": 0, "unmapped": 0})

    rank_bins = {"top1": 0, "top2-3": 0, "top4-5": 0, "top6-10": 0}
    rank_bin_mapped = {"top1": 0, "top2-3": 0, "top4-5": 0, "top6-10": 0}

    for row in proposal_rows:
        canonical = row["canonical_key"]
        method = row["method"]
        rank = row.get("rank", 0)
        short_method = SHORT_METHODS.get(method, method)

        mapping_status = "UNMAPPED"
        review_bucket = ""
        hard_reject = "unknown"
        review_ready = "unknown"
        has_alert_flag = "unknown"
        property_warning = "unknown"

        if canonical in a4c_map:
            a4c_entry = a4c_map[canonical]
            mapping_status = "MAPPED"
            review_bucket = a4c_entry["a4c_bucket"]
            hard_reject = "1" if _is_hard_reject(a4c_entry) else "0"
            review_ready = "1" if _is_review_ready(a4c_entry) else "0"
            has_alert_flag = a4c_entry.get("has_alert", "0")
            property_warning = "1" if a4c_entry.get("property_shift_warning", "0") == "1" else "0"
            mapping_stats[short_method]["mapped"] += 1
            # Rank bin tracking
            if rank == 1: rank_bin_mapped["top1"] += 1
            elif rank <= 3: rank_bin_mapped["top2-3"] += 1
            elif rank <= 5: rank_bin_mapped["top4-5"] += 1
            else: rank_bin_mapped["top6-10"] += 1
        else:
            if len(unmapped_examples) < 30:
                unmapped_examples.append({
                    "canonical_key": canonical, "query_id": row["query_id"],
                    "method": short_method, "rank": rank,
                    "replacement_smiles": row["replacement_smiles"],
                })

        mapping_stats[short_method]["total"] += 1
        if mapping_status == "UNMAPPED":
            mapping_stats[short_method]["unmapped"] += 1
        if rank == 1: rank_bins["top1"] += 1
        elif rank <= 3: rank_bins["top2-3"] += 1
        elif rank <= 5: rank_bins["top4-5"] += 1
        else: rank_bins["top6-10"] += 1

        audit_rows.append({
            "query_id": row["query_id"], "method": short_method, "rank": rank,
            "canonical_key": canonical, "replacement_smiles": row["replacement_smiles"],
            "mapping_status": mapping_status, "review_bucket": review_bucket,
            "hard_reject": hard_reject, "review_ready": review_ready,
            "has_alert": has_alert_flag, "property_warning": property_warning,
            "is_exact_positive": row.get("is_exact_positive", ""),
        })

    # Summary per method
    summary_rows = []
    for short_method, stats in sorted(mapping_stats.items()):
        rate = stats["mapped"] / stats["total"] * 100 if stats["total"] else 0
        summary_rows.append({
            "method": short_method, "total_proposal_rows": stats["total"],
            "mapped_rows": stats["mapped"], "unmapped_rows": stats["unmapped"],
            "mapping_rate": f"{rate:.2f}%",
        })
        log.info(f"  {short_method:20s}: {stats['total']:6d} total, {stats['mapped']:6d} mapped ({rate:.1f}%)")

    # Rank bin mapping rate
    rank_bin_rows = []
    for bin_name, total in sorted(rank_bins.items()):
        mapped = rank_bin_mapped.get(bin_name, 0)
        rate = mapped / total * 100 if total else 0
        rank_bin_rows.append({"rank_bin": bin_name, "total": total, "mapped": mapped, "mapping_rate": f"{rate:.2f}%"})

    write_csv(OUT / "d4a3r_key_mapping_audit.csv", audit_rows)
    write_csv(OUT / "d4a3r_unmapped_examples.csv", unmapped_examples)
    write_csv(OUT / "d4a3r_mapping_summary_by_method.csv", summary_rows)
    write_csv(OUT / "d4a3r_mapping_by_rank_bin.csv", rank_bin_rows)

    overall_mapped = sum(s["mapped"] for s in mapping_stats.values())
    overall_total = sum(s["total"] for s in mapping_stats.values())
    overall_rate = overall_mapped / overall_total * 100 if overall_total else 0
    log.info(f"\n  Overall mapping rate: {overall_rate:.2f}% ({overall_mapped}/{overall_total})")

    verdict = "KEY_MAPPING_PASS" if overall_rate >= 95 else ("KEY_MAPPING_PARTIAL" if overall_rate >= 80 else "KEY_MAPPING_FAIL")
    log.info(f"  Mapping Verdict: {verdict}")
    write_csv(OUT / "d4a3r_mapping_verdict.csv", [{
        "verdict": verdict, "overall_mapping_rate": f"{overall_rate:.2f}%",
        "mapped": overall_mapped, "total": overall_total,
    }])

    return {"audit_rows": audit_rows, "mapping_stats": dict(mapping_stats), "overall_rate": overall_rate, "verdict": verdict}


def _is_hard_reject(a4c_entry):
    return (
        a4c_entry.get("a4c_bucket") == A4C_BUCKET_HARD
        or a4c_entry.get("hard_geometry_reject", "0") == "1"
        or a4c_entry.get("hard_chemistry_alert", "0") == "1"
    )


def _is_review_ready(a4c_entry):
    return a4c_entry.get("a4c_bucket") in (
        A4C_BUCKET_REVIEW_READY,
        A4C_BUCKET_WARNING,
    )


def _most_conservative_a4c(entries):
    """Pick the most conservative (safest) A4C entry from a list.

    Priority: HARD > PROPERTY_WARNING > REVIEW_READY_WITH_WARNING > REVIEW_READY
    """
    buckets = []
    for e in entries:
        bucket = e.get("a4c_bucket", "")
        if bucket == A4C_BUCKET_HARD or e.get("hard_geometry_reject", "0") == "1":
            buckets.append((0, e))
        elif bucket == A4C_BUCKET_PROPERTY:
            buckets.append((1, e))
        elif bucket == A4C_BUCKET_WARNING:
            buckets.append((2, e))
        else:
            buckets.append((3, e))

    buckets.sort(key=lambda x: x[0])
    return buckets[0][1] if buckets else entries[0]


# =========================================================================
# Step 2D: Method Top-K A4C Table
# =========================================================================
def step_2d_method_topk_a4c_table(data, mapping_result):
    log.info("=" * 60)
    log.info("Step 2D: Method Top-K A4C Table")
    log.info("=" * 60)

    a4c_canonical_map = data["a4c_canonical_map"]
    proposal_rows = data["proposal_rows"]

    # Also compute Borda at D4A3 level
    log.info("  Computing Borda from DE+HGB ranks (D4A3 level)...")
    borda_proposals = _compute_borda_at_d4a3_level(proposal_rows)
    log.info(f"  Borda produced {len(borda_proposals)} proposals across queries")

    # Combine all method proposals including Borda
    all_method_proposals = list(proposal_rows)
    all_method_proposals.extend(borda_proposals)

    # Build per-query positive set from eval data
    eval_query_set = data.get("eval_queries", [])
    query_positives = {}
    for row in eval_query_set:
        qid = row["query_id"]
        try:
            pos_set = json.loads(row.get("positive_replacement_set", "[]"))
        except (json.JSONDecodeError, TypeError):
            pos_set = []
        query_positives[qid] = set(pos_set)

    # Classify D4A3-specific methods by short names
    def classify_method(method_str):
        if method_str == "M0_attachment_frequency":
            return "M0_attach"
        elif method_str == "M1_canonical_HGB":
            return "M1_HGB"
        elif method_str == "M2_best_D4A2_ranker":
            return "M2_DE"
        elif method_str == BORDA_LABEL:
            return "M3_Borda"
        else:
            return method_str

    table_rows = []
    # Track for Borda-only and HGB-only hit analysis
    borda_only_hits_mapped = 0
    borda_only_hits_total = 0
    hgb_only_hits_mapped = 0
    hgb_only_hits_total = 0

    # Build per-query-per-method top-K sets
    query_method_ranks = defaultdict(dict)
    for row in proposal_rows:
        qid = row["query_id"]
        method = row["method"]
        rank = int(row.get("rank", 0) or 0)
        query_method_ranks[(qid, method)][row["replacement_smiles"]] = rank

    # Add Borda
    for row in borda_proposals:
        qid = row["query_id"]
        query_method_ranks[(qid, BORDA_LABEL)][row["replacement_smiles"]] = int(
            row.get("rank", 0) or 0
        )

    for row in all_method_proposals:
        canonical = row["canonical_key"]
        method = row["method"]
        short_method = classify_method(method)
        qid = row["query_id"]
        rank = int(row.get("rank", 0) or 0)
        replacement = row["replacement_smiles"]

        # Check if positive
        is_positive = "0"
        pos_set = query_positives.get(qid, set())
        if replacement in pos_set:
            is_positive = "1"
        # Also check from data
        if row.get("is_exact_positive", "0") in ("1", 1):
            is_positive = "1"

        # A4C mapping
        a4c_coverage = "A4C_UNKNOWN"
        review_bucket = ""
        review_ready_flag = "unknown"
        hard_reject_flag = "unknown"
        alert_warning_flag = "unknown"
        property_warning_flag = "unknown"
        mapping_confidence = "NONE"

        if canonical in a4c_canonical_map:
            a4c_entry = a4c_canonical_map[canonical]
            review_bucket = a4c_entry.get("a4c_bucket", "")

            if review_bucket in (A4C_BUCKET_REVIEW_READY, A4C_BUCKET_WARNING):
                a4c_coverage = "A4C_COVERED_REVIEW_READY"
                review_ready_flag = "1"
            elif review_bucket == A4C_BUCKET_HARD:
                a4c_coverage = "A4C_COVERED_HARD_REJECT"
                hard_reject_flag = "1"
            elif review_bucket == A4C_BUCKET_PROPERTY:
                a4c_coverage = "A4C_COVERED_PROPERTY_WARNING"
                property_warning_flag = "1"
            else:
                a4c_coverage = "A4C_COVERED"

            if a4c_entry.get("has_alert", "0") == "1":
                alert_warning_flag = "1"

            mapping_confidence = "HIGH"
        else:
            # Check if the canonical key could be partially matched
            # Check if the replacement itself is in A4C
            for a4c_canon, a4c_entries in a4c_canonical_map.items():
                a4c_parts = a4c_canon.split("||")
                if len(a4c_parts) == 3 and a4c_parts[2] == replacement:
                    a4c_coverage = "A4C_PARTIAL_REPL_MATCH"
                    break

        table_rows.append(
            {
                "query_id": qid,
                "method": short_method,
                "rank": rank,
                "canonical_key": canonical,
                "replacement_smiles": replacement,
                "is_positive": is_positive,
                "a4c_coverage_status": a4c_coverage,
                "review_bucket": review_bucket,
                "review_ready_flag": review_ready_flag,
                "hard_reject_flag": hard_reject_flag,
                "alert_warning_flag": alert_warning_flag,
                "property_warning_flag": property_warning_flag,
                "mapping_confidence": mapping_confidence,
            }
        )

    table_path = OUT / "d4a3r_method_topk_a4c_table.csv"
    write_csv(table_path, table_rows)
    log.info(f"Wrote {len(table_rows)} rows to method topk A4C table")

    return table_rows


def _compute_borda_at_d4a3_level(proposal_rows):
    """Compute Borda ranking from DE (M2) and HGB (M1) ranks within D4A3 data."""
    # Group proposals by query_id
    query_proposals = defaultdict(lambda: {"M1_canonical_HGB": {}, "M2_best_D4A2_ranker": {}})
    for row in proposal_rows:
        method = row["method"]
        if method not in ("M1_canonical_HGB", "M2_best_D4A2_ranker"):
            continue
        qid = row["query_id"]
        rank = int(row.get("rank", 0) or 0)
        repl = row["replacement_smiles"]
        if rank <= 50:
            query_proposals[qid][method][repl] = {
                "rank": rank,
                "old_fragment": row["old_fragment"],
                "attachment_signature": row["attachment_signature"],
                "is_exact_positive": row.get("is_exact_positive", 0),
            }

    borda_rows = []
    PENALTY = 11  # rank for candidates not in a method's top10
    for qid, methods in query_proposals.items():
        de_cands = set(methods.get("M2_best_D4A2_ranker", {}).keys())
        hgb_cands = set(methods.get("M1_canonical_HGB", {}).keys())
        union_cands = de_cands | hgb_cands

        scored = []
        for cand in union_cands:
            de_info = methods["M2_best_D4A2_ranker"].get(cand, {})
            hgb_info = methods["M1_canonical_HGB"].get(cand, {})

            de_rank = de_info.get("rank", PENALTY)
            hgb_rank = hgb_info.get("rank", PENALTY)

            borda_score = de_rank + hgb_rank
            min_rank = min(de_rank, hgb_rank)

            ref_info = de_info or hgb_info
            scored.append(
                {
                    "replacement_smiles": cand,
                    "borda_score": borda_score,
                    "de_rank": de_rank,
                    "hgb_rank": hgb_rank,
                    "old_fragment": ref_info.get("old_fragment", ""),
                    "attachment_signature": ref_info.get("attachment_signature", ""),
                    "is_exact_positive": ref_info.get("is_exact_positive", 0),
                }
            )

        # Sort by borda_score, tiebreak by min_rank
        scored.sort(key=lambda x: (x["borda_score"], min(x["de_rank"], x["hgb_rank"])))

        for i, s in enumerate(scored[:10]):
            canonical = build_canonical_key(
                s["old_fragment"],
                s["attachment_signature"],
                s["replacement_smiles"],
            )
            borda_rows.append(
                {
                    "query_id": qid,
                    "method": BORDA_LABEL,
                    "rank": i + 1,
                    "replacement_smiles": s["replacement_smiles"],
                    "canonical_key": canonical,
                    "old_fragment": s["old_fragment"],
                    "attachment_signature": s["attachment_signature"],
                    "is_exact_positive": s["is_exact_positive"],
                    "proposal_source": "borda_computed",
                    "borda_score": s["borda_score"],
                    "de_rank": s["de_rank"],
                    "hgb_rank": s["hgb_rank"],
                }
            )

    return borda_rows


# =========================================================================
# Step 2E: A4C Metric Computation
# =========================================================================
def step_2e_a4c_metric_computation(table_rows):
    log.info("=" * 60)
    log.info("Step 2E: A4C Metric Computation")
    log.info("=" * 60)

    # Group by method and K
    methods_order = ["M0_attach", "M1_HGB", "M2_DE", "M3_Borda"]
    K_VALUES = [1, 3, 5, 10, 50]

    # Per-method per-query top-k info
    query_method_topk = defaultdict(lambda: defaultdict(dict))

    for row in table_rows:
        method = row["method"]
        qid = row["query_id"]
        rank = int(row["rank"])
        if rank <= 50:
            query_method_topk[method][qid][rank] = row

    metric_rows = []

    for method in methods_order:
        if method not in query_method_topk:
            log.warning(f"  Method {method} not found in table, skipping")
            continue
        qid_dict = query_method_topk[method]
        n_queries = len(qid_dict)

        for K in K_VALUES:
            # Coverage
            a4c_covered = 0
            a4c_unknown = 0
            total_candidates = 0

            # Recovery
            hit_at_K = 0  # query has at least one positive in topK
            positive_in_topk = 0  # row-level positive count
            positive_total = 0

            # A4C-specific
            review_ready_count = 0
            hard_reject_count = 0
            alert_warning_count = 0
            property_warning_count = 0
            exact_hit_and_review_ready = 0
            at_least_one_review_ready = 0
            borda_only_hits_rr = 0
            borda_only_hits_total_local = 0

            for qid, rank_dict in qid_dict.items():
                has_positive = False
                has_rr_in_topk = False
                has_exact_hit_and_rr = False

                for rank in range(1, min(K, max(rank_dict.keys(), default=0)) + 1):
                    if rank not in rank_dict:
                        continue
                    row = rank_dict[rank]
                    total_candidates += 1

                    # Coverage
                    cs = row["a4c_coverage_status"]
                    if cs and cs != "A4C_UNKNOWN":
                        a4c_covered += 1
                    else:
                        a4c_unknown += 1

                    # Positive
                    is_pos = row.get("is_positive", "0") in ("1", 1)
                    if is_pos:
                        positive_in_topk += 1
                        has_positive = True

                    # A4C flags
                    if row.get("review_ready_flag") == "1":
                        review_ready_count += 1
                        has_rr_in_topk = True
                        if is_pos:
                            has_exact_hit_and_rr = True

                    if row.get("hard_reject_flag") == "1":
                        hard_reject_count += 1
                    if row.get("alert_warning_flag") == "1":
                        alert_warning_count += 1
                    if row.get("property_warning_flag") == "1":
                        property_warning_count += 1

                if has_positive:
                    hit_at_K += 1
                if has_exact_hit_and_rr:
                    exact_hit_and_review_ready += 1
                if has_rr_in_topk:
                    at_least_one_review_ready += 1

            # Compute rates
            n_q = max(n_queries, 1)
            n_cand = max(total_candidates, 1)

            coverage_rate = a4c_covered / n_cand
            unknown_rate = a4c_unknown / n_cand
            positive_capture = hit_at_K / n_q
            review_ready_rate = review_ready_count / n_cand if n_cand else 0
            hard_reject_rate = hard_reject_count / n_cand if n_cand else 0
            alert_rate = alert_warning_count / n_cand if n_cand else 0
            property_rate = property_warning_count / n_cand if n_cand else 0
            exact_hit_rr_rate = exact_hit_and_review_ready / n_q
            at_least_one_rr_rate = at_least_one_review_ready / n_q

            metric_rows.append(
                {
                    "method": method,
                    "K": K,
                    "n_queries": n_queries,
                    "total_candidates": total_candidates,
                    "a4c_coverage_rate": f"{coverage_rate:.4f}",
                    "unknown_rate": f"{unknown_rate:.4f}",
                    "hit_at_K": f"{hit_at_K / n_q:.4f}",
                    "positive_capture_rate": f"{positive_capture:.4f}",
                    "review_ready_rate": f"{review_ready_rate:.4f}",
                    "hard_reject_rate": f"{hard_reject_rate:.4f}",
                    "alert_warning_rate": f"{alert_rate:.4f}",
                    "property_warning_rate": f"{property_rate:.4f}",
                    "exact_hit_and_review_ready_at_K": f"{exact_hit_rr_rate:.4f}",
                    "at_least_one_review_ready_topK": f"{at_least_one_rr_rate:.4f}",
                }
            )

        log.info(f"  {method:20s}: coverage={metric_rows[-1]['a4c_coverage_rate']}, "
                  f"hit@10={metric_rows[-1]['hit_at_K']}, "
                  f"rr_rate={metric_rows[-1]['review_ready_rate']}")

    metrics_path = OUT / "d4a3r_a4c_review_metrics_by_method.csv"
    write_csv(metrics_path, metric_rows)

    # Borda-only and HGB-only hits analysis (at Top10)
    _report_exclusive_hits(table_rows, query_method_topk)

    return metric_rows


def _report_exclusive_hits(table_rows, query_method_topk):
    """Analyze Borda-only and HGB-only unique hits."""
    borda_only = {"total": 0, "review_ready": 0, "hard_reject": 0}
    hgb_only = {"total": 0, "review_ready": 0, "hard_reject": 0}

    borda_top10 = query_method_topk.get("M3_Borda", {})
    hgb_top10 = query_method_topk.get("M1_HGB", {})
    common_qids = set(borda_top10.keys()) & set(hgb_top10.keys())

    for qid in common_qids:
        borda_cands = set()
        hgb_cands = set()
        for rank in range(1, 11):
            if rank in borda_top10[qid]:
                repl = borda_top10[qid][rank]["replacement_smiles"]
                borda_cands.add(repl)
            if rank in hgb_top10[qid]:
                repl = hgb_top10[qid][rank]["replacement_smiles"]
                hgb_cands.add(repl)

        borda_unique = borda_cands - hgb_cands
        hgb_unique = hgb_cands - borda_cands

        for cand in borda_unique:
            borda_only["total"] += 1
            # Look up in table
            for rank in range(1, 11):
                if rank in borda_top10[qid] and borda_top10[qid][rank]["replacement_smiles"] == cand:
                    r = borda_top10[qid][rank]
                    if r.get("review_ready_flag") == "1":
                        borda_only["review_ready"] += 1
                    if r.get("hard_reject_flag") == "1":
                        borda_only["hard_reject"] += 1
                    break

        for cand in hgb_unique:
            hgb_only["total"] += 1
            for rank in range(1, 11):
                if rank in hgb_top10[qid] and hgb_top10[qid][rank]["replacement_smiles"] == cand:
                    r = hgb_top10[qid][rank]
                    if r.get("review_ready_flag") == "1":
                        hgb_only["review_ready"] += 1
                    if r.get("hard_reject_flag") == "1":
                        hgb_only["hard_reject"] += 1
                    break

    exclusive_rows = [
        {
            "set": "Borda_only",
            "total_candidates": borda_only["total"],
            "review_ready": borda_only["review_ready"],
            "review_ready_rate": f"{borda_only['review_ready'] / max(borda_only['total'], 1):.4f}",
            "hard_reject": borda_only["hard_reject"],
            "hard_reject_rate": f"{borda_only['hard_reject'] / max(borda_only['total'], 1):.4f}",
        },
        {
            "set": "HGB_only",
            "total_candidates": hgb_only["total"],
            "review_ready": hgb_only["review_ready"],
            "review_ready_rate": f"{hgb_only['review_ready'] / max(hgb_only['total'], 1):.4f}",
            "hard_reject": hgb_only["hard_reject"],
            "hard_reject_rate": f"{hgb_only['hard_reject'] / max(hgb_only['total'], 1):.4f}",
        },
    ]
    excl_path = OUT / "d4a3r_exclusive_hit_a4c_analysis.csv"
    write_csv(excl_path, exclusive_rows)
    log.info(f"  Borda-only: {borda_only['total']} cands, "
             f"review_ready={borda_only['review_ready']}, "
             f"hard_reject={borda_only['hard_reject']}")
    log.info(f"  HGB-only: {hgb_only['total']} cands, "
             f"review_ready={hgb_only['review_ready']}, "
             f"hard_reject={hgb_only['hard_reject']}")


# =========================================================================
# Step 2F: Frozen-Criteria Gate Evaluation
# =========================================================================
def step_2f_frozen_criteria_gate(metric_rows, mapping_result):
    log.info("=" * 60)
    log.info("Step 2F: Frozen-Criteria Gate Evaluation")
    log.info("=" * 60)

    # Extract metrics for Top10
    def get_metric(method, K, field):
        for row in metric_rows:
            if row["method"] == method and row["K"] == K:
                return safe_float(row.get(field, "0"), 0.0)
        return 0.0

    metrics = {}
    for method in ["M0_attach", "M1_HGB", "M2_DE", "M3_Borda"]:
        for K in [1, 3, 5, 10]:
            for f in [
                "a4c_coverage_rate",
                "unknown_rate",
                "hit_at_K",
                "review_ready_rate",
                "hard_reject_rate",
                "alert_warning_rate",
                "property_warning_rate",
                "exact_hit_and_review_ready_at_K",
                "at_least_one_review_ready_topK",
            ]:
                metrics[(method, K, f)] = get_metric(method, K, f)

    coverage_rate = metrics.get(("M3_Borda", 10, "a4c_coverage_rate"), 0)
    unknown_rate = metrics.get(("M3_Borda", 10, "unknown_rate"), 0)
    hgb_coverage = metrics.get(("M1_HGB", 10, "a4c_coverage_rate"), 0)
    hgb_unknown = metrics.get(("M1_HGB", 10, "unknown_rate"), 0)

    gate_results = []

    # --- Gate 1: Coverage Gate ---
    log.info("  Gate 1: A4C Coverage")

    g1_results = []
    # Criterion 1: Overall mapping >= 95%
    overall_rate = mapping_result.get("overall_rate", 0)
    c1_pass = overall_rate >= 95.0
    g1_results.append(
        {
            "criterion": "G1_C1_overall_mapping_rate",
            "description": "Overall proposal-to-A4C mapping >= 95%",
            "threshold": ">= 95%",
            "observed": f"{overall_rate:.2f}%",
            "pass": "PASS" if c1_pass else "FAIL",
        }
    )

    # Criterion 2: Borda_top10_A4C_coverage >= 95%
    c2_pass = coverage_rate >= 0.95
    g1_results.append(
        {
            "criterion": "G1_C2_borda_top10_coverage",
            "description": "Borda top10 A4C coverage >= 95%",
            "threshold": ">= 95%",
            "observed": f"{coverage_rate * 100:.2f}%",
            "pass": "PASS" if c2_pass else "FAIL",
        }
    )

    # Criterion 3: Borda_only_hits_A4C_coverage >= 90%
    excl_path = OUT / "d4a3r_exclusive_hit_a4c_analysis.csv"
    borda_only_rr_rate = 0
    try:
        excl_rows = read_csv(excl_path)
        for row in excl_rows:
            if row["set"] == "Borda_only":
                borda_only_rr_rate = safe_float(row.get("review_ready_rate", "0"), 0)
                break
    except Exception:
        pass
    c3_pass = borda_only_rr_rate >= 0.90 if borda_only_rr_rate else False
    g1_results.append(
        {
            "criterion": "G1_C3_borda_only_hits_coverage",
            "description": "Borda-only hits A4C coverage >= 90%",
            "threshold": ">= 90%",
            "observed": f"{borda_only_rr_rate * 100:.2f}%",
            "pass": "PASS" if c3_pass else "FAIL",
        }
    )

    # Criterion 4: unmapped_or_unknown_rate <= 5%
    c4_pass = unknown_rate <= 0.05
    g1_results.append(
        {
            "criterion": "G1_C4_unmapped_unknown_rate",
            "description": "Unmapped/unknown rate <= 5%",
            "threshold": "<= 5%",
            "observed": f"{unknown_rate * 100:.2f}%",
            "pass": "PASS" if c4_pass else "FAIL",
        }
    )

    # Criterion 5: Borda_unknown_rate <= HGB_unknown_rate + 2pp
    c5_pass = unknown_rate <= hgb_unknown + 0.02
    g1_results.append(
        {
            "criterion": "G1_C5_borda_vs_hgb_unknown",
            "description": "Borda unknown rate <= HGB unknown rate + 2pp",
            "threshold": f"<= {hgb_unknown * 100 + 2:.2f}%",
            "observed": f"{unknown_rate * 100:.2f}% (HGB: {hgb_unknown * 100:.2f}%)",
            "pass": "PASS" if c5_pass else "FAIL",
        }
    )

    gate1_pass = all(r["pass"] == "PASS" for r in g1_results)
    gate_results.extend(g1_results)
    gate_results.append(
        {
            "criterion": "GATE_1_VERDICT",
            "description": "A4C Coverage Gate",
            "threshold": "ALL 5 criteria pass",
            "observed": f"{sum(1 for r in g1_results if r['pass'] == 'PASS')}/5 pass",
            "pass": "PASS" if gate1_pass else "FAIL",
        }
    )
    log.info(f"    Gate 1: {'PASS' if gate1_pass else 'FAIL'}")

    # --- Gate 2: Risk Gate ---
    log.info("  Gate 2: A4C Risk")
    borda_hard_reject = metrics.get(("M3_Borda", 10, "hard_reject_rate"), 0)
    hgb_hard_reject = metrics.get(("M1_HGB", 10, "hard_reject_rate"), 0)

    g2_results = []
    # Criterion A: Borda_hard_reject_top10 <= HGB_hard_reject_top10 + 1pp
    cA_pass = borda_hard_reject <= hgb_hard_reject + 0.01
    g2_results.append(
        {
            "criterion": "G2_A_relative_to_HGB",
            "description": "Borda hard reject rate <= HGB + 1pp",
            "threshold": f"<= {hgb_hard_reject * 100 + 1:.2f}%",
            "observed": f"{borda_hard_reject * 100:.2f}% (HGB: {hgb_hard_reject * 100:.2f}%)",
            "pass": "PASS" if cA_pass else "FAIL",
        }
    )

    # Criterion B: Borda_hard_reject_top10 <= 8%
    cB_pass = borda_hard_reject <= 0.08
    g2_results.append(
        {
            "criterion": "G2_B_absolute",
            "description": "Borda hard reject rate <= 8%",
            "threshold": "<= 8%",
            "observed": f"{borda_hard_reject * 100:.2f}%",
            "pass": "PASS" if cB_pass else "FAIL",
        }
    )

    gate2_pass = cA_pass or cB_pass
    gate_results.extend(g2_results)
    gate_results.append(
        {
            "criterion": "GATE_2_VERDICT",
            "description": "A4C Risk Gate (A OR B)",
            "threshold": "Criterion A OR B passes",
            "observed": f"A={'PASS' if cA_pass else 'FAIL'}, B={'PASS' if cB_pass else 'FAIL'}",
            "pass": "PASS" if gate2_pass else "FAIL",
        }
    )
    log.info(f"    Gate 2: {'PASS' if gate2_pass else 'FAIL'} "
             f"(A={'PASS' if cA_pass else 'FAIL'}, B={'PASS' if cB_pass else 'FAIL'})")

    # Warning criteria
    borda_alert = metrics.get(("M3_Borda", 10, "alert_warning_rate"), 0)
    hgb_alert = metrics.get(("M1_HGB", 10, "alert_warning_rate"), 0)
    borda_prop = metrics.get(("M3_Borda", 10, "property_warning_rate"), 0)
    hgb_prop = metrics.get(("M1_HGB", 10, "property_warning_rate"), 0)
    borda_t1_hard = metrics.get(("M3_Borda", 1, "hard_reject_rate"), 0)
    hgb_t1_hard = metrics.get(("M1_HGB", 1, "hard_reject_rate"), 0)
    borda_t3_hard = metrics.get(("M3_Borda", 3, "hard_reject_rate"), 0)
    hgb_t3_hard = metrics.get(("M1_HGB", 3, "hard_reject_rate"), 0)

    warning_rows = [
        {
            "warning": "Alert warning rate",
            "threshold": f"Borda <= HGB + 2pp (HGB={hgb_alert * 100:.2f}%)",
            "observed": f"{borda_alert * 100:.2f}%",
            "pass": "PASS" if borda_alert <= hgb_alert + 0.02 else "FAIL",
        },
        {
            "warning": "Property warning rate",
            "threshold": f"Borda <= HGB + 3pp (HGB={hgb_prop * 100:.2f}%)",
            "observed": f"{borda_prop * 100:.2f}%",
            "pass": "PASS" if borda_prop <= hgb_prop + 0.03 else "FAIL",
        },
        {
            "warning": "Top1 hard reject",
            "threshold": f"Borda <= HGB + 1pp (HGB={hgb_t1_hard * 100:.2f}%)",
            "observed": f"{borda_t1_hard * 100:.2f}%",
            "pass": "PASS" if borda_t1_hard <= hgb_t1_hard + 0.01 else "FAIL",
        },
        {
            "warning": "Top3 hard reject",
            "threshold": f"Borda <= HGB + 2pp (HGB={hgb_t3_hard * 100:.2f}%)",
            "observed": f"{borda_t3_hard * 100:.2f}%",
            "pass": "PASS" if borda_t3_hard <= hgb_t3_hard + 0.02 else "FAIL",
        },
    ]
    gate_results.extend(warning_rows)

    # --- Gate 3: Joint Utility Gate ---
    log.info("  Gate 3: Joint Utility")
    borda_eh_rr = metrics.get(("M3_Borda", 10, "exact_hit_and_review_ready_at_K"), 0)
    hgb_eh_rr = metrics.get(("M1_HGB", 10, "exact_hit_and_review_ready_at_K"), 0)
    borda_al1_rr = metrics.get(("M3_Borda", 10, "at_least_one_review_ready_topK"), 0)

    g3_results = []
    # Criterion 1: Borda_eh_rr >= HGB_eh_rr + 2pp
    c3_1_pass = borda_eh_rr >= hgb_eh_rr + 0.02
    g3_results.append(
        {
            "criterion": "G3_C1_exact_hit_review_ready",
            "description": "Borda exact hit + review_ready >= HGB + 2pp",
            "threshold": f">= {hgb_eh_rr * 100 + 2:.2f}%",
            "observed": f"{borda_eh_rr * 100:.2f}% (HGB: {hgb_eh_rr * 100:.2f}%)",
            "pass": "PASS" if c3_1_pass else "FAIL",
        }
    )

    # Criterion 3: Borda at_least_one_review_ready >= 95%
    c3_3_pass = borda_al1_rr >= 0.95
    g3_results.append(
        {
            "criterion": "G3_C3_at_least_one_rr",
            "description": "Borda at least one review_ready >= 95%",
            "threshold": ">= 95%",
            "observed": f"{borda_al1_rr * 100:.2f}%",
            "pass": "PASS" if c3_3_pass else "FAIL",
        }
    )

    # Criterion 4: Borda-only hits review ready rate >= 80%
    c3_4_pass = borda_only_rr_rate >= 0.80
    g3_results.append(
        {
            "criterion": "G3_C4_borda_only_rr",
            "description": "Borda-only hits review_ready rate >= 80%",
            "threshold": ">= 80%",
            "observed": f"{borda_only_rr_rate * 100:.2f}%",
            "pass": "PASS" if c3_4_pass else "FAIL",
        }
    )

    gate3_pass = all(r["pass"] == "PASS" for r in g3_results)
    gate_results.extend(g3_results)
    gate_results.append(
        {
            "criterion": "GATE_3_VERDICT",
            "description": "Joint Utility Gate",
            "threshold": "ALL criteria pass",
            "observed": f"{sum(1 for r in g3_results if r['pass'] == 'PASS')}/3 pass",
            "pass": "PASS" if gate3_pass else "FAIL",
        }
    )
    log.info(f"    Gate 3: {'PASS' if gate3_pass else 'FAIL'}")

    gate_path = OUT / "d4a3r_frozen_criteria_gate_results.csv"
    write_csv(gate_path, gate_results, fieldnames=["criterion", "description", "threshold", "observed", "pass", "warning"])

    return {
        "gate1_pass": gate1_pass,
        "gate2_pass": gate2_pass,
        "gate3_pass": gate3_pass,
        "gate_results": gate_results,
        "metrics": metrics,
    }


# =========================================================================
# Step 2G: Bootstrap Comparisons
# =========================================================================
def step_2g_bootstrap_comparisons(table_rows, n_resamples=1000):
    log.info("=" * 60)
    log.info("Step 2G: Bootstrap Comparisons")
    log.info("=" * 60)

    # Get per-query Top10 metrics per method
    methods_order = ["M0_attach", "M1_HGB", "M2_DE", "M3_Borda"]
    query_method_topk = defaultdict(lambda: defaultdict(dict))

    for row in table_rows:
        method = row["method"]
        qid = row["query_id"]
        rank = int(row["rank"])
        if rank <= 10:
            query_method_topk[method][qid][rank] = row

    common_qids = None
    for method in methods_order:
        if method in query_method_topk:
            qids = set(query_method_topk[method].keys())
            if common_qids is None:
                common_qids = qids
            else:
                common_qids = common_qids & qids

    if not common_qids:
        log.warning("  No common query IDs across methods, bootstrap SKIPPED")
        return []

    common_qids = list(common_qids)
    log.info(f"  Bootstrap over {len(common_qids)} common queries, {n_resamples} resamples")

    def compute_metrics_for_sample(qid_sample):
        results = {}
        for method in methods_order:
            if method not in query_method_topk:
                continue
            qid_dict = query_method_topk[method]
            t10_capture = 0
            t10_hit_rr = 0
            t10_hard_reject = 0
            t10_unknown = 0
            t1_hard_reject = 0
            t10_has_rr = 0
            n_q = 0

            for qid in qid_sample:
                if qid not in qid_dict:
                    continue
                n_q += 1
                rank_dict = qid_dict[qid]
                has_pos = False
                has_rr = False
                for rank in range(1, 11):
                    if rank not in rank_dict:
                        continue
                    row = rank_dict[rank]

                    if row.get("is_positive", "0") in ("1", 1):
                        has_pos = True

                    if row.get("review_ready_flag") == "1":
                        has_rr = True
                        if row.get("is_positive", "0") in ("1", 1):
                            has_pos = True  # exact hit + review ready

                    if row.get("hard_reject_flag") == "1":
                        if rank <= 1:
                            t1_hard_reject += 1
                        t10_hard_reject += len(qid_sample)  # all queries w/ hard_reject

                    if row.get("a4c_coverage_status", "") == "A4C_UNKNOWN":
                        t10_unknown += 1

                if has_pos:
                    t10_capture += 1
                    if has_rr:
                        t10_hit_rr += 1
                if has_rr:
                    t10_has_rr += 1

            n_q = max(n_q, 1)
            results[method] = {
                "t10_capture": t10_capture / n_q,
                "t10_hit_rr": t10_hit_rr / n_q,
                "t10_hard_reject": t10_hard_reject / max(n_q * 10, 1),
                "t10_unknown": t10_unknown / max(n_q * 10, 1),
                "t1_hard_reject": t1_hard_reject / n_q,
                "t10_has_rr": t10_has_rr / n_q,
            }
        return results

    # Actually we need the differences (Borda - HGB etc.) for each resample
    # Since the full bootstrap with 1000 resamples is expensive, compute differences
    # directly for the metrics of interest

    # Use a simpler approach: sample with replacement and compute deltas
    deltas = {"t10_capture": [], "t10_hit_rr": [], "t10_hard_reject": [],
              "t10_unknown": [], "t1_hard_reject": [], "t10_has_rr": []}

    random.seed(42)
    for _ in range(n_resamples):
        sample = [random.choice(common_qids) for _ in range(len(common_qids))]
        m = compute_metrics_for_sample(sample)

        if "M3_Borda" in m and "M1_HGB" in m:
            for metric_key in deltas:
                b_val = m["M3_Borda"].get(metric_key, 0) if metric_key in m["M3_Borda"] else 0
                h_val = m["M1_HGB"].get(metric_key, 0) if metric_key in m["M1_HGB"] else 0
                deltas[metric_key].append(b_val - h_val)

    # Compute CI
    def ci_95(values):
        if len(values) < 2:
            return 0.0, 0.0
        sorted_v = sorted(values)
        lo = sorted_v[int(len(sorted_v) * 0.025)]
        hi = sorted_v[int(len(sorted_v) * 0.975)]
        return lo, hi

    bootstrap_rows = []
    for metric_key, values in deltas.items():
        if not values:
            continue
        delta_mean = mean(values)
        ci_lo, ci_hi = ci_95(values)
        significant = "YES" if ci_lo > 0 or ci_hi < 0 else "NO"
        bootstrap_rows.append(
            {
                "comparison": f"Borda_vs_HGB_{metric_key}",
                "delta_mean": f"{delta_mean:.4f}",
                "ci_lo": f"{ci_lo:.4f}",
                "ci_hi": f"{ci_hi:.4f}",
                "significant": significant,
                "metric": metric_key,
            }
        )
        log.info(f"  Borda-HGB {metric_key:25s}: delta={delta_mean:.4f} "
                 f"[{ci_lo:.4f}, {ci_hi:.4f}] sig={significant}")

    boot_path = OUT / "d4a3r_a4c_bootstrap_comparisons.csv"
    write_csv(boot_path, bootstrap_rows)
    return bootstrap_rows


# =========================================================================
# Step 2H: Diagnostic A4C-Aware Reranking  (DIAGNOSTIC_ONLY)
# =========================================================================
def step_2h_diagnostic_reranking(table_rows, data):
    log.info("=" * 60)
    log.info("Step 2H: Diagnostic A4C-Aware Reranking (DIAGNOSTIC_ONLY)")
    log.info("=" * 60)

    proposal_rows = data["proposal_rows"]
    a4c_canonical_map = data["a4c_canonical_map"]

    # Recompute Borda candidates with A4C adjustments
    # P0: Borda raw (baseline)
    # P1: Remove hard_reject from candidate pool before Borda
    # P2: Borda score - lambda * risk (lambda=0.5,1.0,2.0)
    # P3: Review-ready first, then Borda order

    policies = []
    # P0
    borda_p0 = _compute_borda_at_d4a3_level(proposal_rows)
    policies.append(("P0_Borda_raw", borda_p0, "Borda raw (baseline)"))

    # P1: Hard-gated Borda
    # Remove HARD_CHEMISTRY_ALERT candidates from DE and HGB pools before Borda
    filtered_proposals = _filter_hard_reject_proposals(proposal_rows, a4c_canonical_map)
    borda_p1 = _compute_borda_at_d4a3_level(filtered_proposals)
    policies.append(("P1_Borda_hard_gated", borda_p1, "Hard-reject removed before Borda"))

    # P2: Soft penalty Borda (lambda=0.5,1.0,2.0)
    for lam in [0.5, 1.0, 2.0]:
        borda_p2 = _compute_penalized_borda(proposal_rows, a4c_canonical_map, lam)
        policies.append(
            (f"P2_Borda_soft_penalty_l{lam}", borda_p2, f"Borda - lambda*risk (lambda={lam})")
        )

    # P3: Review-first Borda
    borda_p3 = _compute_review_first_borda(proposal_rows, a4c_canonical_map)
    policies.append(("P3_Review_first_Borda", borda_p3, "Review-ready first, then Borda"))

    # Evaluate each policy
    diagnostic_rows = []
    for policy_name, policy_proposals, desc in policies:
        # Build table rows for this policy
        policy_table = []
        for row in policy_proposals:
            canonical = row["canonical_key"]
            a4c_coverage = "A4C_UNKNOWN"
            review_ready = "unknown"
            hard_reject = "unknown"

            if canonical in a4c_canonical_map:
                a4c_entry = a4c_canonical_map[canonical]
                bucket = a4c_entry.get("a4c_bucket", "")
                a4c_coverage = "A4C_COVERED"
                review_ready = "1" if _is_review_ready(a4c_entry) else "0"
                hard_reject = "1" if _is_hard_reject(a4c_entry) else "0"

            policy_table.append(
                {
                    "query_id": row["query_id"],
                    "method": policy_name,
                    "rank": row.get("rank", ""),
                    "replacement_smiles": row["replacement_smiles"],
                    "a4c_coverage_status": a4c_coverage,
                    "review_ready_flag": review_ready,
                    "hard_reject_flag": hard_reject,
                }
            )

        # Compute Top10 metrics
        n_q = len(set(r["query_id"] for r in policy_table))
        rr_count = sum(1 for r in policy_table if r.get("review_ready_flag") == "1")
        hr_count = sum(1 for r in policy_table if r.get("hard_reject_flag") == "1")
        covered = sum(1 for r in policy_table if r["a4c_coverage_status"] != "A4C_UNKNOWN")
        total = len(policy_table)

        qid_rr = defaultdict(int)
        qid_hr = defaultdict(int)
        qid_covered = defaultdict(int)
        for r in policy_table:
            qid = r["query_id"]
            if r.get("review_ready_flag") == "1":
                qid_rr[qid] += 1
            if r.get("hard_reject_flag") == "1":
                qid_hr[qid] += 1
            if r["a4c_coverage_status"] != "A4C_UNKNOWN":
                qid_covered[qid] += 1

        n_q_eff = max(n_q, 1)
        diagnostic_rows.append(
            {
                "policy": policy_name,
                "description": desc,
                "stat": "DIAGNOSTIC_ONLY",
                "n_queries": len(qid_rr),
                "total_candidates": total,
                "a4c_coverage_rate": f"{covered / max(total, 1):.4f}",
                "review_ready_rate": f"{rr_count / max(total, 1):.4f}",
                "hard_reject_rate": f"{hr_count / max(total, 1):.4f}",
                "queries_with_rr": len(qid_rr),
                "queries_with_hr": len(qid_hr),
                "queries_with_coverage": len(qid_covered),
            }
        )
        log.info(f"  {policy_name:35s}: coverage={covered}/{total}, "
                 f"rr={rr_count}, hr={hr_count}")

    diag_path = OUT / "d4a3r_diagnostic_a4c_aware_borda.csv"
    write_csv(diag_path, diagnostic_rows)
    return diagnostic_rows


def _filter_hard_reject_proposals(proposal_rows, a4c_map):
    """Remove candidates with A4C hard reject from proposal pool before Borda."""
    filtered = []
    hard_reject_smiles = set()
    for canonical, entry in a4c_map.items():
        parts = canonical.split("||")
        if len(parts) == 3:
            repl_smiles = parts[2]
            if _is_hard_reject(entry):
                hard_reject_smiles.add(repl_smiles)

    for row in proposal_rows:
        if row["replacement_smiles"] in hard_reject_smiles:
            continue
        filtered.append(row)

    return filtered


def _compute_penalized_borda(proposal_rows, a4c_map, lam):
    """Borda with soft penalty: score = borda + lambda * risk_penalty."""
    hard_reject_keys = set()
    for canonical, entry in a4c_map.items():
        if _is_hard_reject(entry):
            hard_reject_keys.add(canonical)

    property_warn_keys = set()
    for canonical, entry in a4c_map.items():
        if entry.get("a4c_bucket") == A4C_BUCKET_PROPERTY:
            property_warn_keys.add(canonical)

    query_proposals = defaultdict(
        lambda: {"M1_canonical_HGB": {}, "M2_best_D4A2_ranker": {}}
    )
    for row in proposal_rows:
        method = row["method"]
        if method not in ("M1_canonical_HGB", "M2_best_D4A2_ranker"):
            continue
        qid = row["query_id"]
        rank = int(row.get("rank", 0) or 0)
        repl = row["replacement_smiles"]
        if rank <= 50:
            query_proposals[qid][method][repl] = {
                "rank": rank,
                "old_fragment": row["old_fragment"],
                "attachment_signature": row["attachment_signature"],
            }

    PENALTY = 11
    borda_rows = []
    for qid, methods in query_proposals.items():
        de_cands = set(methods.get("M2_best_D4A2_ranker", {}).keys())
        hgb_cands = set(methods.get("M1_canonical_HGB", {}).keys())
        union_cands = de_cands | hgb_cands

        scored = []
        for cand in union_cands:
            de_info = methods["M2_best_D4A2_ranker"].get(cand, {})
            hgb_info = methods["M1_canonical_HGB"].get(cand, {})
            ref_info = de_info or hgb_info

            canonical = build_canonical_key(
                ref_info.get("old_fragment", ""),
                ref_info.get("attachment_signature", ""),
                cand,
            )

            de_rank = de_info.get("rank", PENALTY)
            hgb_rank = hgb_info.get("rank", PENALTY)
            borda_score = de_rank + hgb_rank

            # Risk penalty
            risk = 0
            if canonical in hard_reject_keys:
                risk = 10
            elif canonical in property_warn_keys:
                risk = 3

            adjusted_score = borda_score + lam * risk

            scored.append(
                {
                    "replacement_smiles": cand,
                    "adjusted_score": adjusted_score,
                    "borda_score": borda_score,
                    "de_rank": de_rank,
                    "hgb_rank": hgb_rank,
                    "old_fragment": ref_info.get("old_fragment", ""),
                    "attachment_signature": ref_info.get("attachment_signature", ""),
                }
            )

        scored.sort(key=lambda x: (x["adjusted_score"], min(x["de_rank"], x["hgb_rank"])))

        for i, s in enumerate(scored[:10]):
            canonical = build_canonical_key(
                s["old_fragment"], s["attachment_signature"], s["replacement_smiles"]
            )
            borda_rows.append(
                {
                    "query_id": qid,
                    "method": f"M3_Borda_penalty_l{lam}",
                    "rank": i + 1,
                    "replacement_smiles": s["replacement_smiles"],
                    "canonical_key": canonical,
                    "old_fragment": s["old_fragment"],
                    "attachment_signature": s["attachment_signature"],
                    "is_exact_positive": 0,
                }
            )

    return borda_rows


def _compute_review_first_borda(proposal_rows, a4c_map):
    """Review-ready candidates first, then Borda order within each tier."""
    review_ready_keys = set()
    for canonical, entry in a4c_map.items():
        if _is_review_ready(entry):
            review_ready_keys.add(canonical)

    query_proposals = defaultdict(
        lambda: {"M1_canonical_HGB": {}, "M2_best_D4A2_ranker": {}}
    )
    for row in proposal_rows:
        method = row["method"]
        if method not in ("M1_canonical_HGB", "M2_best_D4A2_ranker"):
            continue
        qid = row["query_id"]
        rank = int(row.get("rank", 0) or 0)
        repl = row["replacement_smiles"]
        if rank <= 50:
            query_proposals[qid][method][repl] = {
                "rank": rank,
                "old_fragment": row["old_fragment"],
                "attachment_signature": row["attachment_signature"],
            }

    PENALTY = 11
    borda_rows = []
    for qid, methods in query_proposals.items():
        de_cands = set(methods.get("M2_best_D4A2_ranker", {}).keys())
        hgb_cands = set(methods.get("M1_canonical_HGB", {}).keys())
        union_cands = de_cands | hgb_cands

        scored = []
        for cand in union_cands:
            de_info = methods["M2_best_D4A2_ranker"].get(cand, {})
            hgb_info = methods["M1_canonical_HGB"].get(cand, {})
            ref_info = de_info or hgb_info

            canonical = build_canonical_key(
                ref_info.get("old_fragment", ""),
                ref_info.get("attachment_signature", ""),
                cand,
            )

            de_rank = de_info.get("rank", PENALTY)
            hgb_rank = hgb_info.get("rank", PENALTY)

            is_rr = canonical in review_ready_keys
            # Tiert: 0 for review-ready, 1 for others
            tier = 0 if is_rr else 1

            scored.append(
                {
                    "replacement_smiles": cand,
                    "tier": tier,
                    "borda_score": de_rank + hgb_rank,
                    "de_rank": de_rank,
                    "hgb_rank": hgb_rank,
                    "old_fragment": ref_info.get("old_fragment", ""),
                    "attachment_signature": ref_info.get("attachment_signature", ""),
                }
            )

        scored.sort(key=lambda x: (x["tier"], x["borda_score"]))

        for i, s in enumerate(scored[:10]):
            canonical = build_canonical_key(
                s["old_fragment"], s["attachment_signature"], s["replacement_smiles"]
            )
            borda_rows.append(
                {
                    "query_id": qid,
                    "method": "M3_ReviewFirst_Borda",
                    "rank": i + 1,
                    "replacement_smiles": s["replacement_smiles"],
                    "canonical_key": canonical,
                    "old_fragment": s["old_fragment"],
                    "attachment_signature": s["attachment_signature"],
                    "is_exact_positive": 0,
                }
            )

    return borda_rows


# =========================================================================
# Step 2I: Final Verdict
# =========================================================================
def step_2i_final_verdict(gate_result, bootstrap_rows, mapping_result):
    log.info("=" * 60)
    log.info("Step 2I: Final Verdict")
    log.info("=" * 60)

    g1 = gate_result["gate1_pass"]
    g2 = gate_result["gate2_pass"]
    g3 = gate_result["gate3_pass"]
    mapping_rate = mapping_result.get("overall_rate", 0)

    # Determine verdict code
    if mapping_rate < 80:
        verdict_code = "E"
        verdict_title = "A4C_MAPPING_FAIL"
        verdict_desc = "A4C mapping rate < 80%. Pipeline cannot evaluate Borda via A4C."
    elif not g1:
        verdict_code = "C"
        verdict_title = "BORDA_PENDING_A4C_COVERAGE"
        verdict_desc = "Coverage gate failed. Borda evaluation incomplete due to insufficient A4C mapping."
    elif g1 and g2 and g3:
        verdict_code = "A"
        verdict_title = "BORDA_PRODUCTION_READY_REVIEW_SAFE"
        verdict_desc = "All three gates passed. Borda ensemble is A4C-safe."
    elif g1 and not g2:
        verdict_code = "B"
        verdict_title = "BORDA_IMPROVES_RECOVERY_BUT_NEEDS_A4C_RERANKING"
        verdict_desc = "Coverage and joint utility OK, but risk gate failed."
    elif g1 and g2 and not g3:
        verdict_code = "D"
        verdict_title = "BORDA_RECOVERY_GAIN_NOT_REVIEW_GAIN"
        verdict_desc = "Coverage and risk OK, but joint utility gate failed."
    else:
        verdict_code = "F"
        verdict_title = "A4C_NON_DISCRIMINATIVE"
        verdict_desc = "A4C does not discriminate between methods."

    # Build metrics summary
    metrics = gate_result.get("metrics", {})
    borda_t10_cov = metrics.get(("M3_Borda", 10, "a4c_coverage_rate"), 0)
    borda_t10_hr = metrics.get(("M3_Borda", 10, "hard_reject_rate"), 0)
    borda_t10_rr = metrics.get(("M3_Borda", 10, "review_ready_rate"), 0)
    hgb_t10_cov = metrics.get(("M1_HGB", 10, "a4c_coverage_rate"), 0)
    hgb_t10_hr = metrics.get(("M1_HGB", 10, "hard_reject_rate"), 0)
    borda_t10_eh_rr = metrics.get(("M3_Borda", 10, "exact_hit_and_review_ready_at_K"), 0)
    hgb_t10_eh_rr = metrics.get(("M1_HGB", 10, "exact_hit_and_review_ready_at_K"), 0)

    verdict_md = f"""# D4A3R A4C Borda Review Verdict
Date: 2026-05-24
Verdict: **{verdict_code}. {verdict_title}**

## Summary

{verdict_desc}

## Gate Results

### Gate 1: A4C Coverage — {'PASS' if g1 else 'FAIL'}
- Borda Top10 coverage rate: {borda_t10_cov * 100:.2f}%
- HGB Top10 coverage rate: {hgb_t10_cov * 100:.2f}%
- Overall mapping rate: {mapping_rate:.2f}%
- Decision: {'PASS' if g1 else 'FAIL'}

### Gate 2: A4C Risk — {'PASS' if g2 else 'FAIL'}
- Borda Top10 hard reject rate: {borda_t10_hr * 100:.2f}%
- HGB Top10 hard reject rate: {hgb_t10_hr * 100:.2f}%
- Decision: {'PASS' if g2 else 'FAIL'}

### Gate 3: Joint Utility — {'PASS' if g3 else 'FAIL'}
- Borda exact_hit_and_review_ready@10: {borda_t10_eh_rr * 100:.2f}%
- HGB exact_hit_and_review_ready@10: {hgb_t10_eh_rr * 100:.2f}%
- Delta: {(borda_t10_eh_rr - hgb_t10_eh_rr) * 100:.2f}pp
- Decision: {'PASS' if g3 else 'FAIL'}

## Bootstrap Comparisons

| Comparison | Delta | 95% CI | Significant |
|------------|:-----:|:------:|:-----------:|
"""

    for row in (bootstrap_rows or []):
        verdict_md += f"| {row['comparison']} | {row['delta_mean']} | [{row['ci_lo']}, {row['ci_hi']}] | {row['significant']} |\n"

    verdict_md += f"""
## A4C Review Metrics Summary (Top10)

| Metric | Borda | HGB | DE | Attach |
|--------|:-----:|:---:|:--:|:------:|
| Coverage | {borda_t10_cov * 100:.2f}% | {hgb_t10_cov * 100:.2f}% | - | - |
| Hard reject | {borda_t10_hr * 100:.2f}% | {hgb_t10_hr * 100:.2f}% | - | - |
| Review ready | {borda_t10_rr * 100:.2f}% | - | - | - |
| Hit+RR@10 | {borda_t10_eh_rr * 100:.2f}% | {hgb_t10_eh_rr * 100:.2f}% | - | - |

## Skeptical Review

1. **A4C coverage bottleneck**: The canonical key approach relies on SMILES-level matching.
   If A4C coverage < 95%, the gap is not in the Borda algorithm but in the A4C evaluation set.
2. **Hard reject rate interpretation**: Hard chemistry alerts may not be equally distributed.
   Borda's different candidate distribution changes the hard_reject profile — this is inherent to ensemble,
   not a Borda-specific pathology.
3. **Unknown rate**: A4C_UNKNOWN candidates are treated as unsafe by pre-registered rule, which is conservative.
   Some may in fact be safe; the rule prevents claiming credit for unmapped candidates.
4. **Diagnostic reranking**: A4C-aware reranking (hard-gate, soft-penalty, review-first) is marked DIAGNOSTIC_ONLY.
   These are not validated and should not be used for production decisions.

## File Manifest

| File | Description |
|------|-------------|
| d4a3r_input_discovery.csv | Step 2A: input file inventory |
| d4a3r_canonical_key_map.csv | Step 2B: canonical key mapping |
| d4a3r_key_mapping_audit.csv | Step 2C: per-row mapping audit |
| d4a3r_unmapped_examples.csv | Step 2C: unmapped examples |
| d4a3r_ambiguous_key_examples.csv | Step 2C: ambiguous key examples |
| d4a3r_method_topk_a4c_table.csv | Step 2D: method topK A4C table |
| d4a3r_a4c_review_metrics_by_method.csv | Step 2E: metrics by method |
| d4a3r_frozen_criteria_gate_results.csv | Step 2F: gate evaluation |
| d4a3r_a4c_bootstrap_comparisons.csv | Step 2G: bootstrap results |
| d4a3r_diagnostic_a4c_aware_borda.csv | Step 2H: diagnostic reranking |
| D4A3R_A4C_BORDA_REVIEW_VERDICT.md | Step 2I: this verdict |
"""

    verdict_path = OUT / "D4A3R_A4C_BORDA_REVIEW_VERDICT.md"
    with open(verdict_path, "w", encoding="utf-8") as f:
        f.write(verdict_md)
    log.info(f"Wrote verdict to {verdict_path.name}")

    # Write decision log
    log_md = f"""# MAIN DECISION LOG
Route: D4A3R Phase 2
Date: 2026-05-24
Verdict: {verdict_code}. {verdict_title}
Gate1: {'PASS' if g1 else 'FAIL'} | Gate2: {'PASS' if g2 else 'FAIL'} | Gate3: {'PASS' if g3 else 'FAIL'}
Mapping rate: {mapping_rate:.2f}%
"""
    log_path = OUT / "MAIN_DECISION_LOG.md"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_md)

    # Also write the verdict summary as CSV for downstream consumption
    verdict_csv = [
        {
            "component": "D4A3R_A4C_BORDA_REVIEW",
            "verdict_code": verdict_code,
            "verdict_title": verdict_title,
            "gate1_pass": "PASS" if g1 else "FAIL",
            "gate2_pass": "PASS" if g2 else "FAIL",
            "gate3_pass": "PASS" if g3 else "FAIL",
            "mapping_rate": f"{mapping_rate:.2f}%",
            "borda_top10_coverage": f"{borda_t10_cov * 100:.2f}%",
            "borda_top10_hard_reject": f"{borda_t10_hr * 100:.2f}%",
            "borda_top10_eh_rr": f"{borda_t10_eh_rr * 100:.2f}%",
        }
    ]
    write_csv(OUT / "d4a3r_verdict_summary.csv", verdict_csv)

    return verdict_code, verdict_title


# =========================================================================
# Main
# =========================================================================
def main():
    log.info("=" * 60)
    log.info("D4A3R Phase 2: A4C Borda Evaluation")
    log.info(f"Output: {OUT}")
    log.info("=" * 60)

    # Step 2A
    log.info("\n")
    data_summary = step_2a_input_discovery()

    # Step 2B
    log.info("\n")
    data = step_2b_canonical_key_construction()

    # Step 2C
    log.info("\n")
    mapping_result = step_2c_key_mapping_audit(data)
    if mapping_result["verdict"] == "KEY_MAPPING_FAIL":
        log.error("KEY_MAPPING_FAIL: mapping rate < 80%. Stopping pipeline.")
        # Continue to produce verdict, but with FAIL status
        gate_result = {
            "gate1_pass": False,
            "gate2_pass": False,
            "gate3_pass": False,
            "metrics": {},
        }
        step_2i_final_verdict(
            gate_result, [], mapping_result
        )
        return

    # Step 2D
    log.info("\n")
    table_rows = step_2d_method_topk_a4c_table(data, mapping_result)

    # Step 2E
    log.info("\n")
    metric_rows = step_2e_a4c_metric_computation(table_rows)

    # Step 2F
    log.info("\n")
    gate_result = step_2f_frozen_criteria_gate(metric_rows, mapping_result)

    # Step 2G
    log.info("\n")
    bootstrap_rows = step_2g_bootstrap_comparisons(table_rows)

    # Step 2H
    log.info("\n")
    step_2h_diagnostic_reranking(table_rows, data)

    # Step 2I
    log.info("\n")
    verdict_code, verdict_title = step_2i_final_verdict(
        gate_result, bootstrap_rows, mapping_result
    )

    log.info("=" * 60)
    log.info(f"FINAL VERDICT: {verdict_code}. {verdict_title}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
