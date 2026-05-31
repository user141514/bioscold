#!/usr/bin/env python3
"""
D4A3S: A4C Coverage Expansion for Borda Gain Region
====================================================
Diagnoses and attempts to repair A4C coverage gap for Borda's gain region.

Borda beats HGB by +4.2pp on full 21,052 test queries.
D4A3R found A4C eval set (6,837 queries) does NOT cover Borda's gain region.
The +4.2pp gain happens on 14,215 queries outside A4C eval set.

Pipeline:
  Part A: Borda Gain Region Definition
  Part B: Join Logic Repair
  Part C: A4C Recompute from SMILES
  Part D: Fragment Graph Gap Analysis
  Part E: Chemical Proxy Screening
  Part F: A4C Re-Evaluation (conditional on coverage >= 95%)
  Part G: Final Verdict

Output: plan_results/routeA_chembl37k_d4a3s_a4c_coverage_expansion/
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

import numpy as np

# ---------------------------------------------------------------------------
# RDKit (optional — Parts C and E)
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Crippen, Descriptors, FilterCatalog
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(os.environ.get("PROJECT_ROOT", "E:/zuhui/bioisosteric_diffusion"))
D4A0_DIR = BASE / "plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze"
D4A1_DIR = BASE / "plan_results/routeA_chembl37k_d4a1_learned_ranker"
D4A2D1R_DIR = BASE / "plan_results/routeA_chembl37k_d4a2d1r_dual_encoder_robustness"
D4A2D2_DIR = BASE / "plan_results/routeA_chembl37k_d4a2d2_de_hgb_ensemble"
D4A3_DIR = BASE / "plan_results/routeA_chembl37k_d4a3_geometry_a4c_evaluation"
D4A3R_DIR = BASE / "plan_results/routeA_chembl37k_d4a3r_a4c_borda_review"
OUT = BASE / "plan_results/routeA_chembl37k_d4a3s_a4c_coverage_expansion"

SEED = 20260523
N_BOOTSTRAP = 1000
TOP_K = 10  # Borda top-K for analysis

np.random.seed(SEED)
random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("d4a3s")

# ---------------------------------------------------------------------------
# A4C bucket constants
# ---------------------------------------------------------------------------
REVIEW_READY = "REVIEW_READY"
REVIEW_WARNING = "REVIEW_READY_WITH_WARNING"
HARD_ALERT = "HARD_CHEMISTRY_ALERT"
PROPERTY_WARNING = "PROPERTY_SHIFT_WARNING"
A4C_RECOMPUTED = "A4C_RECOMPUTED_FROM_SMILES"
PROXY_EVIDENCE = "PROXY_EVIDENCE_NOT_A4C_REVIEW"

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path, fieldnames, rows):
    """Write CSV with header."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    """Write JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_md(path, text):
    """Write markdown text."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# SMILES normalization (cross-system matching)
# ---------------------------------------------------------------------------

def normalize_fragment(smi):
    """
    Normalize a fragment SMILES for cross-system matching.
    - DE format: *c1ccccc1 (strip *, canonicalize) -> c1ccccc1
    - HGB format: *Cc1ccccc1 (strip *C, canonicalize) -> c1ccccc1
    Falls back to sorted-char string for unparseable SMARTS.
    """
    if not smi:
        return smi
    s = smi.lstrip("*")
    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmarts(s)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    # Fallback: collapse whitespace and sort chars
    return "".join(sorted(s.strip()))


def strip_asterisk(smi):
    """Remove leading * for SMILES matching (for A4C cross-reference)."""
    if not smi:
        return smi
    return smi.lstrip("*")


def build_canonical_key(old_fragment, replacement_smiles):
    """Query-agnostic canonical key for A4C cross-reference."""
    return f"{old_fragment}||{replacement_smiles}"


# =========================================================================
# Part A: Borda Gain Region Definition
# =========================================================================

def part_a_borda_gain_region():
    """Define Borda gain groups and analyze A4C coverage."""
    log.info("=" * 60)
    log.info("PART A: Borda Gain Region Definition")
    log.info("=" * 60)

    # 1. Load data
    log.info("Loading DE predictions (M2_DE) ...")
    de_preds = read_jsonl(D4A2D1R_DIR / "d4a2d1r_standardized_predictions.jsonl")
    de_by_q = defaultdict(list)
    for p in de_preds:
        if p["m"] == "M2_DE":
            de_by_q[p["q"]].append(p)
    log.info("  DE: %d queries, %d total predictions", len(de_by_q),
             sum(len(v) for v in de_by_q.values()))

    log.info("Loading HGB predictions ...")
    hgb_preds = read_jsonl(D4A1_DIR / "d4a1_test_predictions.jsonl")
    hgb_by_q = defaultdict(list)
    for p in hgb_preds:
        hgb_by_q[p["query_id"]].append(p)
    log.info("  HGB: %d queries, %d total predictions", len(hgb_by_q),
             sum(len(v) for v in hgb_by_q.values()))

    log.info("Loading D4A0 manifest (query split & positives) ...")
    manifest = read_jsonl(D4A0_DIR / "d4a0_query_split_manifest.jsonl")
    manifest_by_q = {m["query_id"]: m for m in manifest}
    pos_set = {}
    for m in manifest:
        qid = m["query_id"]
        pos_set[qid] = set()
        for ps in m.get("positive_replacement_set", []):
            pos_set[qid].add(normalize_fragment(ps))
    log.info("  Manifest: %d queries", len(manifest))

    log.info("Loading A4C review results (header only for field names) ...")
    with open(D4A3_DIR / "d4a3_a4c_review_results.csv", "r", encoding="utf-8") as f:
        a4c_reader = csv.DictReader(f)
        a4c_fieldnames = a4c_reader.fieldnames
    log.info("  A4C fields: %s", a4c_fieldnames)

    # 2. Build A4C lookup by SMILES pair
    log.info("Building A4C SMILES-level lookup ...")
    a4c_by_smiles = {}  # key: old_fragment||replacement_smiles -> list of A4C rows
    a4c_all_replacements = set()
    # Stream the large CSV to avoid loading all in memory
    with open(D4A3_DIR / "d4a3_a4c_review_results.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = build_canonical_key(row["old_fragment"], row["replacement_smiles"])
            if key not in a4c_by_smiles:
                a4c_by_smiles[key] = []
            a4c_by_smiles[key].append(row)
            a4c_all_replacements.add(strip_asterisk(row["replacement_smiles"]))
    log.info("  A4C SMILES pairs: %d, unique replacement SMILES: %d",
             len(a4c_by_smiles), len(a4c_all_replacements))

    # 3. Shared query set (test queries with both DE and HGB)
    shared_qids = sorted(set(de_by_q.keys()) & set(hgb_by_q.keys()) & set(pos_set.keys()))
    log.info("  Shared test queries: %d", len(shared_qids))

    # 4. Compute Borda, DE, HGB top-K for each query
    def compute_topk(de_ranks, hgb_ranks, k=10):
        """Compute Borda top-K using standard Borda count."""
        all_cands = set(de_ranks.keys()) | set(hgb_ranks.keys())
        scores = {}
        for cand in all_cands:
            score = 0.0
            de_r = de_ranks.get(cand)
            if de_r is not None:
                score += (50.0 - de_r + 1.0) / 50.0
            hg_r = hgb_ranks.get(cand)
            if hg_r is not None:
                score += (200.0 - hg_r + 1.0) / 200.0
            scores[cand] = score
        sorted_cands = sorted(scores.items(), key=lambda x: -x[1])
        return [c for c, s in sorted_cands[:k]]

    log.info("Computing Borda, DE, HGB top-%d for each query ...", TOP_K)
    borda_topk = {}
    de_topk = {}
    hgb_topk = {}
    for qid in shared_qids:
        de_list = [(normalize_fragment(p["c"]), p["r"]) for p in de_by_q.get(qid, [])]
        hgb_list = [(normalize_fragment(p["candidate"]), p.get("score", 0))
                     for p in hgb_by_q.get(qid, [])]

        # DE: rank-based
        de_ranks = {}
        for norm_smi, rank in de_list:
            de_ranks[norm_smi] = rank
        de_sorted = sorted(de_ranks.items(), key=lambda x: x[1])
        de_topk[qid] = [c for c, r in de_sorted[:TOP_K]]

        # HGB: score-based
        hgb_scored = {}
        for p in hgb_by_q.get(qid, []):
            ns = normalize_fragment(p["candidate"])
            hgb_scored[ns] = p["score"]
        hgb_sorted = sorted(hgb_scored.items(), key=lambda x: -x[1])
        hgb_topk[qid] = [c for c, s in hgb_sorted[:TOP_K]]
        hgb_ranks = {ns: i+1 for i, (ns, _) in enumerate(hgb_sorted[:200])}

        # Borda
        borda_topk[qid] = compute_topk(de_ranks, hgb_ranks, TOP_K)

    log.info("  Borda top-K computed for %d queries", len(borda_topk))

    # 5. Define groups
    # G0: all Borda T10 candidates
    # G1: borda_over_hgb_hits — Borda T10 hits AND HGB T10 misses (PRIMARY)
    # G2: pure_borda_only — G1 subset where DE T10 also misses
    # G3: de_only_retained_by_borda — G1 subset where DE T10 hits
    # G4: shared_hits — Borda T10 hits AND HGB T10 hits (validation)

    g0_candidates = []  # list of dict: qid, candidate, old_fragment
    g1_candidates = []
    g2_candidates = []
    g3_candidates = []
    g4_candidates = []

    for qid in shared_qids:
        b_set = set(borda_topk.get(qid, []))
        d_set = set(de_topk.get(qid, []))
        h_set = set(hgb_topk.get(qid, []))
        q_pos = pos_set.get(qid, set())
        m_entry = manifest_by_q.get(qid, {})
        old_smi = m_entry.get("old_fragment_smiles", "")

        # G0: all Borda T10 candidates
        for cand in b_set:
            is_pos = 1 if cand in q_pos else 0
            g0_candidates.append({
                "qid": qid, "candidate_norm": cand,
                "old_fragment": old_smi, "is_pos": is_pos
            })

        # G1: Borda hits ∩ HGB misses
        for cand in b_set & q_pos:
            is_borda_hit = 1
            is_hgb_hit = 1 if cand in h_set else 0
            if not is_hgb_hit:
                is_de_hit = 1 if cand in d_set else 0
                g1_candidates.append({
                    "qid": qid, "candidate_norm": cand,
                    "old_fragment": old_smi, "is_pos": 1,
                    "de_hit": is_de_hit
                })
                if is_de_hit:
                    g3_candidates.append(g1_candidates[-1])
                else:
                    g2_candidates.append(g1_candidates[-1])

        # G4: shared hits (Borda ∩ HGB)
        for cand in b_set & h_set & q_pos:
            g4_candidates.append({
                "qid": qid, "candidate_norm": cand,
                "old_fragment": old_smi, "is_pos": 1
            })

    log.info("  G0 (all Borda T10 candidates): %d", len(g0_candidates))
    log.info("  G1 (Borda_over_HGB hits): %d queries affected, %d candidates",
             len(set(c["qid"] for c in g1_candidates)), len(g1_candidates))
    log.info("  G2 (pure Borda-only): %d", len(g2_candidates))
    log.info("  G3 (DE retained by Borda): %d", len(g3_candidates))
    log.info("  G4 (shared hits): %d", len(g4_candidates))

    # Write G1 candidates with old_fragment for A4C cross-reference
    for group, name in [(g0_candidates, "G0"), (g1_candidates, "G1"),
                         (g2_candidates, "G2"), (g3_candidates, "G3"),
                         (g4_candidates, "G4")]:
        fieldnames = ["qid", "candidate_norm", "old_fragment", "is_pos"]
        if any(c.get("de_hit") is not None for c in group):
            fieldnames.append("de_hit")
        if any(c.get("hg_hit") is not None for c in group):
            fieldnames.append("hg_hit")
        write_csv(OUT / f"d4a3s_{name}_candidates.csv", fieldnames, group)

    # Group summary
    def group_summary(candidates):
        n_q = len(set(c["qid"] for c in candidates))
        n_cand = len(candidates)
        n_uniq = len(set(c["candidate_norm"] for c in candidates))
        return {"n_queries": n_q, "n_candidates": n_cand, "n_unique_replacements": n_uniq}

    group_rows = []
    for name, group in [("G0_all_borda_t10", g0_candidates),
                         ("G1_borda_over_hgb", g1_candidates),
                         ("G2_pure_borda_only", g2_candidates),
                         ("G3_de_retained_by_borda", g3_candidates),
                         ("G4_shared_hits", g4_candidates)]:
        s = group_summary(group)
        group_rows.append({"group": name, **s})
        log.info("  %s: %d queries, %d candidates, %d unique SMILES",
                 name, s["n_queries"], s["n_candidates"], s["n_unique_replacements"])

    write_csv(OUT / "d4a3s_borda_gain_region.csv",
              ["group", "n_queries", "n_candidates", "n_unique_replacements"], group_rows)

    # 6. A4C coverage by group
    log.info("Computing A4C coverage by group ...")
    coverage_rows = []
    for name, group in [("G0", g0_candidates), ("G1", g1_candidates),
                         ("G2", g2_candidates), ("G3", g3_candidates),
                         ("G4", g4_candidates)]:
        covered = 0
        for c in group:
            key = build_canonical_key(c["old_fragment"], f"*{c['candidate_norm']}")
            if key in a4c_by_smiles:
                covered += 1
        rate = covered / max(len(group), 1)
        coverage_rows.append({
            "group": name,
            "total": len(group),
            "a4c_covered": covered,
            "coverage_rate": f"{rate:.4f}"
        })
        log.info("  %s A4C coverage: %d/%d = %.4f", name, covered, len(group), rate)

    write_csv(OUT / "d4a3s_coverage_by_group.csv",
              ["group", "total", "a4c_covered", "coverage_rate"], coverage_rows)

    # 7. Gap classification for G1 uncovered
    log.info("Classifying G1 coverage gaps ...")
    gap_rows = []
    for c in g1_candidates:
        smiles_key = build_canonical_key(c["old_fragment"], f"*{c['candidate_norm']}")
        repl_no_star = strip_asterisk(f"*{c['candidate_norm']}")

        if smiles_key in a4c_by_smiles:
            gap_type = "JOIN_MISSING"
        elif repl_no_star in a4c_all_replacements or strip_asterisk(c["candidate_norm"]) in a4c_all_replacements:
            gap_type = "RECOMPUTABLE_FROM_SMILES"
        elif not c["old_fragment"]:
            gap_type = "QUERY_CONTEXT_MISSING"
        else:
            gap_type = "FRAGMENT_GRAPH_MISSING"

        gap_rows.append({
            "qid": c["qid"],
            "candidate_norm": c["candidate_norm"],
            "old_fragment": c["old_fragment"],
            "gap_type": gap_type,
            "is_de_hit": c.get("de_hit", 0)
        })

    write_csv(OUT / "d4a3s_coverage_diagnosis.csv",
              ["qid", "candidate_norm", "old_fragment", "gap_type", "is_de_hit"],
              gap_rows)

    # Summary
    gap_counts = Counter(r["gap_type"] for r in gap_rows)
    summary_rows = [{"gap_type": k, "count": v, "pct": f"{100*v/len(gap_rows):.1f}"}
                    for k, v in gap_counts.most_common()]
    write_csv(OUT / "d4a3s_coverage_summary.csv",
              ["gap_type", "count", "pct"], summary_rows)
    log.info("  Gap summary:")
    for r in summary_rows:
        log.info("    %s: %d (%.1f%%)", r["gap_type"], r["count"], float(r["pct"].rstrip("%")))

    return {
        "de_by_q": de_by_q,
        "hgb_by_q": hgb_by_q,
        "manifest_by_q": manifest_by_q,
        "pos_set": pos_set,
        "shared_qids": shared_qids,
        "borda_topk": borda_topk,
        "de_topk": de_topk,
        "hgb_topk": hgb_topk,
        "a4c_by_smiles": a4c_by_smiles,
        "a4c_all_replacements": a4c_all_replacements,
        "g0_candidates": g0_candidates,
        "g1_candidates": g1_candidates,
        "g2_candidates": g2_candidates,
        "g3_candidates": g3_candidates,
        "g4_candidates": g4_candidates,
        "gap_rows": gap_rows,
    }


# =========================================================================
# Part B: Join Logic Repair
# =========================================================================

def part_b_join_repair(data):
    """Rejoin Borda top-K to A4C via direct SMILES matching."""
    log.info("=" * 60)
    log.info("PART B: Join Logic Repair")
    log.info("=" * 60)

    a4c_by_smiles = data["a4c_by_smiles"]
    g1_candidates = data["g1_candidates"]
    g0_candidates = data["g0_candidates"]

    # Check if uncovered candidates with JOIN_MISSING type can be resolved
    repaired = 0
    still_missing = 0
    repair_log = []

    for c in g1_candidates:
        key = build_canonical_key(c["old_fragment"], f"*{c['candidate_norm']}")
        # Try with stripped variant
        alt_key = build_canonical_key(
            c["old_fragment"].lstrip("*") if c["old_fragment"] else "",
            c["candidate_norm"]
        )

        found = False
        for k in [key, alt_key]:
            if k in a4c_by_smiles:
                found = True
                break

        if found:
            repaired += 1
            repair_log.append({
                "qid": c["qid"],
                "candidate_norm": c["candidate_norm"],
                "join_status": "REPAIRED",
                "match_key": k
            })
        else:
            still_missing += 1
            repair_log.append({
                "qid": c["qid"],
                "candidate_norm": c["candidate_norm"],
                "join_status": "STILL_MISSING",
                "match_key": key
            })

    log.info("  Join repair: %d repaired, %d still missing", repaired, still_missing)

    write_csv(OUT / "d4a3s_join_repair_log.csv",
              ["qid", "candidate_norm", "join_status", "match_key"], repair_log)

    # Recompute coverage for G1 after repair
    g1_total = len(g1_candidates)
    coverage = repaired / max(g1_total, 1)

    cov_row = [{
        "group": "G1_borda_over_hgb",
        "total_candidates": g1_total,
        "repaired": repaired,
        "still_missing": still_missing,
        "coverage_after_repair": f"{coverage:.4f}"
    }]
    write_csv(OUT / "d4a3s_join_repair_coverage.csv",
              ["group", "total_candidates", "repaired", "still_missing",
               "coverage_after_repair"], cov_row)

    return repair_log


# =========================================================================
# Part C: A4C Recompute from SMILES
# =========================================================================

def part_c_a4c_recompute(data):
    """Recompute A4C bucket from SMILES using RDKit heuristics."""
    log.info("=" * 60)
    log.info("PART C: A4C Recompute from SMILES")
    log.info("=" * 60)

    if not RDKIT_AVAILABLE:
        log.warning("  RDKit not available — skipping Part C")
        write_md(OUT / "d4a3s_a4c_recompute_summary.md",
                 "# A4C Recompute Summary\n\n**SKIPPED**: RDKit not available.\n")
        return None

    g1_candidates = data["g1_candidates"]
    g4_candidates = data["g4_candidates"]
    a4c_by_smiles = data["a4c_by_smiles"]

    # Load PAINS filter
    pains_params = FilterCatalog.FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    pains = FilterCatalog.FilterCatalog(pains_params)
    brenk_params = FilterCatalog.FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    brenk = FilterCatalog.FilterCatalog(brenk_params)

    def compute_properties(smi):
        """Compute RDKit properties. Returns dict or None."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmarts(smi)
        if mol is None:
            return None
        try:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            ha = mol.GetNumHeavyAtoms()
            return {"mol": mol, "mw": mw, "logp": logp, "ha": ha}
        except Exception:
            return None

    def compute_a4c_bucket(old_smi, repl_smi):
        """Heuristic A4C bucket assignment matching D4A3 methodology."""
        old_props = compute_properties(old_smi)
        repl_props = compute_properties(repl_smi)
        if old_props is None or repl_props is None:
            return "PROPERTY_SHIFT_WARNING", {
                "reason": "RDKit_failed_to_parse",
                "ha_delta": -1, "mw_shift": -1
            }

        # PAINS alert (uses Mol, not fingerprint)
        pains_alerts = pains.HasMatch(repl_props["mol"])

        # Brenk alert
        brenk_alerts = brenk.HasMatch(repl_props["mol"])

        has_alert = pains_alerts or brenk_alerts

        # Heavy atom delta
        ha_old = old_props["ha"]
        ha_repl = repl_props["ha"]
        ha_delta_pct = abs(ha_repl - ha_old) / max(ha_old, 1)

        # MW shift
        mw_shift = abs(repl_props["mw"] - old_props["mw"])

        # Property shift
        prop_extreme = ha_delta_pct > 0.5 or mw_shift > 200

        has_alert = pains_alerts or brenk_alerts

        # Heavy atom delta
        ha_old = old_props["ha"]
        ha_repl = repl_props["ha"]
        ha_delta_pct = abs(ha_repl - ha_old) / max(ha_old, 1)

        # MW shift
        mw_shift = abs(repl_props["mw"] - old_props["mw"])

        # Property shift
        prop_extreme = ha_delta_pct > 0.5 or mw_shift > 200

        # Tanimoto
        tanimoto = DataStructs.TanimotoSimilarity(
            Chem.RDKFingerprint(old_props["mol"], fpSize=2048),
            Chem.RDKFingerprint(repl_props["mol"], fpSize=2048)
        )

        details = {
            "ha_delta": ha_repl - ha_old,
            "ha_delta_pct": round(ha_delta_pct, 3),
            "mw_shift": round(mw_shift, 1),
            "tanimoto": round(tanimoto, 4),
            "pains_alerts": int(pains_alerts),
            "brenk_alerts": int(brenk_alerts),
            "has_alert": int(has_alert),
            "prop_extreme": int(prop_extreme),
        }

        if has_alert:
            return HARD_ALERT, details
        if prop_extreme:
            return PROPERTY_WARNING, details
        if ha_delta_pct > 0.3 or mw_shift > 100:
            return REVIEW_WARNING, details
        return REVIEW_READY, details

    # Phase 1: Validation on G4 (which has existing A4C records)
    log.info("  Validating recompute on G4 (shared hits with existing A4C) ...")
    validation_results = []
    g4_agree = 0
    g4_total = 0

    for c in g4_candidates:
        old_frag = c["old_fragment"]
        repl_smi = f"*{c['candidate_norm']}"
        key = build_canonical_key(old_frag, repl_smi)

        if key not in a4c_by_smiles:
            continue

        orig_bucket = a4c_by_smiles[key][0]["a4c_bucket"]
        recompute_bucket, details = compute_a4c_bucket(
            old_frag.lstrip("*"), repl_smi.lstrip("*")
        )
        agree = 1 if orig_bucket == recompute_bucket else 0
        g4_agree += agree
        g4_total += 1

        validation_results.append({
            "old_fragment": old_frag,
            "replacement_smiles": repl_smi,
            "original_bucket": orig_bucket,
            "recomputed_bucket": recompute_bucket,
            "agree": agree,
            **details
        })

    g4_agreement_rate = g4_agree / max(g4_total, 1)
    log.info("  G4 validation: %d/%d agree = %.4f", g4_agree, g4_total, g4_agreement_rate)
    write_csv(OUT / "d4a3s_a4c_recompute_validation.csv",
              ["old_fragment", "replacement_smiles", "original_bucket",
               "recomputed_bucket", "agree", "ha_delta", "ha_delta_pct",
               "mw_shift", "tanimoto", "pains_alerts", "brenk_alerts",
               "has_alert", "prop_extreme"], validation_results)

    val_flag = "A4C_RECOMPUTE_VALIDATION_FAILED" if g4_agreement_rate < 0.80 else "A4C_RECOMPUTE_VALIDATION_PASSED"
    log.info("  Validation: %s (rate=%.4f)", val_flag, g4_agreement_rate)

    # Phase 2: Recompute for RECOMPUTABLE_FROM_SMILES G1 candidates
    log.info("  Recomputing A4C for RECOMPUTABLE_FROM_SMILES G1 candidates ...")
    recompute_results = []
    recomputed_records = []

    gap_rows = data["gap_rows"]
    recompute_candidates = [r for r in gap_rows if r["gap_type"] == "RECOMPUTABLE_FROM_SMILES"]

    for r in recompute_candidates:
        old_frag = r["old_fragment"]
        repl_smi = f"*{r['candidate_norm']}"
        bucket, details = compute_a4c_bucket(
            old_frag.lstrip("*"), r["candidate_norm"]
        )

        recompute_results.append({
            "qid": r["qid"],
            "old_fragment": old_frag,
            "replacement_smiles": repl_smi,
            "recomputed_bucket": bucket,
            "a4c_source": A4C_RECOMPUTED,
            **details
        })

        recomputed_records.append({
            "qid": r["qid"],
            "old_fragment": old_frag,
            "replacement_smiles": repl_smi,
            "candidate_norm": r["candidate_norm"],
            "a4c_bucket": bucket,
            "a4c_source": A4C_RECOMPUTED,
            "ha_delta": details["ha_delta"],
            "ha_delta_pct": details["ha_delta_pct"],
            "tanimoto": details["tanimoto"],
            "pains_alerts": details["pains_alerts"],
            "brenk_alerts": details["brenk_alerts"],
            "has_alert": details["has_alert"],
        })

    log.info("  Recomputed %d records", len(recompute_results))

    if recompute_results:
        write_csv(OUT / "d4a3s_a4c_recompute_attempt.csv",
                  ["qid", "old_fragment", "replacement_smiles", "recomputed_bucket",
                   "a4c_source", "ha_delta", "ha_delta_pct", "mw_shift", "tanimoto",
                   "pains_alerts", "brenk_alerts", "has_alert", "prop_extreme"],
                  recompute_results)
        write_jsonl(OUT / "d4a3s_a4c_recomputed_records.jsonl", recomputed_records)

    # Summary
    bucket_counts = Counter(r["recomputed_bucket"] for r in recompute_results)
    summary = [
        f"# A4C Recompute Summary\n",
        f"\n**Date:** 2026-05-24\n",
        f"\n**G4 Validation Rate:** {g4_agreement_rate:.4f} ({val_flag})\n",
        f"\n**Recomputed Candidates:** {len(recompute_results)}\n",
        f"\n## Bucket Distribution\n\n",
    ]
    for bucket, count in bucket_counts.most_common():
        summary.append(f"- {bucket}: {count}\n")
    summary.append(
        f"\n## Note\n\n"
        f"All recomputed records are marked A4C_RECOMPUTED_FROM_SMILES.\n"
        f"They are PROXY_EVIDENCE, not original A4C expert review.\n"
        f"If G4 validation rate < 80%, mark validation failed.\n"
    )
    write_md(OUT / "d4a3s_a4c_recompute_summary.md", "".join(summary))

    return {
        "recompute_results": recompute_results,
        "recomputed_records": recomputed_records,
        "g4_agreement_rate": g4_agreement_rate,
        "val_flag": val_flag,
    }


# =========================================================================
# Part D: Fragment Graph Gap Analysis
# =========================================================================

def part_d_fragment_audit(data):
    """Analyze fragments missing from A4C database."""
    log.info("=" * 60)
    log.info("PART D: Fragment Graph Gap Analysis")
    log.info("=" * 60)

    a4c_by_smiles = data["a4c_by_smiles"]
    gap_rows = data["gap_rows"]

    fragment_missing = [r for r in gap_rows if r["gap_type"] == "FRAGMENT_GRAPH_MISSING"]

    if not fragment_missing:
        log.info("  No FRAGMENT_GRAPH_MISSING candidates.")
        write_md(OUT / "d4a3s_fragment_gap_summary.md",
                 "# Fragment Gap Summary\n\nNo fragments missing from A4C database.\n")
        return

    # Count by old_fragment
    frag_counts = Counter(r["old_fragment"] for r in fragment_missing)
    log.info("  %d fragments missing, %d unique old fragments",
             len(fragment_missing), len(frag_counts))

    # Compute Morgan FP for each missing fragment and find nearest A4C neighbor
    audit_rows = []
    if RDKIT_AVAILABLE:
        # Build A4C fragment fingerprints
        a4c_frags = {}
        for key in a4c_by_smiles:
            old = key.split("||")[0]
            if old:
                a4c_frags[old] = None
        log.info("  A4C unique old fragments: %d", len(a4c_frags))

        a4c_fp_db = {}
        for frag in a4c_frags:
            mol = Chem.MolFromSmiles(frag.lstrip("*"))
            if mol:
                a4c_fp_db[frag] = Chem.RDKFingerprint(mol, fpSize=2048)

        for frag, count in frag_counts.most_common(50):
            mol = Chem.MolFromSmiles(frag.lstrip("*"))
            if mol is None:
                audit_rows.append({
                    "old_fragment": frag,
                    "count": count,
                    "nearest_a4c_neighbor": "PARSE_FAILED",
                    "tanimoto": "N/A"
                })
                continue
            fp = Chem.RDKFingerprint(mol, fpSize=2048)
            best_sim = 0.0
            best_frag = ""
            for af, afp in a4c_fp_db.items():
                sim = DataStructs.TanimotoSimilarity(fp, afp)
                if sim > best_sim:
                    best_sim = sim
                    best_frag = af
            audit_rows.append({
                "old_fragment": frag,
                "count": count,
                "nearest_a4c_neighbor": best_frag,
                "tanimoto": f"{best_sim:.4f}"
            })
    else:
        for frag, count in frag_counts.most_common():
            audit_rows.append({
                "old_fragment": frag,
                "count": count,
                "nearest_a4c_neighbor": "RDKIT_UNAVAILABLE",
                "tanimoto": "N/A"
            })

    write_csv(OUT / "d4a3s_fragment_audit.csv",
              ["old_fragment", "count", "nearest_a4c_neighbor", "tanimoto"], audit_rows)

    summary_lines = [
        f"# Fragment Gap Summary\n\n",
        f"**Total missing fragments:** {len(fragment_missing)}\n",
        f"**Unique missing old fragments:** {len(frag_counts)}\n",
        f"\n## Top-10 Most Common Missing Fragments\n\n",
    ]
    for frag, count in frag_counts.most_common(10):
        summary_lines.append(f"- `{frag}`: {count} occurrences\n")
    write_md(OUT / "d4a3s_fragment_gap_summary.md", "".join(summary_lines))


# =========================================================================
# Part E: Chemical Proxy Screening
# =========================================================================

def part_e_chemical_screening(data):
    """Chemical proxy screening for G1 vs G4 comparison."""
    log.info("=" * 60)
    log.info("PART E: Chemical Proxy Screening")
    log.info("=" * 60)

    if not RDKIT_AVAILABLE:
        log.warning("  RDKit not available — skipping Part E")
        write_md(OUT / "d4a3s_chemical_screening_summary.md",
                 "# Chemical Screening Summary\n\n**SKIPPED**: RDKit not available.\n")
        return

    g1_candidates = data["g1_candidates"]
    g4_candidates = data["g4_candidates"]

    def compute_tanimoto(old_smi, repl_smi):
        """Tanimoto between old fragment and replacement."""
        try:
            mol_old = Chem.MolFromSmiles(old_smi.lstrip("*"))
            mol_repl = Chem.MolFromSmiles(repl_smi.lstrip("*"))
            if mol_old is None or mol_repl is None:
                return None
            fp_old = Chem.RDKFingerprint(mol_old, fpSize=2048)
            fp_repl = Chem.RDKFingerprint(mol_repl, fpSize=2048)
            return DataStructs.TanimotoSimilarity(fp_old, fp_repl)
        except Exception:
            return None

    def compute_self_diversity(smiles_list):
        """Mean pairwise Tanimoto within a set of SMILES."""
        mols = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s.lstrip("*"))
            if mol:
                mols.append(mol)
        if len(mols) < 2:
            return 0.0
        fps = [Chem.RDKFingerprint(m, fpSize=2048) for m in mols]
        total = 0.0
        count = 0
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                total += DataStructs.TanimotoSimilarity(fps[i], fps[j])
                count += 1
        return total / max(count, 1)

    def structural_alert_rate(smiles_list):
        """Check PAINS and Brenk alerts for a list of SMILES."""
        pains_params = FilterCatalog.FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
        pains = FilterCatalog.FilterCatalog(pains_params)
        brenk_params = FilterCatalog.FilterCatalogParams()
        brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        brenk = FilterCatalog.FilterCatalog(brenk_params)

        alerts = 0
        total = 0
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s.lstrip("*"))
            if mol is None:
                continue
            total += 1
            if pains.HasMatch(mol) or brenk.HasMatch(mol):
                alerts += 1
        return alerts, total

    log.info("E1: Tanimoto distribution ...")
    g1_tani = []
    for c in g1_candidates:
        t = compute_tanimoto(c["old_fragment"], f"*{c['candidate_norm']}")
        if t is not None:
            g1_tani.append(t)

    g4_tani = []
    for c in g4_candidates:
        t = compute_tanimoto(c["old_fragment"], f"*{c['candidate_norm']}")
        if t is not None:
            g4_tani.append(t)

    log.info("  G1 Tanimoto: mean=%.4f median=%.4f std=%.4f n=%d",
             mean(g1_tani), sorted(g1_tani)[len(g1_tani)//2],
             stdev(g1_tani) if len(g1_tani) > 1 else 0, len(g1_tani))
    log.info("  G4 Tanimoto: mean=%.4f median=%.4f std=%.4f n=%d",
             mean(g4_tani), sorted(g4_tani)[len(g4_tani)//2],
             stdev(g4_tani) if len(g4_tani) > 1 else 0, len(g4_tani))

    # Bootstrap CI for difference
    diff_tani = []
    for _ in range(N_BOOTSTRAP):
        g1_sample = np.random.choice(g1_tani, size=len(g1_tani), replace=True)
        g4_sample = np.random.choice(g4_tani, size=len(g4_tani), replace=True)
        diff_tani.append(np.mean(g1_sample) - np.mean(g4_sample))
    diff_tani.sort()
    ci_low, ci_high = diff_tani[int(0.025 * N_BOOTSTRAP)], diff_tani[int(0.975 * N_BOOTSTRAP)]
    log.info("  G1 - G4 Tanimoto delta: %.4f [%.4f, %.4f]",
             mean(g1_tani) - mean(g4_tani), ci_low, ci_high)

    log.info("E2: Structural alert screening ...")
    g1_alerts, g1_total = structural_alert_rate(
        [f"*{c['candidate_norm']}" for c in g1_candidates])
    g4_alerts, g4_total = structural_alert_rate(
        [f"*{c['candidate_norm']}" for c in g4_candidates])

    g1_alert_rate = g1_alerts / max(g1_total, 1)
    g4_alert_rate = g4_alerts / max(g4_total, 1)
    log.info("  G1 alert rate: %d/%d = %.4f", g1_alerts, g1_total, g1_alert_rate)
    log.info("  G4 alert rate: %d/%d = %.4f", g4_alerts, g4_total, g4_alert_rate)

    # Bootstrap CI for alert rate difference
    g1_alerts_arr = np.array([1.0] * g1_alerts + [0.0] * (g1_total - g1_alerts))
    g4_alerts_arr = np.array([1.0] * g4_alerts + [0.0] * (g4_total - g4_alerts))
    diff_alerts = []
    for _ in range(N_BOOTSTRAP):
        b1 = np.random.choice(g1_alerts_arr, size=g1_total, replace=True)
        b4 = np.random.choice(g4_alerts_arr, size=g4_total, replace=True)
        diff_alerts.append(np.mean(b1) - np.mean(b4))
    diff_alerts.sort()
    ci_alert_low = diff_alerts[int(0.025 * N_BOOTSTRAP)]
    ci_alert_high = diff_alerts[int(0.975 * N_BOOTSTRAP)]
    log.info("  G1 - G4 alert rate delta: %.4f [%.4f, %.4f]",
             g1_alert_rate - g4_alert_rate, ci_alert_low, ci_alert_high)

    log.info("E3: Chemical diversity ...")
    g1_smiles = list(set(c["candidate_norm"] for c in g1_candidates))
    g4_smiles = list(set(c["candidate_norm"] for c in g4_candidates))
    g1_diversity = compute_self_diversity(g1_smiles)
    g4_diversity = compute_self_diversity(g4_smiles)
    log.info("  G1 internal diversity: %.4f (%d unique)", g1_diversity, len(g1_smiles))
    log.info("  G4 internal diversity: %.4f (%d unique)", g4_diversity, len(g4_smiles))

    # Write screening results
    rows = [
        {"metric": "E1_g1_tanimoto_mean", "value": f"{mean(g1_tani):.4f}"},
        {"metric": "E1_g1_tanimoto_median", "value": f"{sorted(g1_tani)[len(g1_tani)//2]:.4f}"},
        {"metric": "E1_g1_tanimoto_std", "value": f"{stdev(g1_tani) if len(g1_tani) > 1 else 0:.4f}"},
        {"metric": "E1_g4_tanimoto_mean", "value": f"{mean(g4_tani):.4f}"},
        {"metric": "E1_g4_tanimoto_median", "value": f"{sorted(g4_tani)[len(g4_tani)//2]:.4f}"},
        {"metric": "E1_g4_tanimoto_std", "value": f"{stdev(g4_tani) if len(g4_tani) > 1 else 0:.4f}"},
        {"metric": "E1_delta_mean", "value": f"{mean(g1_tani) - mean(g4_tani):.4f}"},
        {"metric": "E1_delta_ci_low", "value": f"{ci_low:.4f}"},
        {"metric": "E1_delta_ci_high", "value": f"{ci_high:.4f}"},
        {"metric": "E2_g1_alert_rate", "value": f"{g1_alert_rate:.4f}"},
        {"metric": "E2_g4_alert_rate", "value": f"{g4_alert_rate:.4f}"},
        {"metric": "E2_delta_alert_rate", "value": f"{g1_alert_rate - g4_alert_rate:.4f}"},
        {"metric": "E2_delta_alert_ci_low", "value": f"{ci_alert_low:.4f}"},
        {"metric": "E2_delta_alert_ci_high", "value": f"{ci_alert_high:.4f}"},
        {"metric": "E3_g1_diversity", "value": f"{g1_diversity:.4f}"},
        {"metric": "E3_g4_diversity", "value": f"{g4_diversity:.4f}"},
    ]

    sort_key = {"E1": 1, "E2": 2, "E3": 3}
    rows.sort(key=lambda r: sort_key.get(r["metric"][:2], 0))
    write_csv(OUT / "d4a3s_chemical_screening.csv",
              ["metric", "value"], rows)

    # Mark all screening results as proxy evidence
    screen_note = (
        "# Chemical Proxy Screening Summary\n\n"
        "**ALL RESULTS MARKED: PROXY_EVIDENCE_NOT_A4C_REVIEW**\n\n"
        "Chemical screening is computational proxy evidence, not A4C expert review.\n"
        "These results do NOT replace curated A4C evaluation.\n\n"
    )
    signif_alert = ci_alert_low > 0
    screen_note += "## E1: Tanimoto Distribution\n\n"
    screen_note += f"- G1 mean Tanimoto: {mean(g1_tani):.4f}\n"
    screen_note += f"- G4 mean Tanimoto: {mean(g4_tani):.4f}\n"
    screen_note += f"- Delta (G1-G4): {mean(g1_tani) - mean(g4_tani):.4f}\n"
    screen_note += f"- Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]\n\n"
    screen_note += "## E2: Structural Alerts\n\n"
    screen_note += f"- G1 alert rate: {g1_alert_rate:.4f} ({g1_alerts}/{g1_total})\n"
    screen_note += f"- G4 alert rate: {g4_alert_rate:.4f} ({g4_alerts}/{g4_total})\n"
    screen_note += f"- Delta (G1-G4): {g1_alert_rate - g4_alert_rate:.4f}\n"
    screen_note += f"- Bootstrap 95% CI: [{ci_alert_low:.4f}, {ci_alert_high:.4f}]\n"
    screen_note += f"- **CI lower bound > 0**: {'YES - G1 has SIGNIFICANTLY more alerts' if signif_alert else 'NO - not significant'}\n\n"
    screen_note += "## E3: Chemical Diversity\n\n"
    screen_note += f"- G1 internal diversity: {g1_diversity:.4f}\n"
    screen_note += f"- G4 internal diversity: {g4_diversity:.4f}\n"
    screen_note += f"- Delta: {g1_diversity - g4_diversity:.4f}\n"

    write_md(OUT / "d4a3s_chemical_screening_summary.md", screen_note)


# =========================================================================
# Part F: A4C Re-Evaluation (conditional on coverage >= 95%)
# =========================================================================

def part_f_a4c_reeval(data, recompute_data):
    """Full A4C metrics on G1 with repaired+recomputed coverage."""
    log.info("=" * 60)
    log.info("PART F: A4C Re-Evaluation")
    log.info("=" * 60)

    # Pre-registered success criteria (DO NOT MODIFY)
    CRITERIA = {
        "coverage_borda_over_hgb_hits": 0.95,
        "hard_reject_rate_max_delta_pp": 2.0,  # Borda <= HGB + 2pp
        "bootstrap_hard_reject_ci_upper": 0.03,  # upper bound <= +3pp
    }

    g1_candidates = data["g1_candidates"]
    g4_candidates = data["g4_candidates"]
    a4c_by_smiles = data["a4c_by_smiles"]
    gap_rows = data["gap_rows"]

    # Build full A4C lookup including recomputed
    a4c_full = dict(a4c_by_smiles)

    # Add recomputed records
    recomputed_records = (recompute_data or {}).get("recomputed_records", [])
    for rec in recomputed_records:
        key = build_canonical_key(rec["old_fragment"], rec["replacement_smiles"])
        a4c_full[key] = [rec]

    # Coverage check
    g1_covered = 0
    for c in g1_candidates:
        key = build_canonical_key(c["old_fragment"], f"*{c['candidate_norm']}")
        if key in a4c_full:
            g1_covered += 1
    g1_total = len(g1_candidates)
    coverage_rate = g1_covered / max(g1_total, 1)
    log.info("  G1 coverage after repair+recompute: %d/%d = %.4f", g1_covered, g1_total, coverage_rate)

    coverage_gate = coverage_rate >= CRITERIA["coverage_borda_over_hgb_hits"]
    if not coverage_gate:
        log.warning("  Coverage gate FAILED (%.4f < %.2f) — skipping re-evaluation",
                    coverage_rate, CRITERIA["coverage_borda_over_hgb_hits"])
        write_csv(OUT / "d4a3s_a4c_reeval_metrics.csv",
                  ["group", "metric", "value"],
                  [{"group": "COVERAGE_GATE", "metric": "coverage_rate",
                    "value": f"{coverage_rate:.4f}"},
                   {"group": "COVERAGE_GATE", "metric": "passed",
                    "value": str(coverage_gate)}])
        return False

    log.info("  Coverage gate PASSED — proceeding with full re-evaluation")

    # Compute A4C metrics for G1 (with full coverage)
    g1_buckets = Counter()
    for c in g1_candidates:
        key = build_canonical_key(c["old_fragment"], f"*{c['candidate_norm']}")
        a4c_row = a4c_full.get(key, [{}])[0]
        bucket = a4c_row.get("a4c_bucket", "UNKNOWN")
        g1_buckets[bucket] += 1

    g1_hard_reject = g1_buckets.get(HARD_ALERT, 0)
    g1_review_ready = g1_buckets.get(REVIEW_READY, 0) + g1_buckets.get(REVIEW_WARNING, 0)
    g1_unknown = g1_buckets.get("UNKNOWN", 0)

    # G4 reference
    g4_buckets = Counter()
    for c in g4_candidates:
        key = build_canonical_key(c["old_fragment"], f"*{c['candidate_norm']}")
        a4c_row = a4c_full.get(key, [{}])[0]
        bucket = a4c_row.get("a4c_bucket", "UNKNOWN")
        g4_buckets[bucket] += 1

    g4_hard_reject = g4_buckets.get(HARD_ALERT, 0) + g4_buckets.get(PROPERTY_WARNING, 0)

    # Bootstrap
    log.info("  Bootstrap comparisons ...")
    g1_hr_arr = np.array([1.0 if c["a4c_bucket"] == HARD_ALERT else 0.0
                          for c in g1_candidates
                          if (key := build_canonical_key(c["old_fragment"],
                              f"*{c['candidate_norm']}")) in a4c_full
                          and (a4c_full[key][0].get("a4c_bucket", "UNKNOWN") != "UNKNOWN"
                               for _ in [1])])  # noqa — generator for comprehension assignment

    # Simpler bootstrap
    g1_hr = []
    for c in g1_candidates:
        key = build_canonical_key(c["old_fragment"], f"*{c['candidate_norm']}")
        if key in a4c_full:
            bucket = a4c_full[key][0].get("a4c_bucket", "UNKNOWN")
            if bucket != "UNKNOWN":
                g1_hr.append(1.0 if bucket == HARD_ALERT else 0.0)

    g4_hr = []
    for c in g4_candidates:
        key = build_canonical_key(c["old_fragment"], f"*{c['candidate_norm']}")
        if key in a4c_full:
            bucket = a4c_full[key][0].get("a4c_bucket", "UNKNOWN")
            if bucket != "UNKNOWN":
                g4_hr.append(1.0 if bucket == HARD_ALERT or bucket == PROPERTY_WARNING else 0.0)

    log.info("  G1 hard reject rate: %d/%d = %.4f",
             int(sum(g1_hr)), len(g1_hr), mean(g1_hr) if g1_hr else 0)
    log.info("  G4 hard reject rate: %d/%d = %.4f",
             int(sum(g4_hr)), len(g4_hr), mean(g4_hr) if g4_hr else 0)

    # Bootstrap Borda hard_reject vs HGB hard_reject
    # Use G4 as proxy for HGB (shared hits with A4C)
    bootstrap_diffs = []
    for _ in range(N_BOOTSTRAP):
        b1 = np.random.choice(g1_hr if g1_hr else [0.0], size=max(len(g1_hr), 1), replace=True)
        b4 = np.random.choice(g4_hr if g4_hr else [0.0], size=max(len(g4_hr), 1), replace=True)
        bootstrap_diffs.append(np.mean(b1) - np.mean(b4))
    bootstrap_diffs.sort()
    ci_hr = (bootstrap_diffs[int(0.025 * N_BOOTSTRAP)] if N_BOOTSTRAP > 25 else 0,
             bootstrap_diffs[int(0.975 * N_BOOTSTRAP)] if N_BOOTSTRAP > 25 else 0)

    reeval_rows = [
        {"group": "G1_borda_over_hgb", "metric": "n_candidates", "value": len(g1_hr)},
        {"group": "G1_borda_over_hgb", "metric": "hard_reject_rate",
         "value": f"{mean(g1_hr) if g1_hr else 0:.4f}"},
        {"group": "G1_borda_over_hgb", "metric": "coverage_rate", "value": f"{coverage_rate:.4f}"},
        {"group": "G4_shared_hits", "metric": "n_candidates", "value": len(g4_hr)},
        {"group": "G4_shared_hits", "metric": "hard_reject_rate",
         "value": f"{mean(g4_hr) if g4_hr else 0:.4f}"},
        {"group": "Borda_vs_HGB", "metric": "hard_reject_delta",
         "value": f"{mean(g1_hr) - mean(g4_hr) if g1_hr and g4_hr else 0:.4f}"},
        {"group": "Borda_vs_HGB", "metric": "bootstrap_ci_low", "value": f"{ci_hr[0]:.4f}"},
        {"group": "Borda_vs_HGB", "metric": "bootstrap_ci_high", "value": f"{ci_hr[1]:.4f}"},
    ]
    write_csv(OUT / "d4a3s_a4c_reeval_metrics.csv",
              ["group", "metric", "value"], reeval_rows)

    # Gate check
    hard_reject_delta = (mean(g1_hr) - mean(g4_hr)) if g1_hr and g4_hr else 0
    risk_gate = hard_reject_delta <= (CRITERIA["hard_reject_rate_max_delta_pp"] / 100.0)
    bootstrap_gate = ci_hr[1] <= CRITERIA["bootstrap_hard_reject_ci_upper"]

    log.info("  Hard reject delta: %.4f (gate: %s, threshold: +%.2fpp)",
             hard_reject_delta, "PASS" if risk_gate else "FAIL",
             CRITERIA["hard_reject_rate_max_delta_pp"])
    log.info("  Bootstrap CI upper: %.4f (gate: %s, threshold: +%.3f)",
             ci_hr[1], "PASS" if bootstrap_gate else "FAIL",
             CRITERIA["bootstrap_hard_reject_ci_upper"])

    write_csv(OUT / "d4a3s_a4c_reeval_bootstrap.csv",
              ["comparison", "delta", "ci_low", "ci_high",
               "coverage_gate", "risk_gate", "bootstrap_gate"],
              [{
                  "comparison": "Borda_vs_HGB_hard_reject",
                  "delta": f"{hard_reject_delta:.4f}",
                  "ci_low": f"{ci_hr[0]:.4f}",
                  "ci_high": f"{ci_hr[1]:.4f}",
                  "coverage_gate": "PASS" if coverage_gate else "FAIL",
                  "risk_gate": "PASS" if risk_gate else "FAIL",
                  "bootstrap_gate": "PASS" if bootstrap_gate else "FAIL",
              }])

    # G1 A4C profile
    profile_rows = []
    for c in g1_candidates:
        key = build_canonical_key(c["old_fragment"], f"*{c['candidate_norm']}")
        a4c_entry = a4c_full.get(key, [{}])[0]
        profile_rows.append({
            "qid": c["qid"],
            "candidate_norm": c["candidate_norm"],
            "a4c_bucket": a4c_entry.get("a4c_bucket", "UNKNOWN"),
            "a4c_source": a4c_entry.get("a4c_source", "ORIGINAL"),
        })
    write_csv(OUT / "d4a3s_borda_gain_region_a4c_profile.csv",
              ["qid", "candidate_norm", "a4c_bucket", "a4c_source"], profile_rows)

    return {
        "coverage_rate": coverage_rate,
        "coverage_gate": coverage_gate,
        "risk_gate": risk_gate,
        "bootstrap_gate": bootstrap_gate,
        "g1_hard_reject_rate": mean(g1_hr) if g1_hr else 0,
        "g4_hard_reject_rate": mean(g4_hr) if g4_hr else 0,
        "bootstrap_ci": ci_hr,
    }


# =========================================================================
# Part G: Final Verdict
# =========================================================================

def part_g_final_verdict(data, recompute_data, reeval_data):
    """Generate final verdict and decision log."""
    log.info("=" * 60)
    log.info("PART G: Final Verdict")
    log.info("=" * 60)

    g1_candidates = data["g1_candidates"]
    g4_candidates = data["g4_candidates"]
    gap_rows = data["gap_rows"]
    a4c_by_smiles = data["a4c_by_smiles"]

    # 9 questions
    questions = [
        ("Q1", "What fraction of 21,052 test queries have Borda>T10 hits that HGB misses?"),
        ("Q2", "What is the A4C coverage rate for G1 candidates?"),
        ("Q3", "What type of gap dominates uncovered G1 candidates?"),
        ("Q4", "Did join repair improve coverage?"),
        ("Q5", "Could SMILES-based recompute fill remaining gaps?"),
        ("Q6", "Are G1 candidates chemically different from G4 (validation set)?"),
        ("Q7", "Is proxy chemical evidence sufficient to claim review-safety?"),
        ("Q8", "Does re-evaluation meet pre-registered success criteria?"),
        ("Q9", "What is the recommended next action?"),
    ]

    # Compute answers
    g1_qids = set(c["qid"] for c in g1_candidates)
    n_g1_queries = len(g1_qids)
    g1_total = len(g1_candidates)

    # Initial coverage
    g1_covered_init = sum(1 for c in g1_candidates
                          if build_canonical_key(c["old_fragment"],
                              f"*{c['candidate_norm']}") in a4c_by_smiles)
    coverage_init = g1_covered_init / max(g1_total, 1)

    # Domination gap type
    gap_counts = Counter(r["gap_type"] for r in gap_rows)

    # Coverage after repair
    recompute_count = len((recompute_data or {}).get("recomputed_records", []))

    # Re-eval status
    reeval_passed = reeval_data is not False and reeval_data is not None

    verdict_lines = [
        "# D4A3S A4C Coverage Expansion Verdict\n",
        f"**Date:** 2026-05-24\n",
        f"\n## Questions and Answers\n\n",
    ]

    verdict_lines.append(f"### Q1: {questions[0][1]}\n")
    verdict_lines.append(f"- G1 query count: {n_g1_queries} / 21052 ({100*n_g1_queries/21052:.1f}%)\n")
    verdict_lines.append(f"- G1 total candidates: {g1_total}\n\n")

    verdict_lines.append(f"### Q2: {questions[1][1]}\n")
    verdict_lines.append(f"- Initial A4C coverage: {g1_covered_init}/{g1_total} = {coverage_init:.4f}\n\n")

    verdict_lines.append(f"### Q3: {questions[2][1]}\n")
    for gt, cnt in gap_counts.most_common():
        verdict_lines.append(f"- {gt}: {cnt} ({100*cnt/max(g1_total,1):.1f}% of G1)\n")
    verdict_lines.append("\n")

    verdict_lines.append(f"### Q4: {questions[3][1]}\n")
    verdict_lines.append(f"- SMILES-based join repair attempted\n")
    verdict_lines.append(f"- Recomputed candidates: {recompute_count}\n\n")

    verdict_lines.append(f"### Q5: {questions[4][1]}\n")
    verdict_lines.append(f"- RDKit available: {RDKIT_AVAILABLE}\n")
    if recompute_data:
        vr = recompute_data.get("g4_agreement_rate", 0)
        verdict_lines.append(f"- G4 validation agreement rate: {vr:.4f}\n")
        verdict_lines.append(f"- Validation status: {recompute_data.get('val_flag', 'N/A')}\n")
    verdict_lines.append("\n")

    verdict_lines.append(f"### Q6: {questions[5][1]}\n")
    verdict_lines.append("- See d4a3s_chemical_screening.csv for full analysis\n")
    verdict_lines.append("- G1 vs G4 Tanimoto, alert rate, diversity compared\n\n")

    verdict_lines.append(f"### Q7: {questions[6][1]}\n")
    verdict_lines.append(f"- **NO**. All proxy evidence is marked {PROXY_EVIDENCE}.\n")
    verdict_lines.append("- Computational screening cannot replace expert A4C review.\n\n")

    verdict_lines.append(f"### Q8: {questions[7][1]}\n")
    if reeval_passed and isinstance(reeval_data, dict):
        verdict_lines.append(f"- Coverage gate: {'PASS' if reeval_data.get('coverage_gate') else 'FAIL'} (rate={reeval_data.get('coverage_rate', 0):.4f})\n")
        verdict_lines.append(f"- Risk gate: {'PASS' if reeval_data.get('risk_gate') else 'FAIL'}\n")
        verdict_lines.append(f"- Bootstrap gate: {'PASS' if reeval_data.get('bootstrap_gate') else 'FAIL'}\n")
    else:
        verdict_lines.append("- Re-evaluation was not completed (coverage < 95%)\n\n")

    verdict_lines.append(f"### Q9: {questions[8][1]}\n")
    if reeval_passed and isinstance(reeval_data, dict) and all([
        reeval_data.get("coverage_gate"), reeval_data.get("risk_gate"),
        reeval_data.get("bootstrap_gate")
    ]):
        verdict_letter = "A"
        verdict_desc = "BORDA_PRODUCTION_READY_REVIEW_SAFE"
    elif reeval_passed and isinstance(reeval_data, dict) and reeval_data.get("coverage_gate"):
        verdict_letter = "B"
        verdict_desc = "BORDA_IMPROVES_RECOVERY_BUT_NEEDS_A4C_RERANKING"
    else:
        verdict_letter = "C"
        verdict_desc = "BORDA_PENDING_A4C_COVERAGE"

    verdict_lines.append(f"- **Verdict: {verdict_letter}. {verdict_desc}**\n")
    verdict_lines.append(f"- Next: See MAIN_DECISION_LOG.md for detailed next actions\n\n")

    # Verdict row for summary
    verdict_row = {
        "component": "D4A3S_A4C_COVERAGE_EXPANSION",
        "verdict_code": verdict_letter,
        "verdict_title": verdict_desc,
        "n_g1_queries": n_g1_queries,
        "n_g1_candidates": g1_total,
        "initial_coverage": f"{coverage_init:.4f}",
        "recomputed_count": recompute_count,
        "dominant_gap": gap_counts.most_common(1)[0][0] if gap_counts else "NONE",
    }
    write_csv(OUT / "D4A3S_VERDICT_SUMMARY.csv",
              ["component", "verdict_code", "verdict_title", "n_g1_queries",
               "n_g1_candidates", "initial_coverage", "recomputed_count", "dominant_gap"],
              [verdict_row])

    # Skeptical review
    verdict_lines.extend([
        "\n## Skeptical Review\n\n",
        "1. **Coverage expansion is limited by available data.**\n",
        "   SMILES-based recompute is a heuristic, not expert review.\n",
        "   Chemical proxy screening is explicitly NOT A4C review.\n\n",
        "2. **Gap classification depends on A4C SMILES matching quality.**\n",
        "   JOIN_MISSING cases may be join logic issues or genuine absences.\n\n",
        "3. **Borda gains on 14,215 non-A4C-eval queries are harder to verify**\n",
        "   than gains on the 6,837 A4C eval queries. This asymmetry is structural.\n\n",
        "4. **Hard reject rate comparison uses G4 as HGB proxy.**\n",
        "   G4 = shared hits, which may not represent the full HGB distribution.\n\n",
        "5. **No model retraining was performed.**\n",
        "   All analysis is post-hoc diagnostic, not a production fix.\n",
        "   If A4C coverage is structurally incomplete, retraining may be needed.\n",
    ])

    verdict_text = "".join(verdict_lines)
    write_md(OUT / "D4A3S_A4C_COVERAGE_VERDICT.md", verdict_text)

    # Main decision log
    log_lines = [
        "# Main Decision Log\n\n",
        f"**Script:** routeA_d4a3s_coverage_expansion.py\n",
        f"**Date:** 2026-05-24\n",
        f"**Seed:** {SEED}\n\n",
        f"## Summary\n\n",
        f"- G1 (Borda-over-HGB hits) queries: {n_g1_queries}\n",
        f"- G1 candidates: {g1_total}\n",
        f"- Initial A4C coverage: {100*coverage_init:.2f}%\n",
        f"- Dominant gap: {gap_counts.most_common(1)[0][0] if gap_counts else 'NONE'}\n",
        f"- Recomputed records: {recompute_count}\n",
        f"- Final verdict: {verdict_letter}. {verdict_desc}\n\n",
        f"## Key Decisions\n\n",
        f"1. Coverage expansion uses SMILES-level matching across all A4C data\n",
        f"2. All recomputed records marked {A4C_RECOMPUTED} (not original)\n",
        f"3. Chemical screening marked {PROXY_EVIDENCE}\n",
        f"4. Pre-registered criteria from D4A3R used for evaluation\n",
        f"5. No model retraining performed\n\n",
        f"## Files Written\n\n",
        f"- d4a3s_borda_gain_region.csv\n",
        f"- d4a3s_coverage_by_group.csv\n",
        f"- d4a3s_coverage_diagnosis.csv\n",
        f"- d4a3s_coverage_summary.csv\n",
        f"- d4a3s_join_repair_log.csv\n",
        f"- d4a3s_join_repair_coverage.csv\n",
        f"- d4a3s_a4c_recompute_attempt.csv\n",
        f"- d4a3s_a4c_recomputed_records.jsonl\n",
        f"- d4a3s_a4c_recompute_summary.md\n",
        f"- d4a3s_fragment_audit.csv\n",
        f"- d4a3s_fragment_gap_summary.md\n",
        f"- d4a3s_chemical_screening.csv\n",
        f"- d4a3s_chemical_screening_summary.md\n",
        f"- d4a3s_a4c_reeval_metrics.csv\n",
        f"- D4A3S_A4C_COVERAGE_VERDICT.md\n",
        f"- MAIN_DECISION_LOG.md\n",
    ]

    write_md(OUT / "MAIN_DECISION_LOG.md", "".join(log_lines))
    log.info("  Verdict: %s. %s", verdict_letter, verdict_desc)
    log.info("Part G complete")


# =========================================================================
# Main Pipeline
# =========================================================================

def main():
    """Execute D4A3S coverage expansion pipeline."""
    log.info("=" * 60)
    log.info("D4A3S: A4C Coverage Expansion for Borda Gain Region")
    log.info("=" * 60)
    log.info("Output: %s", OUT)
    os.makedirs(OUT, exist_ok=True)

    # Part A
    try:
        data = part_a_borda_gain_region()
        log.info("PART A complete")
    except Exception as e:
        log.error("PART A failed: %s", e)
        raise

    # Part B
    try:
        part_b_join_repair(data)
        log.info("PART B complete")
    except Exception as e:
        log.error("PART B failed: %s", e)
        raise

    # Part C
    try:
        recompute_data = part_c_a4c_recompute(data)
        log.info("PART C complete")
    except Exception as e:
        log.error("PART C failed: %s", e)
        recompute_data = None
        raise

    # Part D
    try:
        part_d_fragment_audit(data)
        log.info("PART D complete")
    except Exception as e:
        log.error("PART D failed: %s", e)
        raise

    # Part E
    try:
        part_e_chemical_screening(data)
        log.info("PART E complete")
    except Exception as e:
        log.error("PART E failed: %s", e)
        raise

    # Part F (conditional)
    try:
        reeval_data = part_f_a4c_reeval(data, recompute_data)
        log.info("PART F complete")
    except Exception as e:
        log.error("PART F failed: %s", e)
        reeval_data = False
        raise

    # Part G
    try:
        part_g_final_verdict(data, recompute_data, reeval_data)
        log.info("PART G complete")
    except Exception as e:
        log.error("PART G failed: %s", e)
        raise

    log.info("=" * 60)
    log.info("D4A3S pipeline complete. Output in %s", OUT)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
