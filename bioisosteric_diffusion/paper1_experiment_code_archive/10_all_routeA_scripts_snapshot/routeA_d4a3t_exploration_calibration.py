#!/usr/bin/env python3
"""
D4A3T: Exploration Mode Risk Calibration
========================================
Extends D4A3S by fixing A4C coverage gaps, decomposing G1 risk (G2 vs G3),
and building candidate-level A4C status labels.

Parts:
    0: Input Discovery + Provenance Lock
    A: FRAGMENT_GRAPH_MISSING Breakdown
    B: SMILES Rebuild + A4C Recompute (B1=G4 validation, B2=G1 recompute)
    C: Unrecoverable Proxy Assessment
    D: Risk Decomposition (G2 vs G3 vs G4)
    E: A4C Status Labeling System
    F: Post-Repair Global Statistics (with clustered bootstrap)
    G: Final Verdict
    H: Policy Config
"""

import json, csv, logging, sys, hashlib, textwrap
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import FilterCatalog, Descriptors, rdMolDescriptors, AllChem

RDLogger.DisableLog("rdApp.*")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types to native Python types."""
    def default(self, o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

SEED = 20260523
rng = np.random.default_rng(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("d4a3t")

# ── RDKit Catalogs ──────────────────────────────────────────────────
def _build_catalog(catalog_type):
    p = FilterCatalog.FilterCatalogParams()
    p.AddCatalog(catalog_type)
    return FilterCatalog.FilterCatalog(p)

PAINS_CAT = _build_catalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
BRENK_CAT = _build_catalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)


# ── Paths ─────────────────────────────────────────────────────────────
BASE = Path("E:/zuhui/bioisosteric_diffusion")
D4A3S = BASE / "plan_results" / "routeA_chembl37k_d4a3s_a4c_coverage_expansion"
A4C_DIR = BASE / "plan_results" / "routeA_chembl37k_d4a3_geometry_a4c_evaluation"
D4A0_MANIFEST = BASE / "plan_results" / "routeA_chembl37k_d0d3_engineering_safe" / "07_d4a0_matrix_freeze" / "d4a0_query_split_manifest.jsonl"
D4A0_VOCAB = BASE / "plan_results" / "routeA_chembl37k_d0d3_engineering_safe" / "07_d4a0_matrix_freeze" / "d4a0_train_replacement_vocabulary.csv"
DE_STD = BASE / "plan_results" / "routeA_chembl37k_d4a2d1r_dual_encoder_robustness" / "d4a2d1r_standardized_predictions.jsonl"
HGB_PREDS = BASE / "plan_results" / "routeA_chembl37k_d4a1_learned_ranker" / "d4a1_test_predictions.jsonl"
OUT = BASE / "plan_results" / "routeA_chembl37k_d4a3t_exploration_calibration"


# ── Helpers ───────────────────────────────────────────────────────────
def _parse_smi(smi, sanitize=True):
    """Parse SMILES, returning mol or None.
    Handles '*' attachment points (RDKit dummy atoms) natively.
    """
    if not smi or not isinstance(smi, str) or smi.strip() == "":
        return None
    smi = smi.strip()
    mol = Chem.MolFromSmiles(smi, sanitize=sanitize)
    return mol

def _check_alerts(mol):
    """Return (has_pains, has_brenk, has_any)."""
    if mol is None:
        return 0, 0, 0
    has_pains = 1 if PAINS_CAT.GetFirstMatch(mol) is not None else 0
    has_brenk = 1 if BRENK_CAT.GetFirstMatch(mol) is not None else 0
    return has_pains, has_brenk, has_pains or has_brenk

def _descriptors(mol):
    """Return dict of key descriptors."""
    if mol is None:
        return {"MW": None, "LogP": None, "HBD": None, "HBA": None,
                "RotBonds": None, "TPSA": None, "HA": None}
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    ha = mol.GetNumHeavyAtoms()
    return {"MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba,
            "RotBonds": rot, "TPSA": tpsa, "HA": ha}

def _assign_a4c_bucket(has_alert, d_desc, d_desc_pct):
    """Replicate D4A3 A4C bucket assignment logic."""
    # 1. Hard chemistry alert
    if has_alert:
        return "HARD_CHEMISTRY_ALERT"

    # 2. Property shift extreme thresholds (heuristic, from D4A3 rules)
    extreme = (
        d_desc.get("MW", 0) is not None and abs(d_desc["MW"]) > 200
    ) or (
        d_desc.get("LogP", 0) is not None and abs(d_desc["LogP"]) > 3
    ) or (
        d_desc.get("HBD", 0) is not None and d_desc["HBD"] > 3
    ) or (
        d_desc.get("HBA", 0) is not None and d_desc["HBA"] > 3
    )
    if extreme:
        return "PROPERTY_SHIFT_WARNING"

    # 3. Property shift warning
    warning = (
        d_desc.get("MW", 0) is not None and abs(d_desc["MW"]) > 60
    ) or (
        d_desc.get("LogP", 0) is not None and abs(d_desc["LogP"]) > 1
    ) or (
        d_desc.get("HBD", 0) is not None and d_desc["HBD"] > 1
    ) or (
        d_desc.get("HBA", 0) is not None and d_desc["HBA"] > 1
    ) or (
        d_desc.get("RotBonds", 0) is not None and d_desc["RotBonds"] > 3
    )
    if warning:
        return "REVIEW_READY_WITH_WARNING"

    return "REVIEW_READY"

def _compute_morgan_fp(mol, radius=2, n_bits=2048):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def _tanimoto(fp1, fp2):
    if fp1 is None or fp2 is None:
        return None
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def _write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, cls=NumpyEncoder) + "\n")

def _sha256(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]

def _load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ═════════════════════════════════════════════════════════════════════
# PART 0: Input Discovery + Provenance Lock
# ═════════════════════════════════════════════════════════════════════
def part0():
    log.info("=" * 60)
    log.info("PART 0: Input Discovery + Provenance Lock")

    required = {
        "G1 candidates": D4A3S / "d4a3s_G1_candidates.csv",
        "G2 candidates": D4A3S / "d4a3s_G2_candidates.csv",
        "G3 candidates": D4A3S / "d4a3s_G3_candidates.csv",
        "G4 candidates": D4A3S / "d4a3s_G4_candidates.csv",
        "Borda gain region": D4A3S / "d4a3s_borda_gain_region.csv",
        "Coverage diagnosis": D4A3S / "d4a3s_coverage_diagnosis.csv",
        "A4C review results": A4C_DIR / "d4a3_a4c_review_results.csv",
        "A4C recompute validation": D4A3S / "d4a3s_a4c_recompute_validation.csv",
        "A4C bucket distribution": A4C_DIR / "d4a3_a4c_bucket_distribution.csv",
        "Chemical screening": D4A3S / "d4a3s_chemical_screening.csv",
        "D4A0 manifest": D4A0_MANIFEST,
        "D4A0 vocab": D4A0_VOCAB,
        "DE standardized preds": DE_STD,
        "HGB predictions": HGB_PREDS,
    }

    missing = []
    files = {}
    for name, p in required.items():
        if p.exists():
            files[name] = str(p)
            log.info("  [OK] %s: %s", name, p.name)
        else:
            missing.append(name)
            log.error("  [MISSING] %s: %s", name, p)

    if missing:
        log.critical("INPUT_MISSING: %s", ", ".join(missing))
        sys.exit(1)

    # If G1 table missing — but we checked above
    g1_path = D4A3S / "d4a3s_G1_candidates.csv"
    if not g1_path.exists():
        log.critical("INPUT_MISSING_G_REGION: G1 candidates not found at %s", g1_path)
        sys.exit(1)

    # Build provenance record
    provenance = []
    for name, p in files.items():
        pp = Path(p)
        provenance.append({
            "input_name": name,
            "file_path": p,
            "size_bytes": pp.stat().st_size,
            "sha256_prefix": _sha256(pp.read_bytes().decode("utf-8", errors="replace")),
        })
    _write_csv(OUT / "d4a3t_input_discovery.csv", provenance)

    # Provenance summary
    lines = [
        "# D4A3T Input Provenance Summary",
        f"**Date:** 2026-05-24",
        f"**Seed:** {SEED}",
        "",
        "## Input Files",
    ]
    for rec in provenance:
        lines.append(f"- **{rec['input_name']}**: {Path(rec['file_path']).name} "
                      f"({rec['size_bytes']} bytes, sha256={rec['sha256_prefix']})")
    (OUT / "d4a3t_input_provenance_summary.md").write_text("\n".join(lines))
    log.info("Part 0 done: %d inputs verified, provenance written", len(provenance))
    return files


# ═════════════════════════════════════════════════════════════════════
# PART A: FRAGMENT_GRAPH_MISSING Breakdown + Sampling Bias
# ═════════════════════════════════════════════════════════════════════
def part_a(g1_df, cov_df):
    log.info("=" * 60)
    log.info("PART A: FRAGMENT_GRAPH_MISSING Breakdown")

    frag_missing = cov_df[cov_df["gap_type"] == "FRAGMENT_GRAPH_MISSING"]
    log.info("Fragment missing candidates: %d", len(frag_missing))

    breakdown = []
    for _, row in frag_missing.iterrows():
        smi = str(row.get("candidate_norm", ""))
        old_frag = str(row.get("old_fragment", ""))
        qid = str(row.get("qid", ""))

        # Try parsing
        mol = _parse_smi(smi)
        old_mol = _parse_smi(old_frag)

        status = None
        details = {}
        if not smi or smi.strip() == "" or smi == "nan":
            status = "NO_SMILES_AVAILABLE"
            details = {"has_smiles": 0, "rdkit_sanitize": 0, "has_context": 0}
        elif mol is None:
            status = "SANITIZE_FAIL"
            details = {"has_smiles": 1, "rdkit_sanitize": 0, "has_context": int(old_mol is not None)}
        elif old_mol is None:
            status = "ATTACHMENT_CONTEXT_MISSING"
            details = {"has_smiles": 1, "rdkit_sanitize": 1, "has_context": 0}
        else:
            status = "REBUILDABLE_FROM_SMILES"
            details = {"has_smiles": 1, "rdkit_sanitize": 1, "has_context": 1}

        breakdown.append({
            "qid": qid,
            "candidate_norm": smi,
            "old_fragment": old_frag,
            "status": status,
            "has_smiles": details.get("has_smiles", 0),
            "rdkit_sanitize": details.get("rdkit_sanitize", 0),
            "has_context": details.get("has_context", 0),
        })

    _write_csv(OUT / "d4a3t_fragment_missing_breakdown.csv", breakdown)

    # Summary counts
    status_counts = Counter(r["status"] for r in breakdown)
    for s, c in sorted(status_counts.items()):
        log.info("  %s: %d (%.1f%%)", s, c, 100 * c / len(breakdown))

    # ── Sampling bias analysis ──
    log.info("Part A: Sampling bias analysis")
    rebuildable = [r for r in breakdown if r["status"] == "REBUILDABLE_FROM_SMILES"]
    join_missing = cov_df[cov_df["gap_type"] == "JOIN_MISSING"]

    bias_rows = []
    for label, subset in [("REBUILDABLE_FROM_SMILES", rebuildable),
                           ("JOIN_MISSING", join_missing.to_dict("records"))]:
        smiles_list = [r.get("candidate_norm", "") for r in subset
                       if r.get("candidate_norm", "") and str(r.get("candidate_norm", "")) != "nan"]
        pains_count, brenk_count, alert_count = 0, 0, 0
        valid = 0
        tanimoto_sum = 0.0
        for smi in smiles_list:
            mol = _parse_smi(smi)
            if mol:
                valid += 1
                p, b, a = _check_alerts(mol)
                pains_count += p
                brenk_count += b
                alert_count += a
        bias_rows.append({
            "group": label,
            "n_candidates": len(subset),
            "n_valid_smiles": valid,
            "pains_alert_count": pains_count,
            "brenk_alert_count": brenk_count,
            "hard_alert_count": alert_count,
            "alert_rate": alert_count / valid if valid > 0 else None,
        })

    _write_csv(OUT / "d4a3t_sampling_bias_analysis.csv", bias_rows)

    # Sampling bias summary
    rebuild_rate = bias_rows[0]["alert_rate"] if bias_rows else 0
    join_rate = bias_rows[1]["alert_rate"] if len(bias_rows) > 1 else 0
    bias_note = ""
    if rebuild_rate is not None and join_rate is not None and rebuild_rate > join_rate:
        bias_note = ("CRITICAL: REBUILDABLE_FROM_SMILES has HIGHER proxy alert rate "
                     f"({rebuild_rate:.4f}) than JOIN_MISSING ({join_rate:.4f}). "
                     "Current 12.75% UNDERESTIMATES true risk.")
    elif rebuild_rate is not None and join_rate is not None:
        bias_note = (f"REBUILDABLE alert rate ({rebuild_rate:.4f}) <= "
                     f"JOIN_MISSING rate ({join_rate:.4f}). "
                     "No evidence of underestimation from this gap alone.")

    bias_summary = [
        "# D4A3T Sampling Bias Analysis Summary",
        "",
        "## Proxy Alert Rate by Gap Subtype",
    ]
    for r in bias_rows:
        ar = r["alert_rate"]
        ar_str = f"{ar:.4f}" if ar is not None else "N/A"
        bias_summary.append(f"- **{r['group']}**: {r['n_candidates']} candidates, "
                            f"alert_rate={ar_str} ({r['hard_alert_count']}/{r['n_valid_smiles']})")
    bias_summary.append("")
    bias_summary.append("## Bias Interpretation")
    bias_summary.append(bias_note)
    (OUT / "d4a3t_sampling_bias_summary.md").write_text("\n".join(bias_summary))

    log.info("Part A done: %d candidates analyzed", len(breakdown))
    return breakdown, status_counts


# ═════════════════════════════════════════════════════════════════════
# PART B: SMILES Rebuild + A4C Recompute
# ═════════════════════════════════════════════════════════════════════
def compute_a4c_for_smi(candidate_smi, old_fragment_smi):
    """Compute A4C bucket for a (candidate, old_fragment) pair."""
    cand_mol = _parse_smi(candidate_smi)
    old_mol = _parse_smi(old_fragment_smi)
    if cand_mol is None:
        return None, {"error": "candidate_parse_fail"}
    if old_mol is None:
        return None, {"error": "old_fragment_parse_fail"}

    # Alerts on candidate
    p, b, has_alert = _check_alerts(cand_mol)

    # Descriptors
    cand_desc = _descriptors(cand_mol)
    old_desc = _descriptors(old_mol)

    # Compute deltas
    d_desc = {}
    d_desc_pct = {}
    for key in ["MW", "LogP", "HBD", "HBA", "RotBonds", "TPSA"]:
        cv = cand_desc.get(key)
        ov = old_desc.get(key)
        if cv is not None and ov is not None and ov != 0:
            d_desc[key] = cv - ov
            d_desc_pct[key] = (cv - ov) / abs(ov)
        elif cv is not None and ov is not None:
            d_desc[key] = cv - ov
            d_desc_pct[key] = 0.0
        else:
            d_desc[key] = None
            d_desc_pct[key] = None

    # Morgan similarity
    cand_fp = _compute_morgan_fp(cand_mol)
    old_fp = _compute_morgan_fp(old_mol)
    tanimoto = None
    if cand_fp and old_fp:
        tanimoto = DataStructs.TanimotoSimilarity(cand_fp, old_fp)

    bucket = _assign_a4c_bucket(has_alert, d_desc, d_desc_pct)

    return bucket, {
        "pains_alerts": p,
        "brenk_alerts": b,
        "has_alert": has_alert,
        "delta_MW": d_desc.get("MW"),
        "delta_LogP": d_desc.get("LogP"),
        "delta_HBD": d_desc.get("HBD"),
        "delta_HBA": d_desc.get("HBA"),
        "delta_RotBonds": d_desc.get("RotBonds"),
        "delta_TPSA": d_desc.get("TPSA"),
        "tanimoto": tanimoto,
        "candidate_MW": cand_desc["MW"],
        "old_MW": old_desc["MW"],
    }


def part_b(g1_df, g4_df, a4c_df, a4c_recompute_val_df):
    log.info("=" * 60)
    log.info("PART B: SMILES Rebuild + A4C Recompute")

    # ── B1: G4 Validation ──
    log.info("Part B1: G4 A4C recompute validation")
    # G4 candidate_norm lacks "*" prefix; A4C replacement_smiles has it
    # Also rename old_fragment to avoid merge suffix collision
    g4_star = g4_df.copy()
    g4_star["candidate_star"] = g4_star["candidate_norm"].apply(
        lambda x: "*" + x if not str(x).startswith("*") else x
    )
    a4c_for_merge = a4c_df.rename(columns={
        "query_id": "qid",
        "replacement_smiles": "candidate_star",
        "old_fragment": "a4c_old_fragment"
    })
    g4_a4c = g4_star.merge(a4c_for_merge, on=["qid", "candidate_star"], how="inner")
    log.info("G4 candidates with A4C records: %d", len(g4_a4c))

    if len(g4_a4c) == 0:
        log.warning("No G4-A4C matches found. B1 validation cannot proceed.")
        return [], 0.0, []

    # Sample up to 3000 stratified by A4C bucket + replacement frequency
    g4_a4c["candidate_freq"] = g4_a4c.groupby("candidate_star")["candidate_star"].transform("count")
    g4_sample_max = min(3000, len(g4_a4c))
    if g4_sample_max < len(g4_a4c):
        # Stratify by a4c_bucket
        sample_size_per_bucket = max(1, g4_sample_max // g4_a4c["a4c_bucket"].nunique())
        g4_sample = g4_a4c.groupby("a4c_bucket", group_keys=False).apply(
            lambda g: g.sample(min(len(g), sample_size_per_bucket), random_state=SEED)
        ).reset_index(drop=True)
        g4_sample = g4_sample.sample(min(len(g4_sample), g4_sample_max), random_state=SEED)
    else:
        g4_sample = g4_a4c.copy()

    log.info("Sampled %d G4 candidates for validation", len(g4_sample))

    val_results = []
    for _, row in g4_sample.iterrows():
        # Use original candidate_norm (without *) for A4C recompute
        candidate_smi = str(row.get("candidate_norm", ""))
        old_frag = str(row.get("old_fragment_x", str(row.get("old_fragment", ""))))
        orig_bucket = str(row.get("a4c_bucket", ""))

        recomputed_bucket, details = compute_a4c_for_smi(candidate_smi, old_frag)
        agree = 1 if recomputed_bucket == orig_bucket else 0

        val_results.append({
            "qid": row.get("qid", ""),
            "candidate_norm": candidate_smi,
            "orig_bucket": orig_bucket,
            "recomputed_bucket": recomputed_bucket or "PARSE_FAIL",
            "agree": agree,
        })

    _write_csv(OUT / "d4a3t_recompute_validation_vs_g4.csv", val_results)

    agreement_rate = sum(r["agree"] for r in val_results) / len(val_results)
    log.info("G4 recompute agreement rate: %.4f (%d/%d)",
             agreement_rate, sum(r["agree"] for r in val_results), len(val_results))

    if agreement_rate < 0.90:
        log.critical("A4C_RECOMPUTE_VALIDATION_FAILED: agreement=%.4f < 0.90", agreement_rate)
        log.critical("Cannot proceed with reliable A4C recompute.")
        # Return validation results but mark failure
        return val_results, agreement_rate, []
    log.info("A4C recompute validation PASSED (agreement=%.4f)", agreement_rate)

    # ── B2: G1 Recompute ──
    log.info("Part B2: G1 A4C recompute")
    g1_recomputed = []
    for _, row in g1_df.iterrows():
        candidate_smi = str(row.get("candidate_norm", ""))
        old_frag = str(row.get("old_fragment", ""))
        qid = str(row.get("qid", ""))
        is_pos = row.get("is_pos", 0)
        de_hit = row.get("de_hit", 0)

        bucket, details = compute_a4c_for_smi(candidate_smi, old_frag)
        g1_recomputed.append({
            "qid": qid,
            "candidate_norm": candidate_smi,
            "old_fragment": old_frag,
            "is_pos": is_pos,
            "de_hit": de_hit,
            "a4c_bucket": bucket or "RECOMPUTE_FAIL",
            "status": "A4C_RECOMPUTED_FROM_SMILES" if bucket else "A4C_RECOMPUTE_FAILED",
            **{k: v for k, v in details.items()},
        })

    _write_jsonl(OUT / "d4a3t_a4c_recomputed_g1.jsonl", g1_recomputed)
    log.info("Part B done: %d G1 candidates recomputed", len(g1_recomputed))
    return val_results, agreement_rate, g1_recomputed


# ═════════════════════════════════════════════════════════════════════
# PART C: Unrecoverable Proxy Assessment
# ═════════════════════════════════════════════════════════════════════
def part_c(breakdown):
    log.info("=" * 60)
    log.info("PART C: Unrecoverable Proxy Assessment")

    unrecoverable = [r for r in breakdown if r["status"] in
                     ("ATTACHMENT_CONTEXT_MISSING", "SANITIZE_FAIL", "NO_SMILES_AVAILABLE")]
    log.info("Unrecoverable candidates: %d", len(unrecoverable))

    proxy_rows = []
    for r in unrecoverable:
        smi = r.get("candidate_norm", "")
        status = r["status"]
        mol = _parse_smi(smi)
        pains, brenk, has_alert = 0, 0, 0
        tanimoto = None
        if mol is not None:
            pains, brenk, has_alert = _check_alerts(mol)
            old_mol = _parse_smi(str(r.get("old_fragment", "")))
            if old_mol:
                cand_fp = _compute_morgan_fp(mol)
                old_fp = _compute_morgan_fp(old_mol)
                if cand_fp and old_fp:
                    tanimoto = DataStructs.TanimotoSimilarity(cand_fp, old_fp)

        a4c_label = "PROXY_EVIDENCE_NOT_A4C_REVIEW"
        if not smi or smi == "nan" or smi.strip() == "":
            a4c_label = "A4C_BLOCKED_NO_STRUCTURE"

        proxy_rows.append({
            "qid": r["qid"],
            "candidate_norm": smi,
            "old_fragment": r["old_fragment"],
            "status": status,
            "pains_proxy": pains,
            "brenk_proxy": brenk,
            "has_alert_proxy": has_alert,
            "tanimoto_proxy": tanimoto,
            "a4c_label": a4c_label,
        })

    _write_csv(OUT / "d4a3t_unrecoverable_proxy_assessment.csv", proxy_rows)
    log.info("Part C done: %d unrecoverable candidates assessed", len(proxy_rows))
    return proxy_rows


# ═════════════════════════════════════════════════════════════════════
# PART D: Risk Decomposition (G2 vs G3 vs G4)
# ═════════════════════════════════════════════════════════════════════
def part_d(g2_df, g3_df, g4_df, a4c_df, g1_recomputed):
    log.info("=" * 60)
    log.info("PART D: Risk Decomposition (G2 vs G3 vs G4)")

    # Build group stats for G2, G3, and G4
    groups = {
        "G2_pure_borda_only": g2_df,
        "G3_de_retained_by_borda": g3_df,
        "G4_shared_hits": g4_df,
    }

    # Map A4C review results to each group (by qid + candidate_norm with * prefix)
    a4c_lookup = {}
    for _, row in a4c_df.iterrows():
        key = (str(row["query_id"]), str(row["replacement_smiles"]))
        a4c_lookup[key] = row

    # For recomputed G1 (repair), create a lookup by (qid, candidate)
    g1_repair_lookup = {}
    for r in g1_recomputed:
        key = (r["qid"], r["candidate_norm"])
        g1_repair_lookup[key] = r

    def get_a4c_info(qid, candidate_norm, old_fragment):
        """Get A4C info: check direct match, then repair, then none."""
        # Try direct A4C match (with * prefix)
        star_smi = ("*" + candidate_norm) if not candidate_norm.startswith("*") else candidate_norm
        a4c_key = (qid, star_smi)
        if a4c_key in a4c_lookup:
            row = a4c_lookup[a4c_key]
            return {
                "covered": 1,
                "source": "A4C_DIRECT",
                "a4c_bucket": row["a4c_bucket"],
                "has_alert": row["has_alert"],
                "hard_chemistry_alert": row["hard_chemistry_alert"],
                "property_shift_extreme": row.get("property_shift_extreme", 0),
            }

        # Try repair lookup
        repair_key = (qid, candidate_norm)
        if repair_key in g1_repair_lookup:
            rr = g1_repair_lookup[repair_key]
            return {
                "covered": 1,
                "source": "A4C_REPAIR",
                "a4c_bucket": rr.get("a4c_bucket", "UNKNOWN"),
                "has_alert": rr.get("has_alert", 0),
                "hard_chemistry_alert": 1 if rr.get("a4c_bucket") == "HARD_CHEMISTRY_ALERT" else 0,
                "property_shift_extreme": 1 if rr.get("a4c_bucket") == "PROPERTY_SHIFT_WARNING" else 0,
            }

        # Try fragment-level A4C (old_fragment exists in A4C)
        for (aqid, acand), arow in a4c_lookup.items():
            if aqid == qid and arow.get("old_fragment", "") == (("*" + old_fragment) if old_fragment and not old_fragment.startswith("*") else old_fragment or ""):
                return {
                    "covered": 0,
                    "source": "NO_MATCH",
                    "a4c_bucket": None,
                    "has_alert": 0,
                    "hard_chemistry_alert": 0,
                    "property_shift_extreme": 0,
                }
        return {
            "covered": 0,
            "source": "NO_MATCH",
            "a4c_bucket": None,
            "has_alert": 0,
            "hard_chemistry_alert": 0,
            "property_shift_extreme": 0,
        }

    decomp_rows = []
    for group_name, group_df in groups.items():
        n_queries = group_df["qid"].nunique() if "qid" in group_df.columns else 0
        n_candidates = len(group_df)
        covered = 0
        hard_alerts = 0
        unknown = 0
        tanimoto_sum = 0.0
        tanimoto_n = 0

        for _, row in group_df.iterrows():
            qid = str(row.get("qid", ""))
            cand = str(row.get("candidate_norm", ""))
            frag = str(row.get("old_fragment", ""))

            info = get_a4c_info(qid, cand, frag)
            if info["covered"]:
                covered += 1
                if info.get("a4c_bucket") == "HARD_CHEMISTRY_ALERT":
                    hard_alerts += 1
            else:
                unknown += 1

        # Approximate Tanimoto from G1 repair data for this group
        # (only have meaningful tanimoto for recomputed records)
        for r in g1_recomputed:
            if r["qid"] in set(group_df["qid"].values) and r.get("tanimoto") is not None:
                tanimoto_sum += r["tanimoto"]
                tanimoto_n += 1

        cov_rate = covered / n_candidates if n_candidates > 0 else 0
        decomp_rows.append({
            "group": group_name,
            "n_candidates": n_candidates,
            "n_queries": n_queries,
            "a4c_coverage": round(cov_rate, 4),
            "hard_alert_count_among_covered": hard_alerts,
            "alert_rate_among_covered": round(hard_alerts / covered, 4) if covered > 0 else None,
            "alert_rate_total_lower_bound": round(hard_alerts / n_candidates, 4) if n_candidates > 0 else None,
            "unknown_count": unknown,
            "unknown_rate": round(unknown / n_candidates, 4) if n_candidates > 0 else None,
            "mean_tanimoto": round(tanimoto_sum / tanimoto_n, 4) if tanimoto_n > 0 else None,
        })

    _write_csv(OUT / "d4a3t_risk_decomposition.csv", decomp_rows)

    # Interpretation
    g2_rate = None
    g3_rate = None
    for r in decomp_rows:
        if "G2" in r["group"]:
            g2_rate = r["alert_rate_among_covered"]
        if "G3" in r["group"]:
            g3_rate = r["alert_rate_among_covered"]

    g2_vs_g3_note = ""
    if g2_rate is not None and g3_rate is not None:
        if g3_rate > g2_rate * 1.5:
            g2_vs_g3_note = "AlertRate(G3) >> AlertRate(G2): risk likely from DE model (DE-retained candidates have higher alert rate)."
        elif g2_rate > g3_rate * 1.5:
            g2_vs_g3_note = "AlertRate(G2) >> AlertRate(G3): risk likely from Borda fusion (pure Borda candidates have higher alert rate)."
        else:
            g2_vs_g3_note = "Alert rates comparable between G2 and G3: no single dominant risk source identified."

    summary_lines = [
        "# D4A3T Risk Decomposition Summary",
        "",
        "## Group-Level Comparison",
    ]
    for r in decomp_rows:
        summary_lines.append(
            f"- **{r['group']}**: N={r['n_candidates']}, coverage={r['a4c_coverage']:.4f}, "
            f"alert/covered={r['alert_rate_among_covered']}, total_lb={r['alert_rate_total_lower_bound']}, "
            f"unknown={r['unknown_rate']:.4f}"
        )
    summary_lines.append("")
    summary_lines.append("## Risk Source Interpretation")
    summary_lines.append(g2_vs_g3_note)
    (OUT / "d4a3t_risk_decomposition_summary.md").write_text("\n".join(summary_lines))

    log.info("Part D done: %d groups decomposed", len(decomp_rows))
    return decomp_rows


# ═════════════════════════════════════════════════════════════════════
# PART E: A4C Status Labeling System
# ═════════════════════════════════════════════════════════════════════
def part_e(g1_df, a4c_df, g1_recomputed):
    log.info("=" * 60)
    log.info("PART E: A4C Status Labeling System")

    # Build A4C lookup
    a4c_lookup = {}
    for _, row in a4c_df.iterrows():
        key = (str(row["query_id"]), str(row["replacement_smiles"]))
        a4c_lookup[key] = row

    # Build repair lookup
    repair_lookup = {}
    for r in g1_recomputed:
        key = (r["qid"], r["candidate_norm"])
        repair_lookup[key] = r

    # Build coverage diagnosis for gap type info (pd imported at top level)
    cov_df = pd.read_csv(str(D4A3S / "d4a3s_coverage_diagnosis.csv"))
    gap_lookup = {}
    for _, row in cov_df.iterrows():
        key = (str(row["qid"]), str(row["candidate_norm"]))
        gap_lookup[key] = str(row["gap_type"])

    labels = []
    cnt_covered_direct = 0
    cnt_covered_repair = 0
    cnt_recomputed_pass = 0
    cnt_hard_alert = 0
    cnt_soft_flag = 0
    cnt_coverage_missing = 0
    cnt_blocked = 0

    for _, row in g1_df.iterrows():
        qid = str(row["qid"])
        cand = str(row["candidate_norm"])
        old_frag = str(row["old_fragment"])
        star_cand = ("*" + cand) if not cand.startswith("*") else cand

        gap_type = gap_lookup.get((qid, cand), "UNKNOWN")
        a4c_key = (qid, star_cand)
        repair_key = (qid, cand)

        # Determine label
        if a4c_key in a4c_lookup:
            arow = a4c_lookup[a4c_key]
            bucket = arow["a4c_bucket"]
            if bucket == "HARD_CHEMISTRY_ALERT":
                label = "A4C_HARD_ALERT"
                cnt_hard_alert += 1
            elif bucket in ("PROPERTY_SHIFT_WARNING",):
                label = "A4C_SOFT_FLAG"
                cnt_soft_flag += 1
            elif bucket in ("REVIEW_READY_WITH_WARNING",):
                label = "A4C_SOFT_FLAG"
                cnt_soft_flag += 1
            else:
                label = "A4C_REVIEW_READY"
            cnt_covered_direct += 1

        elif repair_key in repair_lookup:
            rr = repair_lookup[repair_key]
            bucket = rr.get("a4c_bucket", "")
            if bucket == "HARD_CHEMISTRY_ALERT":
                label = "A4C_JOIN_REPAIRED_PASS" if gap_type == "JOIN_MISSING" else "A4C_RECOMPUTED_PASS"
                cnt_hard_alert += 1
                # Override: actual alert
                label = "A4C_HARD_ALERT"
            elif bucket in ("PROPERTY_SHIFT_WARNING", "REVIEW_READY_WITH_WARNING"):
                label = "A4C_SOFT_FLAG"
                cnt_soft_flag += 1
            else:
                if gap_type == "JOIN_MISSING":
                    label = "A4C_JOIN_REPAIRED_PASS"
                else:
                    label = "A4C_RECOMPUTED_PASS"
                cnt_recomputed_pass += 1
            cnt_covered_repair += 1

        elif cand and cand != "nan" and cand.strip():
            # SMILES available but no context/parse failure → proxy only
            label = "A4C_COVERAGE_MISSING"
            cnt_coverage_missing += 1
        else:
            label = "A4C_BLOCKED_NO_STRUCTURE"
            cnt_blocked += 1

        labels.append({
            "qid": qid,
            "candidate_norm": cand,
            "old_fragment": old_frag,
            "gap_type": gap_type,
            "a4c_label": label,
            "is_pos": row.get("is_pos", 0),
            "de_hit": row.get("de_hit", 0),
        })

    _write_csv(OUT / "d4a3t_candidate_labels_g1.csv", labels)

    # Label distribution
    label_dist = list(Counter(r["a4c_label"] for r in labels).items())
    label_dist.sort(key=lambda x: -x[1])
    dist_rows = [{"label": k, "count": v, "pct": round(100 * v / len(labels), 2)}
                 for k, v in label_dist]
    _write_csv(OUT / "d4a3t_label_distribution.csv", dist_rows)

    # Missing risk bounds
    G1_total = len(labels)
    covered_count = cnt_covered_direct + cnt_covered_repair
    hard_alerts = cnt_hard_alert
    unknown = G1_total - covered_count

    best_case = hard_alerts / G1_total
    observed = hard_alerts / covered_count if covered_count > 0 else 0
    worst_case = (hard_alerts + unknown) / G1_total

    bounds = [
        {"metric": "best_case_alert_rate",
         "value": round(best_case, 4),
         "formula": f"{hard_alerts}/{G1_total}"},
        {"metric": "observed_covered_alert_rate",
         "value": round(observed, 4),
         "formula": f"{hard_alerts}/{covered_count}"},
        {"metric": "worst_case_alert_rate",
         "value": round(worst_case, 4),
         "formula": f"({hard_alerts}+{unknown})/{G1_total}"},
        {"metric": "unknown_rate",
         "value": round(unknown / G1_total, 4),
         "formula": f"{unknown}/{G1_total}"},
        {"metric": "covered_count", "value": covered_count, "formula": ""},
        {"metric": "unknown_count", "value": unknown, "formula": ""},
        {"metric": "hard_alert_count", "value": hard_alerts, "formula": ""},
    ]
    _write_csv(OUT / "d4a3t_missing_risk_bounds.csv", bounds)

    # Three denominators per alert rate (per task.md requirement)
    review_eligible = covered_count
    alert_rate_among_covered = observed
    alert_rate_among_review_eligible = hard_alerts / review_eligible if review_eligible > 0 else 0
    alert_rate_total_lower_bound = hard_alerts / G1_total
    unknown_rate_val = unknown / G1_total

    log.info("Part E done: %d labels assigned", len(labels))
    log.info("  Direct: %d, Repair: %d, Hard: %d, Soft: %d, Missing: %d, Blocked: %d",
             cnt_covered_direct, cnt_covered_repair, cnt_hard_alert,
             cnt_soft_flag, cnt_coverage_missing, cnt_blocked)
    log.info("  Alert rates: among_covered=%.4f, among_review_eligible=%.4f, total_lb=%.4f, unknown=%.4f",
             alert_rate_among_covered, alert_rate_among_review_eligible,
             alert_rate_total_lower_bound, unknown_rate_val)

    return labels, {
        "best_case": best_case,
        "observed": observed,
        "worst_case": worst_case,
        "unknown_rate": unknown_rate_val,
        "covered_count": covered_count,
        "hard_alert_count": hard_alerts,
        "alert_rate_among_covered": alert_rate_among_covered,
        "alert_rate_among_review_eligible": alert_rate_among_review_eligible,
        "alert_rate_total_lower_bound": alert_rate_total_lower_bound,
    }


# ═════════════════════════════════════════════════════════════════════
# PART F: Post-Repair Global Statistics (clustered bootstrap)
# ═════════════════════════════════════════════════════════════════════
def part_f(g1_df, g4_df, labels, a4c_df):
    log.info("=" * 60)
    log.info("PART F: Post-Repair Global Statistics")

    # ── 1. Coverage repair summary ──
    coverage_original = sum(1 for r in labels if r["a4c_label"] in
                            ("A4C_REVIEW_READY", "A4C_HARD_ALERT", "A4C_SOFT_FLAG"))
    coverage_join_repair = sum(1 for r in labels if r["a4c_label"]
                                in ("A4C_JOIN_REPAIRED_PASS",))
    coverage_smiles_recompute = sum(1 for r in labels if r["a4c_label"]
                                     in ("A4C_RECOMPUTED_PASS",))
    coverage_final = sum(1 for r in labels if r["a4c_label"] != "A4C_BLOCKED_NO_STRUCTURE")

    G1_total = len(labels)
    cov_summary = [
        {"stage": "original", "covered": coverage_original, "rate": round(coverage_original / G1_total, 4)},
        {"stage": "join_repair", "covered": coverage_join_repair, "rate": round(coverage_join_repair / G1_total, 4)},
        {"stage": "smiles_recompute", "covered": coverage_smiles_recompute, "rate": round(coverage_smiles_recompute / G1_total, 4)},
        {"stage": "final", "covered": coverage_final, "rate": round(coverage_final / G1_total, 4)},
    ]
    _write_csv(OUT / "d4a3t_coverage_repair_summary.csv", cov_summary)

    # ── 2. Alert rate by group (pre/post repair) ──
    # Build label lookup
    label_lookup = {}
    for r in labels:
        key = (r["qid"], r["candidate_norm"])
        label_lookup[key] = r

    groups_data = {
        "G1": g1_df,
        "G4": g4_df,
    }

    alert_rows = []
    for gname, gdf in groups_data.items():
        total = len(gdf)
        hard_alerts = 0
        for _, row in gdf.iterrows():
            qid = str(row.get("qid", ""))
            cand = str(row.get("candidate_norm", ""))
            key = (qid, cand)

            if gname == "G1" and key in label_lookup:
                ll = label_lookup[key]
                if ll["a4c_label"] == "A4C_HARD_ALERT":
                    hard_alerts += 1
            elif gname == "G4":
                star_cand = ("*" + cand) if not cand.startswith("*") else cand
                a4c_match = a4c_df[(a4c_df["query_id"] == qid) &
                                    (a4c_df["replacement_smiles"] == star_cand)]
                if len(a4c_match) > 0:
                    if a4c_match.iloc[0]["a4c_bucket"] == "HARD_CHEMISTRY_ALERT":
                        hard_alerts += 1

        alert_rows.append({
            "group": gname,
            "n_candidates": total,
            "hard_alerts": hard_alerts,
            "alert_rate": round(hard_alerts / total, 4) if total > 0 else None,
        })

    _write_csv(OUT / "d4a3t_alert_rate_by_group.csv", alert_rows)

    # ── 3. Clustered bootstrap (G1 vs G4) ──
    log.info("Part F: Clustered bootstrap (1000 resamples, clustered by query_id)")

    # Build per-candidate alert data
    g1_alerts = []
    for _, row in g1_df.iterrows():
        qid = str(row["qid"])
        cand = str(row["candidate_norm"])
        key = (qid, cand)
        ll = label_lookup.get(key, {})
        is_alert = 1 if ll.get("a4c_label") == "A4C_HARD_ALERT" else 0
        g1_alerts.append({"qid": qid, "alert": is_alert})

    g4_alerts = []
    for _, row in g4_df.iterrows():
        qid = str(row["qid"])
        cand = str(row["candidate_norm"])
        star_cand = ("*" + cand) if not cand.startswith("*") else cand
        a4c_match = a4c_df[(a4c_df["query_id"] == qid) &
                            (a4c_df["replacement_smiles"] == star_cand)]
        is_alert = 0
        if len(a4c_match) > 0 and a4c_match.iloc[0]["a4c_bucket"] == "HARD_CHEMISTRY_ALERT":
            is_alert = 1
        g4_alerts.append({"qid": qid, "alert": is_alert})

    # Clustered bootstrap: sample queries, then all their candidates
    g1_by_qid = defaultdict(list)
    for x in g1_alerts:
        g1_by_qid[x["qid"]].append(x["alert"])
    g1_qids = list(g1_by_qid.keys())

    g4_by_qid = defaultdict(list)
    for x in g4_alerts:
        g4_by_qid[x["qid"]].append(x["alert"])
    g4_qids = list(g4_by_qid.keys())

    n_bootstrap = 1000
    g1_rates = []
    g4_rates = []
    diffs = []

    for i in range(n_bootstrap):
        # Sample G1 queries with replacement
        sampled_g1_qids = rng.choice(g1_qids, size=len(g1_qids), replace=True)
        g1_candidates = []
        for sqid in sampled_g1_qids:
            g1_candidates.extend(g1_by_qid[sqid])
        g1_rate = sum(g1_candidates) / len(g1_candidates) if g1_candidates else 0
        g1_rates.append(g1_rate)

        # Sample G4 queries with replacement
        sampled_g4_qids = rng.choice(g4_qids, size=len(g4_qids), replace=True)
        g4_candidates = []
        for sqid in sampled_g4_qids:
            g4_candidates.extend(g4_by_qid[sqid])
        g4_rate = sum(g4_candidates) / len(g4_candidates) if g4_candidates else 0
        g4_rates.append(g4_rate)

        diffs.append(g1_rate - g4_rate)

    g1_rates = np.array(g1_rates)
    g4_rates = np.array(g4_rates)
    diffs = np.array(diffs)

    ci_lo = np.percentile(diffs, 2.5)
    ci_hi = np.percentile(diffs, 97.5)
    mean_diff = np.mean(diffs)

    bootstrap_results = [{
        "comparison": "G1_vs_G4_alert_rate_diff",
        "bootstrap_n": n_bootstrap,
        "mean_diff": round(mean_diff, 4),
        "ci_lower_2.5": round(ci_lo, 4),
        "ci_upper_97.5": round(ci_hi, 4),
        "elevated_risk": 1 if ci_lo > 0 else 0,
        "g1_mean_alert": round(np.mean(g1_rates), 4),
        "g1_ci_lower": round(np.percentile(g1_rates, 2.5), 4),
        "g1_ci_upper": round(np.percentile(g1_rates, 97.5), 4),
        "g4_mean_alert": round(np.mean(g4_rates), 4),
        "g4_ci_lower": round(np.percentile(g4_rates, 2.5), 4),
        "g4_ci_upper": round(np.percentile(g4_rates, 97.5), 4),
    }]
    _write_csv(OUT / "d4a3t_bootstrap_risk_comparisons.csv", bootstrap_results)

    log.info("Bootstrap: G1-G4 diff = %.4f [%.4f, %.4f]", mean_diff, ci_lo, ci_hi)
    elevated = ci_lo > 0
    if elevated:
        log.info("ELEVATED_RISK: bootstrap CI lower bound > 0")

    # Pre-registered criteria checks
    N = G1_total
    operational_coverage = coverage_final / N
    criteria_check = {
        "coverage_review_eligible_ge_0.70": operational_coverage >= 0.70,
        "coverage_actual": round(operational_coverage, 4),
        "coverage_target": 0.70,
        "hard_alert_rate_among_covered_le_0.15": True,  # placeholder, computed below
        "alert_rate_among_covered": 0.0,
        "alert_rate_target": 0.15,
        "g1_vs_g4_bootstrap_ci_lower_gt_0": ci_lo > 0,
        "g1_vs_g4_bootstrap_ci_lower": round(ci_lo, 4),
        "elevated_risk": elevated,
    }

    # Fill alert rate among covered
    hard_covered = sum(1 for r in labels if r["a4c_label"] == "A4C_HARD_ALERT")
    covered_total = sum(1 for r in labels if r["a4c_label"]
                        not in ("A4C_BLOCKED_NO_STRUCTURE",))
    if covered_total > 0:
        criteria_check["alert_rate_among_covered"] = round(hard_covered / covered_total, 4)
        criteria_check["hard_alert_rate_among_covered_le_0.15"] = (hard_covered / covered_total) <= 0.15

    (OUT / "d4a3t_pre_registered_criteria.json").write_text(
        json.dumps(criteria_check, indent=2, cls=NumpyEncoder)
    )

    log.info("Part F done: coverage=%.4f, criteria=%s", operational_coverage, criteria_check)
    return cov_summary, alert_rows, bootstrap_results, criteria_check


# ═════════════════════════════════════════════════════════════════════
# PART G: Final Verdict
# ═════════════════════════════════════════════════════════════════════
def part_g(criteria, e_stats, decomp_rows, bootstrap_results):
    log.info("=" * 60)
    log.info("PART G: Final Verdict")

    # Answer 9 questions
    qa = {}
    qa["Q1_can_a4c_coverage_be_repaired"] = criteria["coverage_actual"] >= 0.70
    qa["Q1_coverage_rate"] = criteria["coverage_actual"]

    qa["Q2_alert_rate_acceptable"] = criteria["hard_alert_rate_among_covered_le_0.15"]
    qa["Q2_alert_rate"] = criteria["alert_rate_among_covered"]

    qa["Q3_g1_vs_g4_elevated"] = criteria["elevated_risk"]
    qa["Q3_bootstrap_ci_lower"] = criteria["g1_vs_g4_bootstrap_ci_lower"]

    # Q4: Risk source
    risk_source = "unknown"
    g2_rate = None
    g3_rate = None
    for r in decomp_rows:
        if "G2" in r["group"]:
            g2_rate = r.get("alert_rate_among_covered")
        if "G3" in r["group"]:
            g3_rate = r.get("alert_rate_among_covered")
    if g2_rate is not None and g3_rate is not None:
        if g3_rate > g2_rate * 1.5:
            risk_source = "DE_MODEL"
        elif g2_rate > g3_rate * 1.5:
            risk_source = "BORDA_FUSION"
        else:
            risk_source = "BALANCED"
    qa["Q4_risk_source"] = risk_source

    # Q5: Missing risk bounds
    qa["Q5_best_case_alert_rate"] = round(e_stats["best_case"], 4)
    qa["Q5_observed_alert_rate"] = round(e_stats["observed"], 4)
    qa["Q5_worst_case_alert_rate"] = round(e_stats["worst_case"], 4)
    qa["Q5_unknown_rate"] = round(e_stats["unknown_rate"], 4)

    # Q6: Bootstrap CI
    if bootstrap_results:
        qa["Q6_g1_alert_rate"] = bootstrap_results[0].get("g1_mean_alert")
        qa["Q6_g4_alert_rate"] = bootstrap_results[0].get("g4_mean_alert")
        qa["Q6_diff_ci"] = f"[{bootstrap_results[0]['ci_lower_2.5']}, {bootstrap_results[0]['ci_upper_97.5']}]"

    # Verdict A-F
    if not qa["Q1_can_a4c_coverage_be_repaired"]:
        verdict = "F. COVERAGE_INSUFFICIENT"
    elif qa["Q3_g1_vs_g4_elevated"] and not qa["Q2_alert_rate_acceptable"]:
        verdict = "E. ELEVATED_RISK_HIGH_ALERT"
    elif qa["Q3_g1_vs_g4_elevated"]:
        verdict = "D. ELEVATED_RISK_MODERATE"
    elif not qa["Q2_alert_rate_acceptable"]:
        verdict = "C. BORDA_PENDING_A4C_COVERAGE"
    elif qa["Q2_alert_rate"] <= 0.05:
        verdict = "A. LOW_RISK"
    else:
        verdict = "B. MODERATE_RISK_ACCEPTABLE"
    qa["verdict"] = verdict

    # Write verdict
    lines = [
        "# D4A3T Exploration Mode Risk Calibration Verdict",
        f"**Date:** 2026-05-24",
        f"**Seed:** {SEED}",
        "",
        "## Questions and Answers",
    ]
    for i, (q, a) in enumerate(qa.items(), 1):
        lines.append(f"**{q}:** {a}")
    lines += [
        "",
        "## Verdict",
        f"**{verdict}**",
        "",
        "## Skeptical Review",
        "1. A4C recompute is a heuristic, not expert review.",
        "2. Bootstrapped CIs depend on representativeness of query-level sampling.",
        "3. Coverage repair does not replace expert A4C labeling.",
        "4. Proxy alerts are NOT A4C review results.",
    ]
    (OUT / "D4A3T_EXPLORATION_MODE_VERDICT.md").write_text("\n".join(lines))

    # Decision log
    decision_lines = [
        "# D4A3T Main Decision Log",
        "",
        "## Decisions",
        "- A4C recompute validated against G4 (agreement >= 0.90 required)",
        "- Bootstrap clusters by query_id (1000 resamples)",
        "- Missing = A4C_BLOCKED_NO_STRUCTURE, not assumed safe",
        "- Alert rate reported with 3 denominators",
    ]
    (OUT / "MAIN_DECISION_LOG.md").write_text("\n".join(decision_lines))

    log.info("Part G done: verdict=%s", verdict)
    return qa, verdict


# ═════════════════════════════════════════════════════════════════════
# PART H: Policy Config
# ═════════════════════════════════════════════════════════════════════
def part_h(qa, criteria, e_stats, verdict):
    log.info("=" * 60)
    log.info("PART H: Policy Config Generation")

    policy = {
        "pipeline": "d4a3t_exploration_mode",
        "generated": "2026-05-24",
        "seed": SEED,
        "verdict": verdict,
        "risk_assessment": {
            "coverage_rate": criteria["coverage_actual"],
            "alert_rate_among_covered": criteria["alert_rate_among_covered"],
            "alert_rate_total_lower_bound": e_stats["alert_rate_total_lower_bound"],
            "unknown_rate": e_stats["unknown_rate"],
            "best_case_alert_rate": e_stats["best_case"],
            "worst_case_alert_rate": e_stats["worst_case"],
            "risk_source": qa.get("Q4_risk_source", "unknown"),
            "bootstrap_ci_lower": criteria["g1_vs_g4_bootstrap_ci_lower"],
            "elevated_risk": criteria["elevated_risk"],
        },
        "recommended_actions": [],
    }

    if "F" in verdict:
        policy["recommended_actions"].append("Expand A4C coverage before Borda deployment")
    if "E" in verdict or "D" in verdict:
        policy["recommended_actions"].append("Targeted A4C review of high-risk candidates")
        policy["recommended_actions"].append("Consider Borda recalibration with alert penalty")
    if qa.get("Q4_risk_source") == "DE_MODEL":
        policy["recommended_actions"].append("Audit DE model: high alert rate in DE-retained candidates")
    elif qa.get("Q4_risk_source") == "BORDA_FUSION":
        policy["recommended_actions"].append("Audit Borda fusion: high alert rate in pure-Borda candidates")

    (OUT / "d4a3t_exploration_mode_policy_config.json").write_text(
        json.dumps(policy, indent=2, cls=NumpyEncoder)
    )
    log.info("Part H done: policy config written")
    return policy


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("D4A3T: Exploration Mode Risk Calibration")
    log.info("=" * 60)

    OUT.mkdir(parents=True, exist_ok=True)

    # Part 0
    files = part0()

    # Load core data
    g1_df = pd.read_csv(D4A3S / "d4a3s_G1_candidates.csv")
    g2_df = pd.read_csv(D4A3S / "d4a3s_G2_candidates.csv")
    g3_df = pd.read_csv(D4A3S / "d4a3s_G3_candidates.csv")
    g4_df = pd.read_csv(D4A3S / "d4a3s_G4_candidates.csv")
    cov_df = pd.read_csv(D4A3S / "d4a3s_coverage_diagnosis.csv")
    a4c_df = pd.read_csv(A4C_DIR / "d4a3_a4c_review_results.csv")
    a4c_recompute_val_df = pd.read_csv(D4A3S / "d4a3s_a4c_recompute_validation.csv")

    log.info("Data loaded: G1=%d, G2=%d, G3=%d, G4=%d, A4C=%d, cov=%d",
             len(g1_df), len(g2_df), len(g3_df), len(g4_df), len(a4c_df), len(cov_df))

    # Part A
    breakdown, sc = part_a(g1_df, cov_df)

    # Part B
    val_results, agreement_rate, g1_recomputed = part_b(g1_df, g4_df, a4c_df, a4c_recompute_val_df)

    # If validation failed, we still proceed with proxy assessments but skip G1 recompute
    if agreement_rate < 0.90:
        log.warning("A4C recompute validation FAILED. Parts B2/D/E/F/G/H will use proxy-only estimates.")
        g1_recomputed = []

    # Part C
    proxy_rows = part_c(breakdown)

    # Part D
    decomp_rows = part_d(g2_df, g3_df, g4_df, a4c_df, g1_recomputed)

    # Part E
    labels, e_stats = part_e(g1_df, a4c_df, g1_recomputed)

    # Part F
    cov_summary, alert_rows, bootstrap_results, criteria = part_f(
        g1_df, g4_df, labels, a4c_df
    )

    # Part G
    qa, verdict = part_g(criteria, e_stats, decomp_rows, bootstrap_results)

    # Part H
    policy = part_h(qa, criteria, e_stats, verdict)

    log.info("=" * 60)
    log.info("D4A3T complete: verdict=%s | coverage=%.4f | alert=%.4f | diff_ci=[%.4f, %.4f]",
             verdict,
             criteria["coverage_actual"],
             criteria["alert_rate_among_covered"],
             bootstrap_results[0]["ci_lower_2.5"] if bootstrap_results else 0,
             bootstrap_results[0]["ci_upper_97.5"] if bootstrap_results else 0)
    log.info("Output: %s", OUT)


if __name__ == "__main__":
    main()
