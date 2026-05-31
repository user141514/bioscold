#!/usr/bin/env python3
"""
A4C Semantic Review v0 — Fragment-level medchem review table for B5 geometry candidates.

Usage:
  python core/scripts/a4c_semantic_review_v0.py \
    --manifest plan_results/P2A_M2R_UNIFIED_BENCHMARK/m2r_raw_800_frozen_manifest.json \
    --b5_pool plan_results/H1_F2R_POOL_FUSION_PILOT/h1_b5_candidate_pool_541.jsonl \
    --n_cases 10 \
    --selected_only \
    --output plan_results/A4C_SEMANTIC_REVIEW_V0/smoke_test/

A4C v0 scope:
  - RMSD bucket, clean flag, clash score, source method
  - Basic fragment RDKit properties (MolWt, LogP, TPSA, HBD, HBA, RotB, rings, charge)
  - PAINS / Brenk alerts (graceful fallback if unavailable)
  - review_bucket assignment (rule-based, no oracle RMSD)

NOT in v0:
  - property shift (no replacement mol)
  - shape similarity (no candidate 3D coords)
  - pharmacophore compatibility
"""

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

# ── RDKit imports ──────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

# ── FilterCatalog (PAINS / Brenk) ─────────────────────────────────────────────
PAINS_OK = False
BRENK_OK = False
pains_catalog = None
brenk_catalog = None

if RDKIT_OK:
    try:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
        params_pains = FilterCatalogParams()
        params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        pains_catalog = FilterCatalog(params_pains)
        PAINS_OK = True
    except Exception:
        pass

    try:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
        params_brenk = FilterCatalogParams()
        params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        brenk_catalog = FilterCatalog(params_brenk)
        BRENK_OK = True
    except Exception:
        pass


# ── Review bucket rules ────────────────────────────────────────────────────────
def assign_review_bucket(rmsd, clean, clash_score, pains_hit, brenk_hit):
    """Priority-ordered bucket assignment. No oracle RMSD used."""
    has_medchem_warning = (pains_hit is True) or (brenk_hit is True)

    if rmsd is None:
        return "GENERATION_FAIL"
    if (clean is False) or (clash_score is not None and clash_score > 0.5):
        return "HARD_CLASH"
    if clean is True and (clash_score is None or clash_score <= 0.5) and rmsd < 1.0 and not has_medchem_warning:
        return "REVIEW_READY"
    if clean is True and (clash_score is None or clash_score <= 0.5) and rmsd < 1.0 and has_medchem_warning:
        return "GEOMETRY_OK_PROPERTY_SHIFT"
    if has_medchem_warning:
        return "NEEDS_MEDCHEM_REVIEW"
    if clean is True and (clash_score is None or clash_score <= 0.5) and 1.0 <= rmsd < 2.0:
        return "CLEAN_BUT_SEMANTIC_WARNING"
    if rmsd >= 2.0 and clean is True:
        return "POOR_GEOMETRY"
    return "UNCLASSIFIED"


def rmsd_to_bucket(rmsd):
    if rmsd is None:
        return "null"
    if rmsd < 1.0:
        return "<1.0"
    if rmsd < 2.0:
        return "1.0-2.0"
    return ">2.0"


# ── Fragment property computation ──────────────────────────────────────────────
def compute_fragment_props(mol):
    """Compute basic RDKit descriptors on a fragment mol. Returns dict."""
    if mol is None:
        return {k: None for k in [
            "MolWt", "LogP", "TPSA", "HBD", "HBA", "RotB",
            "ring_count", "aromatic_ring_count", "formal_charge"
        ]}
    try:
        return {
            "MolWt": round(Descriptors.MolWt(mol), 4),
            "LogP": round(Crippen.MolLogP(mol), 4),
            "TPSA": round(Descriptors.TPSA(mol), 4),
            "HBD": Lipinski.NumHDonors(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "RotB": Lipinski.NumRotatableBonds(mol),
            "ring_count": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(mol),
            "formal_charge": Chem.GetFormalCharge(mol),
        }
    except Exception as e:
        return {k: None for k in [
            "MolWt", "LogP", "TPSA", "HBD", "HBA", "RotB",
            "ring_count", "aromatic_ring_count", "formal_charge"
        ]}


def run_filters(mol):
    """Run PAINS and Brenk filters. Returns (pains_hit, brenk_hit, flags)."""
    if mol is None:
        return None, None, ["MOL_NONE"]

    flags = []
    pains_hit = None
    brenk_hit = None

    if PAINS_OK and pains_catalog is not None:
        try:
            pains_hit = pains_catalog.HasMatch(mol)
        except Exception:
            flags.append("PAINS_ERROR")
    else:
        flags.append("FILTERCATALOG_UNAVAILABLE")

    if BRENK_OK and brenk_catalog is not None:
        try:
            brenk_hit = brenk_catalog.HasMatch(mol)
        except Exception:
            flags.append("BRENK_ERROR")
    else:
        if "FILTERCATALOG_UNAVAILABLE" not in flags:
            flags.append("FILTERCATALOG_UNAVAILABLE")

    if pains_hit is True:
        flags.append("PAINS")
    if brenk_hit is True:
        flags.append("BRENK")

    return pains_hit, brenk_hit, flags


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="A4C Semantic Review v0")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--b5_pool", required=True)
    parser.add_argument("--n_cases", type=int, default=None,
                        help="Limit to first N cases (for smoke test)")
    parser.add_argument("--selected_only", action="store_true",
                        help="Output only the energy-selected candidate per case")
    parser.add_argument("--output", required=True,
                        help="Output directory")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    b5_pool_path = Path(args.b5_pool)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = {}

    # ── Q1: manifest exists ────────────────────────────────────────────────────
    checks["Q1_manifest_exists"] = "PASS" if manifest_path.exists() else "FAIL"
    if not manifest_path.exists():
        print(f"FAIL: manifest not found: {manifest_path}")
        sys.exit(1)

    # ── Q2: b5_pool exists ────────────────────────────────────────────────────
    checks["Q2_b5_pool_exists"] = "PASS" if b5_pool_path.exists() else "FAIL"
    if not b5_pool_path.exists():
        print(f"FAIL: b5_pool not found: {b5_pool_path}")
        sys.exit(1)

    # ── Q3: parse manifest ────────────────────────────────────────────────────
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest_data = json.load(f)
        entries = manifest_data.get("entries", [])
        manifest_by_id = {e["case_id"]: e for e in entries}
        checks["Q3_manifest_parseable"] = f"PASS (n_entries={len(entries)})"
    except Exception as e:
        checks["Q3_manifest_parseable"] = f"FAIL: {e}"
        print(f"FAIL parsing manifest: {e}")
        sys.exit(1)

    # ── Q4: parse b5_pool ─────────────────────────────────────────────────────
    try:
        b5_pool = []
        with open(b5_pool_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    b5_pool.append(json.loads(line))
        checks["Q4_b5_pool_parseable"] = f"PASS (n_rows={len(b5_pool)})"
    except Exception as e:
        checks["Q4_b5_pool_parseable"] = f"FAIL: {e}"
        print(f"FAIL parsing b5_pool: {e}")
        sys.exit(1)

    # ── Limit cases ───────────────────────────────────────────────────────────
    if args.n_cases is not None:
        b5_pool = b5_pool[:args.n_cases]

    # ── Q5: case_id join ──────────────────────────────────────────────────────
    missing_ids = [r["case_id"] for r in b5_pool if r["case_id"] not in manifest_by_id]
    if missing_ids:
        checks["Q5_case_id_join"] = f"WARN: {len(missing_ids)} case_ids not in manifest"
    else:
        checks["Q5_case_id_join"] = f"PASS (all {len(b5_pool)} case_ids found)"

    # ── Q6: fragment_mol deserialization ──────────────────────────────────────
    deser_ok = 0
    deser_fail = 0
    for r in b5_pool[:3]:
        cid = r["case_id"]
        entry = manifest_by_id.get(cid)
        if entry and entry.get("fragment_mol_pkl"):
            try:
                mol = Chem.Mol(bytes.fromhex(entry["fragment_mol_pkl"]))
                if mol is not None:
                    deser_ok += 1
                else:
                    deser_fail += 1
            except Exception:
                deser_fail += 1
    if deser_fail == 0:
        checks["Q6_fragment_mol_deser"] = f"PASS (tested {deser_ok} mols)"
    else:
        checks["Q6_fragment_mol_deser"] = f"WARN: {deser_fail} deser failures in first 3"

    # ── Q7: RDKit descriptors ─────────────────────────────────────────────────
    checks["Q7_rdkit_descriptors"] = "PASS" if RDKIT_OK else "FAIL: RDKit not importable"

    # ── Q8/Q9: FilterCatalog ──────────────────────────────────────────────────
    checks["Q8_PAINS_catalog"] = "PASS" if PAINS_OK else "WARN: PAINS FilterCatalog unavailable"
    checks["Q9_Brenk_catalog"] = "PASS" if BRENK_OK else "WARN: Brenk FilterCatalog unavailable"

    # ── Process candidates ────────────────────────────────────────────────────
    rows = []
    prop_fail_count = 0

    for pool_row in b5_pool:
        cid = pool_row["case_id"]
        entry = manifest_by_id.get(cid, {})

        # Deserialize fragment mol
        mol = None
        if entry.get("fragment_mol_pkl"):
            try:
                mol = Chem.Mol(bytes.fromhex(entry["fragment_mol_pkl"]))
            except Exception:
                mol = None

        # Compute fragment properties (same for all candidates of this case)
        props = compute_fragment_props(mol)
        if props["MolWt"] is None:
            prop_fail_count += 1

        # Run filters (same for all candidates of this case)
        pains_hit, brenk_hit, filter_flags = run_filters(mol)

        # Build medchem_warning_flags string
        medchem_flags = [f for f in filter_flags if f not in ("PAINS", "BRENK")]
        if pains_hit is True:
            medchem_flags.append("PAINS")
        if brenk_hit is True:
            medchem_flags.append("BRENK")
        medchem_warning_flags = ",".join(medchem_flags) if medchem_flags else "none"

        # Candidate selection
        n_cands = pool_row.get("n_candidates", 0)
        cand_rmsds = pool_row.get("candidate_rmsds") or []
        cand_clean = pool_row.get("candidate_clean") or []
        cand_clash = pool_row.get("candidate_clash_scores") or []
        best_idx = pool_row.get("best_energy_idx")

        if args.selected_only:
            # Only the energy-selected candidate
            if best_idx is None or best_idx >= len(cand_rmsds):
                # No valid selected candidate
                row = {
                    "case_id": cid,
                    "candidate_id": f"{cid}_c_none",
                    "source_method": pool_row.get("source_method", "B5"),
                    "selected_rank": -1,
                    "rmsd": None,
                    "rmsd_bucket": "null",
                    "clean": None,
                    "clash_score": None,
                    **props,
                    "PAINS_hit": pains_hit,
                    "Brenk_hit": brenk_hit,
                    "medchem_warning_flags": medchem_warning_flags,
                    "review_bucket": "GENERATION_FAIL",
                }
                rows.append(row)
            else:
                rmsd = cand_rmsds[best_idx] if best_idx < len(cand_rmsds) else None
                clean = cand_clean[best_idx] if best_idx < len(cand_clean) else None
                clash = cand_clash[best_idx] if best_idx < len(cand_clash) else None
                bucket = assign_review_bucket(rmsd, clean, clash, pains_hit, brenk_hit)
                row = {
                    "case_id": cid,
                    "candidate_id": f"{cid}_c{best_idx}",
                    "source_method": pool_row.get("source_method", "B5"),
                    "selected_rank": 0,
                    "rmsd": rmsd,
                    "rmsd_bucket": rmsd_to_bucket(rmsd),
                    "clean": clean,
                    "clash_score": clash,
                    **props,
                    "PAINS_hit": pains_hit,
                    "Brenk_hit": brenk_hit,
                    "medchem_warning_flags": medchem_warning_flags,
                    "review_bucket": bucket,
                }
                rows.append(row)
        else:
            # All candidates
            for idx in range(n_cands):
                rmsd = cand_rmsds[idx] if idx < len(cand_rmsds) else None
                clean = cand_clean[idx] if idx < len(cand_clean) else None
                clash = cand_clash[idx] if idx < len(cand_clash) else None
                is_selected = (idx == best_idx)
                bucket = assign_review_bucket(rmsd, clean, clash, pains_hit, brenk_hit)
                row = {
                    "case_id": cid,
                    "candidate_id": f"{cid}_c{idx}",
                    "source_method": pool_row.get("source_method", "B5"),
                    "selected_rank": 0 if is_selected else -1,
                    "rmsd": rmsd,
                    "rmsd_bucket": rmsd_to_bucket(rmsd),
                    "clean": clean,
                    "clash_score": clash,
                    **props,
                    "PAINS_hit": pains_hit,
                    "Brenk_hit": brenk_hit,
                    "medchem_warning_flags": medchem_warning_flags,
                    "review_bucket": bucket,
                }
                rows.append(row)

    # ── Output checks ─────────────────────────────────────────────────────────
    n_rows = len(rows)
    expected_rows = len(b5_pool) if args.selected_only else sum(
        r.get("n_candidates", 0) for r in b5_pool
    )

    checks["Q10_csv_row_count"] = f"PASS ({n_rows} rows)" if n_rows == expected_rows else f"WARN: {n_rows} vs expected {expected_rows}"
    checks["Q11_json_row_count"] = checks["Q10_csv_row_count"]

    bucket_none = sum(1 for r in rows if not r.get("review_bucket"))
    checks["Q12_review_bucket_assigned"] = "PASS" if bucket_none == 0 else f"FAIL: {bucket_none} rows missing bucket"

    molwt_zero = sum(1 for r in rows if r.get("MolWt") is None or r.get("MolWt", 0) <= 0)
    checks["Q13_MolWt_positive"] = "PASS" if molwt_zero == 0 else f"WARN: {molwt_zero} rows with MolWt<=0 or null"

    rmsd_bucket_none = sum(1 for r in rows if not r.get("rmsd_bucket") or r.get("rmsd_bucket") == "")
    checks["Q14_rmsd_bucket_nonempty"] = "PASS" if rmsd_bucket_none == 0 else f"FAIL: {rmsd_bucket_none} rows missing rmsd_bucket"

    has_shift = any("property_shift" in r or "shape_similarity" in r for r in rows)
    checks["Q15_no_property_shift_fields"] = "PASS" if not has_shift else "FAIL: forbidden fields present"

    checks["Q16_m2r_h1_data_unchanged"] = "PASS (read-only access to input files)"
    checks["Q17_no_f2r_rdkit_rerun"] = "PASS (no generation pipeline invoked)"

    # ── Write CSV ─────────────────────────────────────────────────────────────
    is_smoke = args.n_cases is not None
    if is_smoke:
        csv_path = out_dir / "a4c_candidate_review_table_smoke.csv"
        json_path = out_dir / "a4c_candidate_review_table_smoke.json"
        summary_path = out_dir / "a4c_summary_smoke.json"
        verdict_path = out_dir / "A4C_V0_SMOKE_VERDICT.md"
    else:
        csv_path = out_dir / "a4c_candidate_review_table.csv"
        json_path = out_dir / "a4c_candidate_review_table.json"
        summary_path = out_dir / "a4c_summary.json"
        verdict_path = out_dir / "A4C_V0_FULL_VERDICT.md"

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # ── Bucket distribution ───────────────────────────────────────────────────
    from collections import Counter
    bucket_dist = Counter(r["review_bucket"] for r in rows)
    rmsd_bucket_dist = Counter(r["rmsd_bucket"] for r in rows)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    warnings = []
    if not PAINS_OK:
        warnings.append("PAINS FilterCatalog unavailable")
    if not BRENK_OK:
        warnings.append("Brenk FilterCatalog unavailable")
    if prop_fail_count > 0:
        warnings.append(f"{prop_fail_count} cases had MolWt=None (mol deserialization failed)")

    # Determine final status
    fail_checks = [k for k, v in checks.items() if str(v).startswith("FAIL")]
    warn_checks = [k for k, v in checks.items() if str(v).startswith("WARN")]

    if fail_checks:
        final_status = "FAIL_OUTPUT_SCHEMA" if "Q12" in str(fail_checks) or "Q14" in str(fail_checks) else "FAIL_EXCEPTION"
    elif warn_checks or warnings:
        final_status = "PASS_WITH_WARNINGS"
    else:
        final_status = "PASS"

    summary = {
        "task": "A4C_V0_SMOKE_TEST",
        "n_cases": len(b5_pool),
        "n_rows": n_rows,
        "selected_only": args.selected_only,
        "rdkit_ok": RDKIT_OK,
        "pains_ok": PAINS_OK,
        "brenk_ok": BRENK_OK,
        "bucket_distribution": dict(bucket_dist),
        "rmsd_bucket_distribution": dict(rmsd_bucket_dist),
        "checks": checks,
        "warnings": warnings,
        "final_status": final_status,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Verdict MD ────────────────────────────────────────────────────────────
    verdict_lines = [
        "# A4C v0 Smoke Test Verdict",
        "",
        "## Executive Summary",
        "",
        f"A4C v0 smoke test ran on {len(b5_pool)} cases (selected_only={args.selected_only}).",
        f"Output: {n_rows} rows. Final status: **{final_status}**",
        "",
        "## Input Checks (Q1-Q17)",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for k, v in checks.items():
        verdict_lines.append(f"| {k} | {v} |")

    verdict_lines += [
        "",
        "## Output Row Count",
        "",
        f"- Expected: {expected_rows}",
        f"- Actual: {n_rows}",
        "",
        "## Bucket Distribution",
        "",
        "| Bucket | Count |",
        "|---|---|",
    ]
    for bucket, count in sorted(bucket_dist.items()):
        verdict_lines.append(f"| {bucket} | {count} |")

    verdict_lines += [
        "",
        "## RMSD Bucket Distribution",
        "",
        "| RMSD Bucket | Count |",
        "|---|---|",
    ]
    for rb, count in sorted(rmsd_bucket_dist.items()):
        verdict_lines.append(f"| {rb} | {count} |")

    verdict_lines += [
        "",
        "## Descriptor Sanity Check",
        "",
        f"- MolWt null/zero count: {molwt_zero}",
        f"- Prop fail count: {prop_fail_count}",
        f"- RDKit available: {RDKIT_OK}",
        "",
        "## FilterCatalog Status",
        "",
        f"- PAINS_A: {'AVAILABLE' if PAINS_OK else 'UNAVAILABLE'}",
        f"- Brenk: {'AVAILABLE' if BRENK_OK else 'UNAVAILABLE'}",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        for w in warnings:
            verdict_lines.append(f"- {w}")
    else:
        verdict_lines.append("- None")

    verdict_lines += [
        "",
        "## Final Status",
        "",
        f"**A4C_V0_SMOKE_STATUS = {final_status}**",
        "",
        "## Scope Confirmation",
        "",
        "- property_shift fields: NOT present (deferred to A4C v1)",
        "- shape_similarity fields: NOT present (deferred to A4C v1)",
        "- M2R/H1 data: NOT modified (read-only)",
        "- F2R/RDKit pipeline: NOT re-run",
    ]

    with open(verdict_path, "w", encoding="utf-8") as f:
        f.write("\n".join(verdict_lines) + "\n")

    # ── CLI summary ───────────────────────────────────────────────────────────
    print(f"\nA4C v0 Smoke Test")
    print(f"  cases:   {len(b5_pool)}")
    print(f"  rows:    {n_rows}")
    print(f"  PAINS:   {'OK' if PAINS_OK else 'UNAVAILABLE'}")
    print(f"  Brenk:   {'OK' if BRENK_OK else 'UNAVAILABLE'}")
    print(f"\nBucket distribution:")
    for bucket, count in sorted(bucket_dist.items()):
        print(f"  {bucket}: {count}")
    print(f"\nRMSD bucket distribution:")
    for rb, count in sorted(rmsd_bucket_dist.items()):
        print(f"  {rb}: {count}")
    if warnings:
        print(f"\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    print(f"\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {summary_path}")
    print(f"  {verdict_path}")
    print(f"\nA4C_V0_SMOKE_STATUS = {final_status}")

    sys.exit(0 if final_status in ("PASS", "PASS_WITH_WARNINGS") else 1)


if __name__ == "__main__":
    main()
