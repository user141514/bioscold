#!/usr/bin/env python3
"""Route-A S7: target-aware retrospective activity validation pilot.

This script joins Route-A test-set weak-positive MMP pairs to public ChEMBL
activity records through molecule ChEMBL IDs. It reports only same-assay,
same-target, same-standard-type pChEMBL comparisons as strong retrospective
activity evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from routeA_d4s2_common import load_baseline_artifacts, score_matrix_to_ranks  # noqa: E402
import routeA_topjournal_s3_curated_bioisostere_recall as s3  # noqa: E402

PROJECT = Path("E:/zuhui/bioisosteric_diffusion")
PLAN = PROJECT / "plan_results"
D0 = PLAN / "routeA_chembl37k_d0d3_engineering_safe"
PAIR_PATH = D0 / "05_d2_labeling_repaired" / "d2r_pair_benchmark_manifest_ratio1_split.jsonl"
CHUNK_DIR = D0 / "01_d0_data_audit" / "standardized_chunks"
OUT = PLAN / "routeA_topjournal_s7_target_aware_activity_validation"
OUT.mkdir(parents=True, exist_ok=True)

MAX_TEST_POSITIVE_PAIRS = 0
BATCH_SIZE = 100
REQUEST_SLEEP_SEC = 0.15
ACTIVITY_TYPES = {"IC50", "EC50", "AC50", "Ki", "Kd", "Potency"}
CHEMBL_ACTIVITY_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
SEED = 20260528
BOOTSTRAP_N = 5000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_test_positive_pairs(limit: int) -> pd.DataFrame:
    rows = []
    for row in read_jsonl(PAIR_PATH):
        if row.get("split") != "test" or row.get("label") != "WEAK_POSITIVE":
            continue
        rows.append(
            {
                "pair_id": row["pair_id"],
                "old_mol_id": row["old_mol_id"],
                "replacement_mol_id": row["replacement_mol_id"],
                "old_fragment_smiles": row["old_fragment_smiles"],
                "replacement_fragment_smiles": row["replacement_fragment_smiles"],
                "attachment_signature": row["attachment_signature"],
                "core_key": row["core_key"],
                "core_smiles": row["core_smiles"],
                "transform_key": row["transform_key"],
                "same_core_group_size": row.get("same_core_group_size"),
            }
        )
        if limit and len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def load_molecule_map(mol_ids: set[str]) -> pd.DataFrame:
    found = []
    remaining = set(mol_ids)
    for path in sorted(CHUNK_DIR.glob("standardized_chunk_*.jsonl")):
        if not remaining:
            break
        for row in read_jsonl(path):
            mol_id = row.get("mol_id")
            if mol_id not in remaining:
                continue
            found.append(
                {
                    "mol_id": mol_id,
                    "chembl_id": row.get("chembl_id"),
                    "canonical_smiles": row.get("canonical_smiles"),
                    "standardized_smiles": row.get("standardized_smiles"),
                }
            )
            remaining.remove(mol_id)
    return pd.DataFrame(found)


def valid_activity(row: dict) -> bool:
    if row.get("pchembl_value") in (None, ""):
        return False
    if row.get("standard_relation") != "=":
        return False
    if str(row.get("standard_type", "")) not in ACTIVITY_TYPES:
        return False
    if not row.get("assay_chembl_id") or not row.get("target_chembl_id"):
        return False
    try:
        float(row["pchembl_value"])
    except Exception:
        return False
    return True


def fetch_activity_batch(chembl_ids: list[str], session: requests.Session) -> list[dict]:
    activities = []
    offset = 0
    while True:
        params = {
            "molecule_chembl_id__in": ",".join(chembl_ids),
            "limit": 1000,
            "offset": offset,
            "only": ",".join(
                [
                    "activity_id",
                    "molecule_chembl_id",
                    "target_chembl_id",
                    "assay_chembl_id",
                    "standard_type",
                    "standard_relation",
                    "standard_units",
                    "standard_value",
                    "pchembl_value",
                    "document_chembl_id",
                ]
            ),
        }
        response = session.get(CHEMBL_ACTIVITY_URL, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        batch_rows = payload.get("activities", [])
        activities.extend(batch_rows)
        page_meta = payload.get("page_meta", {})
        total = int(page_meta.get("total_count") or len(activities))
        offset += int(page_meta.get("limit") or 1000)
        if offset >= total or not batch_rows:
            break
        time.sleep(REQUEST_SLEEP_SEC)
    return activities


def fetch_or_load_activities(chembl_ids: list[str]) -> pd.DataFrame:
    cache_path = OUT / "s7_chembl_activity_cache.jsonl"
    fetched_path = OUT / "s7_chembl_activity_fetched_molecules.txt"
    cached = []
    cached_ids = set()
    fetched_ids = set()
    if fetched_path.exists():
        fetched_ids = {line.strip() for line in fetched_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if cache_path.exists():
        for row in read_jsonl(cache_path):
            if valid_activity(row):
                cached.append(row)
                cached_ids.add(str(row.get("molecule_chembl_id")))
    need_ids = [cid for cid in chembl_ids if cid and cid not in fetched_ids and cid not in cached_ids]
    log(
        f"ChEMBL activity cache has {len(cached_ids)} molecules with valid records, "
        f"{len(fetched_ids)} fetched molecules; fetching {len(need_ids)} new molecules."
    )
    session = requests.Session()
    with cache_path.open("a", encoding="utf-8") as handle:
        fetched_handle = fetched_path.open("a", encoding="utf-8")
        for start in range(0, len(need_ids), BATCH_SIZE):
            batch = need_ids[start : start + BATCH_SIZE]
            rows = fetch_activity_batch(batch, session)
            for chembl_id in batch:
                fetched_handle.write(f"{chembl_id}\n")
            for row in rows:
                if valid_activity(row):
                    slim = {
                        "activity_id": row.get("activity_id"),
                        "molecule_chembl_id": row.get("molecule_chembl_id"),
                        "target_chembl_id": row.get("target_chembl_id"),
                        "assay_chembl_id": row.get("assay_chembl_id"),
                        "standard_type": row.get("standard_type"),
                        "standard_relation": row.get("standard_relation"),
                        "standard_units": row.get("standard_units"),
                        "standard_value": row.get("standard_value"),
                        "pchembl_value": float(row.get("pchembl_value")),
                        "document_chembl_id": row.get("document_chembl_id"),
                    }
                    handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
                    cached.append(slim)
            log(f"Fetched batch {start // BATCH_SIZE + 1}/{max(1, (len(need_ids) + BATCH_SIZE - 1) // BATCH_SIZE)}")
            time.sleep(REQUEST_SLEEP_SEC)
        fetched_handle.close()
    if not cached:
        return pd.DataFrame()
    return pd.DataFrame(cached).drop_duplicates()


def aggregate_activities(activity_df: pd.DataFrame) -> pd.DataFrame:
    if activity_df.empty:
        return activity_df
    key_cols = ["molecule_chembl_id", "target_chembl_id", "assay_chembl_id", "standard_type"]
    grouped = (
        activity_df.groupby(key_cols, as_index=False)
        .agg(
            pchembl_median=("pchembl_value", "median"),
            pchembl_min=("pchembl_value", "min"),
            pchembl_max=("pchembl_value", "max"),
            n_activity_records=("activity_id", "nunique"),
            document_chembl_ids=("document_chembl_id", lambda x: ";".join(sorted(set(str(v) for v in x if pd.notna(v))))),
        )
    )
    return grouped


def build_comparable_pairs(pairs: pd.DataFrame, mol_map: pd.DataFrame, activities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mol_to_chembl = dict(zip(mol_map["mol_id"], mol_map["chembl_id"]))
    pairs = pairs.copy()
    pairs["old_chembl_id"] = pairs["old_mol_id"].map(mol_to_chembl)
    pairs["replacement_chembl_id"] = pairs["replacement_mol_id"].map(mol_to_chembl)
    agg = aggregate_activities(activities)
    old = agg.rename(columns={"molecule_chembl_id": "old_chembl_id", "pchembl_median": "old_pchembl"})
    repl = agg.rename(columns={"molecule_chembl_id": "replacement_chembl_id", "pchembl_median": "replacement_pchembl"})
    comparable_rows = []
    for pair in pairs.itertuples(index=False):
        old_sub = old.loc[old["old_chembl_id"].eq(pair.old_chembl_id)]
        repl_sub = repl.loc[repl["replacement_chembl_id"].eq(pair.replacement_chembl_id)]
        if old_sub.empty or repl_sub.empty:
            continue
        merged = old_sub.merge(
            repl_sub,
            on=["target_chembl_id", "assay_chembl_id", "standard_type"],
            suffixes=("_old", "_replacement"),
        )
        for row in merged.itertuples(index=False):
            comparable_rows.append(
                {
                    "pair_id": pair.pair_id,
                    "old_mol_id": pair.old_mol_id,
                    "replacement_mol_id": pair.replacement_mol_id,
                    "old_chembl_id": pair.old_chembl_id,
                    "replacement_chembl_id": pair.replacement_chembl_id,
                    "old_fragment_smiles": pair.old_fragment_smiles,
                    "replacement_fragment_smiles": pair.replacement_fragment_smiles,
                    "attachment_signature": pair.attachment_signature,
                    "core_key": pair.core_key,
                    "target_chembl_id": row.target_chembl_id,
                    "assay_chembl_id": row.assay_chembl_id,
                    "standard_type": row.standard_type,
                    "old_pchembl": float(row.old_pchembl),
                    "replacement_pchembl": float(row.replacement_pchembl),
                    "delta_pchembl_replacement_minus_old": float(row.replacement_pchembl - row.old_pchembl),
                    "old_n_activity_records": int(row.n_activity_records_old),
                    "replacement_n_activity_records": int(row.n_activity_records_replacement),
                    "old_document_chembl_ids": row.document_chembl_ids_old,
                    "replacement_document_chembl_ids": row.document_chembl_ids_replacement,
                }
            )
    comparable = pd.DataFrame(comparable_rows)
    if comparable.empty:
        return pairs, comparable
    pair_summary = (
        comparable.groupby("pair_id", as_index=False)
        .agg(
            n_same_assay_comparisons=("pair_id", "size"),
            n_targets=("target_chembl_id", "nunique"),
            n_assays=("assay_chembl_id", "nunique"),
            median_delta_pchembl=("delta_pchembl_replacement_minus_old", "median"),
            min_delta_pchembl=("delta_pchembl_replacement_minus_old", "min"),
            max_delta_pchembl=("delta_pchembl_replacement_minus_old", "max"),
            median_old_pchembl=("old_pchembl", "median"),
            median_replacement_pchembl=("replacement_pchembl", "median"),
        )
    )
    return pairs, comparable.merge(pair_summary, on="pair_id", how="left")


def add_routea_ranks(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pair_df
    artifacts = load_baseline_artifacts()
    vocab_index = {smi: i for i, smi in enumerate(artifacts.vocab)}
    rows = pair_df.drop_duplicates("pair_id").copy()
    rows = rows.loc[rows["replacement_fragment_smiles"].isin(vocab_index)].reset_index(drop=True)
    if rows.empty:
        return pair_df.assign(replacement_in_routeA_vocab=0)
    labels = np.zeros((len(rows), len(artifacts.vocab)), dtype=np.uint8)
    scores = s3.compute_routea_scores(rows[["old_fragment_smiles", "attachment_signature"]], labels)
    rank_cols = {}
    for method in ["HGB", "Borda_DE_HGB", "M5_mlp_F0", "S1_MLP_HGB_score_blend"]:
        ranks = score_matrix_to_ranks(scores[method])
        rank_cols[f"{method}_rank"] = [
            int(ranks[i, vocab_index[frag]]) for i, frag in enumerate(rows["replacement_fragment_smiles"])
        ]
    rank_df = rows[["pair_id"]].copy()
    for col, values in rank_cols.items():
        rank_df[col] = values
        rank_df[col.replace("_rank", "_hit10")] = (rank_df[col] <= 10).astype(int)
    rank_df["replacement_in_routeA_vocab"] = 1
    return pair_df.merge(rank_df, on="pair_id", how="left")


def summarize(enriched: pd.DataFrame, pairs: pd.DataFrame, mol_map: pd.DataFrame, activities: pd.DataFrame) -> dict:
    unique_pairs = enriched.drop_duplicates("pair_id") if not enriched.empty else pd.DataFrame()
    retained_05 = unique_pairs["median_delta_pchembl"].ge(-0.5) if not unique_pairs.empty else pd.Series(dtype=bool)
    retained_10 = unique_pairs["median_delta_pchembl"].ge(-1.0) if not unique_pairs.empty else pd.Series(dtype=bool)
    improved_05 = unique_pairs["median_delta_pchembl"].ge(0.5) if not unique_pairs.empty else pd.Series(dtype=bool)
    degraded_10 = unique_pairs["median_delta_pchembl"].le(-1.0) if not unique_pairs.empty else pd.Series(dtype=bool)
    summary = {
        "source_pair_file": str(PAIR_PATH),
        "sampled_test_positive_pairs": int(len(pairs)),
        "unique_sampled_molecules": int(len(set(pairs["old_mol_id"]) | set(pairs["replacement_mol_id"]))),
        "molecules_with_chembl_id": int(mol_map["chembl_id"].notna().sum()) if not mol_map.empty else 0,
        "molecules_with_valid_pchembl_activity": int(activities["molecule_chembl_id"].nunique()) if not activities.empty else 0,
        "valid_pchembl_activity_records": int(len(activities)),
        "same_assay_comparable_pairs": int(unique_pairs["pair_id"].nunique()) if not unique_pairs.empty else 0,
        "same_assay_comparable_records": int(len(enriched)),
        "activity_retained_delta_ge_minus_0p5_pairs": int(retained_05.sum()) if not unique_pairs.empty else 0,
        "activity_retained_delta_ge_minus_1p0_pairs": int(retained_10.sum()) if not unique_pairs.empty else 0,
        "activity_improved_delta_ge_0p5_pairs": int(improved_05.sum()) if not unique_pairs.empty else 0,
        "activity_degraded_delta_le_minus_1p0_pairs": int(degraded_10.sum()) if not unique_pairs.empty else 0,
    }
    for method in ["HGB", "Borda_DE_HGB", "M5_mlp_F0", "S1_MLP_HGB_score_blend"]:
        hit_col = f"{method}_hit10"
        rank_col = f"{method}_rank"
        if hit_col not in unique_pairs:
            continue
        supported = unique_pairs.loc[unique_pairs[hit_col].notna()]
        retained = unique_pairs.loc[retained_05 & unique_pairs[hit_col].notna()]
        summary[f"{method}_same_assay_pair_Top10"] = float(supported[hit_col].mean()) if len(supported) else None
        summary[f"{method}_same_assay_pair_MRR"] = float((1.0 / supported[rank_col]).mean()) if len(supported) else None
        summary[f"{method}_retained0p5_pair_Top10"] = float(retained[hit_col].mean()) if len(retained) else None
        summary[f"{method}_retained0p5_pair_MRR"] = float((1.0 / retained[rank_col]).mean()) if len(retained) else None
    return summary


def unique_pair_table(enriched: pd.DataFrame) -> pd.DataFrame:
    if enriched.empty:
        return pd.DataFrame()
    sort_cols = ["pair_id", "n_same_assay_comparisons", "median_delta_pchembl"]
    ascending = [True, False, False]
    return (
        enriched.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates("pair_id")
        .reset_index(drop=True)
    )


def activity_band(delta: float) -> str:
    if delta >= 0.5:
        return "improved_delta_ge_0p5"
    if delta >= -0.5:
        return "retained_delta_ge_minus_0p5"
    if delta <= -1.0:
        return "degraded_delta_le_minus_1p0"
    return "ambiguous_delta_minus1p0_to_minus0p5"


def write_activity_rank_diagnostics(enriched: pd.DataFrame) -> None:
    unique_pairs = unique_pair_table(enriched)
    if unique_pairs.empty:
        return
    unique_pairs["activity_band"] = unique_pairs["median_delta_pchembl"].map(activity_band)
    unique_pairs.to_csv(OUT / "s7_unique_pair_activity_rank_table.csv", index=False)

    methods = ["HGB", "Borda_DE_HGB", "M5_mlp_F0", "S1_MLP_HGB_score_blend"]
    strata_defs = {
        "all_same_assay": unique_pairs.index.to_numpy(),
        "retained_delta_ge_minus_0p5": unique_pairs.index[unique_pairs["median_delta_pchembl"].ge(-0.5)].to_numpy(),
        "retained_delta_ge_minus_1p0": unique_pairs.index[unique_pairs["median_delta_pchembl"].ge(-1.0)].to_numpy(),
        "improved_delta_ge_0p5": unique_pairs.index[unique_pairs["median_delta_pchembl"].ge(0.5)].to_numpy(),
        "degraded_delta_le_minus_1p0": unique_pairs.index[unique_pairs["median_delta_pchembl"].le(-1.0)].to_numpy(),
    }
    rows = []
    for stratum, idx in strata_defs.items():
        sub = unique_pairs.loc[idx]
        for method in methods:
            rank_col = f"{method}_rank"
            hit_col = f"{method}_hit10"
            if rank_col not in sub or sub.empty:
                continue
            method_sub = sub.loc[sub[rank_col].notna() & sub[hit_col].notna()]
            if method_sub.empty:
                continue
            rows.append(
                {
                    "stratum": stratum,
                    "method": method,
                    "n_pairs": int(len(method_sub)),
                    "median_delta_pchembl": float(method_sub["median_delta_pchembl"].median()),
                    "Top10": float(method_sub[hit_col].mean()),
                    "MRR": float((1.0 / method_sub[rank_col].astype(float)).mean()),
                    "median_rank": float(method_sub[rank_col].median()),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "s7_activity_stratified_rank_metrics.csv", index=False)

    rng = np.random.default_rng(SEED)
    boot_rows = []
    comparisons = [("S1_MLP_HGB_score_blend", "Borda_DE_HGB"), ("S1_MLP_HGB_score_blend", "HGB")]
    for stratum, idx in strata_defs.items():
        sub = unique_pairs.loc[idx].reset_index(drop=True)
        if len(sub) < 2:
            continue
        for method in methods:
            method_sub = sub.loc[sub[f"{method}_rank"].notna() & sub[f"{method}_hit10"].notna()].reset_index(drop=True)
            if len(method_sub) < 2:
                continue
            hit = method_sub[f"{method}_hit10"].astype(float).to_numpy()
            mrr = (1.0 / method_sub[f"{method}_rank"].astype(float)).to_numpy()
            top10_boot = np.empty(BOOTSTRAP_N, dtype=float)
            mrr_boot = np.empty(BOOTSTRAP_N, dtype=float)
            for i in range(BOOTSTRAP_N):
                sample_idx = rng.integers(0, len(method_sub), size=len(method_sub))
                top10_boot[i] = float(hit[sample_idx].mean())
                mrr_boot[i] = float(mrr[sample_idx].mean())
            boot_rows.append(
                {
                    "stratum": stratum,
                    "comparison": f"{method}_absolute",
                    "n_pairs": int(len(method_sub)),
                    "Top10": float(hit.mean()),
                    "Top10_ci_low": float(np.quantile(top10_boot, 0.025)),
                    "Top10_ci_high": float(np.quantile(top10_boot, 0.975)),
                    "MRR": float(mrr.mean()),
                    "MRR_ci_low": float(np.quantile(mrr_boot, 0.025)),
                    "MRR_ci_high": float(np.quantile(mrr_boot, 0.975)),
                }
            )
        for left, right in comparisons:
            cmp_sub = sub.loc[
                sub[f"{left}_rank"].notna()
                & sub[f"{left}_hit10"].notna()
                & sub[f"{right}_rank"].notna()
                & sub[f"{right}_hit10"].notna()
            ].reset_index(drop=True)
            if len(cmp_sub) < 2:
                continue
            left_hit = cmp_sub[f"{left}_hit10"].astype(float).to_numpy()
            right_hit = cmp_sub[f"{right}_hit10"].astype(float).to_numpy()
            left_mrr = (1.0 / cmp_sub[f"{left}_rank"].astype(float)).to_numpy()
            right_mrr = (1.0 / cmp_sub[f"{right}_rank"].astype(float)).to_numpy()
            top10_delta = left_hit - right_hit
            mrr_delta = left_mrr - right_mrr
            top10_boot = np.empty(BOOTSTRAP_N, dtype=float)
            mrr_boot = np.empty(BOOTSTRAP_N, dtype=float)
            for i in range(BOOTSTRAP_N):
                sample_idx = rng.integers(0, len(cmp_sub), size=len(cmp_sub))
                top10_boot[i] = float(top10_delta[sample_idx].mean())
                mrr_boot[i] = float(mrr_delta[sample_idx].mean())
            boot_rows.append(
                {
                    "stratum": stratum,
                    "comparison": f"{left}_minus_{right}",
                    "n_pairs": int(len(cmp_sub)),
                    "Top10": float(top10_delta.mean()),
                    "Top10_ci_low": float(np.quantile(top10_boot, 0.025)),
                    "Top10_ci_high": float(np.quantile(top10_boot, 0.975)),
                    "MRR": float(mrr_delta.mean()),
                    "MRR_ci_low": float(np.quantile(mrr_boot, 0.025)),
                    "MRR_ci_high": float(np.quantile(mrr_boot, 0.975)),
                }
            )
    pd.DataFrame(boot_rows).to_csv(OUT / "s7_activity_rank_bootstrap.csv", index=False)

    distribution = []
    for col in ["standard_type", "target_chembl_id", "assay_chembl_id"]:
        counts = enriched[col].value_counts().head(50)
        for key, count in counts.items():
            distribution.append({"field": col, "value": key, "n_records": int(count)})
    pd.DataFrame(distribution).to_csv(OUT / "s7_activity_distribution_top50.csv", index=False)

    examples = unique_pairs.sort_values(
        ["median_delta_pchembl", "S1_MLP_HGB_score_blend_hit10", "S1_MLP_HGB_score_blend_rank"],
        ascending=[False, False, True],
    ).head(100)
    examples.to_csv(OUT / "s7_activity_supported_case_examples.csv", index=False)


def write_verdict(summary: dict) -> None:
    status = "C. TARGET_AWARE_ACTIVITY_PILOT_FEASIBLE_BUT_NOT_MAIN_CLAIM"
    if summary["same_assay_comparable_pairs"] >= 100:
        status = "B. TARGET_AWARE_ACTIVITY_RETROSPECTIVE_READY_AS_LIMITED_SUPPORTING_EVIDENCE"
    if summary["same_assay_comparable_pairs"] == 0:
        status = "D. TARGET_AWARE_ACTIVITY_VALIDATION_BLOCKED_BY_NO_COMPARABLE_ASSAYS"
    s1_top10 = summary.get("S1_MLP_HGB_score_blend_retained0p5_pair_Top10")
    s1_mrr = summary.get("S1_MLP_HGB_score_blend_retained0p5_pair_MRR")
    borda_top10 = summary.get("Borda_DE_HGB_retained0p5_pair_Top10")
    s1_minus_borda_top10 = None
    s1_minus_borda_top10_ci = None
    boot_path = OUT / "s7_activity_rank_bootstrap.csv"
    s1_top10_ci = None
    s1_mrr_ci = None
    if boot_path.exists():
        boot = pd.read_csv(boot_path)
        sub = boot.loc[
            (boot["stratum"] == "retained_delta_ge_minus_0p5")
            & (boot["comparison"] == "S1_MLP_HGB_score_blend_absolute")
        ]
        if not sub.empty:
            s1_top10_ci = (float(sub["Top10_ci_low"].iloc[0]), float(sub["Top10_ci_high"].iloc[0]))
            s1_mrr_ci = (float(sub["MRR_ci_low"].iloc[0]), float(sub["MRR_ci_high"].iloc[0]))
        sub = boot.loc[
            (boot["stratum"] == "retained_delta_ge_minus_0p5")
            & (boot["comparison"] == "S1_MLP_HGB_score_blend_minus_Borda_DE_HGB")
        ]
        if not sub.empty:
            s1_minus_borda_top10 = float(sub["Top10"].iloc[0])
            s1_minus_borda_top10_ci = (float(sub["Top10_ci_low"].iloc[0]), float(sub["Top10_ci_high"].iloc[0]))
    lines = [
        "# S7 Target-Aware Activity Validation Verdict",
        "",
        f"**Verdict:** {status}",
        "",
        "## Protocol",
        "",
        "External ChEMBL activities were joined by molecule ChEMBL ID. Valid activities require pChEMBL values, standard_relation = '=', a supported standard_type, and target/assay identifiers. Strong retrospective comparisons require the old and replacement molecules to share the same target_chembl_id, assay_chembl_id, and standard_type.",
        "",
        "## Coverage",
        "",
        f"- Sampled Route-A test weak-positive pairs: {summary['sampled_test_positive_pairs']}",
        f"- Valid pChEMBL activity records: {summary['valid_pchembl_activity_records']}",
        f"- Molecules with valid pChEMBL activity: {summary['molecules_with_valid_pchembl_activity']}",
        f"- Same-assay comparable pairs: {summary['same_assay_comparable_pairs']}",
        f"- Same-assay comparable records: {summary['same_assay_comparable_records']}",
        f"- Activity-retained pairs (median delta pChEMBL >= -0.5): {summary['activity_retained_delta_ge_minus_0p5_pairs']}",
        f"- Activity-retained pairs (median delta pChEMBL >= -1.0): {summary['activity_retained_delta_ge_minus_1p0_pairs']}",
        "",
        "## Route-A Recall on Activity-Retained Pairs",
        "",
        f"- S1 MLP+HGB score blend retained-pair Top10: {s1_top10 if s1_top10 is not None else 'NA'}"
        + (f" (95% CI [{s1_top10_ci[0]:.4f}, {s1_top10_ci[1]:.4f}])" if s1_top10_ci else ""),
        f"- S1 MLP+HGB score blend retained-pair MRR: {s1_mrr if s1_mrr is not None else 'NA'}"
        + (f" (95% CI [{s1_mrr_ci[0]:.4f}, {s1_mrr_ci[1]:.4f}])" if s1_mrr_ci else ""),
        f"- Borda(DE,HGB) retained-pair Top10: {borda_top10 if borda_top10 is not None else 'NA'}",
        f"- S1 minus Borda retained-pair Top10 delta: {s1_minus_borda_top10 if s1_minus_borda_top10 is not None else 'NA'}"
        + (
            f" (95% CI [{s1_minus_borda_top10_ci[0]:.4f}, {s1_minus_borda_top10_ci[1]:.4f}])"
            if s1_minus_borda_top10_ci
            else ""
        ),
        "",
        "## Claim Boundary",
        "",
        "This is target-aware retrospective supporting evidence for activity-retained replacement recall and vocabulary safety. It is not prospective validation, not an activity model, and not evidence that S1 is superior to Borda on the activity-retained subset.",
    ]
    (OUT / "S7_TARGET_AWARE_ACTIVITY_VALIDATION_VERDICT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route-A S7 target-aware activity validation")
    parser.add_argument(
        "--max-test-positive-pairs",
        type=int,
        default=MAX_TEST_POSITIVE_PAIRS,
        help="Maximum test weak-positive pairs to process; 0 means all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log("Loading Route-A test weak-positive pairs.")
    pairs = load_test_positive_pairs(args.max_test_positive_pairs)
    pairs.to_csv(OUT / "s7_sampled_test_positive_pairs.csv", index=False)
    mol_ids = set(pairs["old_mol_id"]) | set(pairs["replacement_mol_id"])
    log(f"Loading molecule map for {len(mol_ids)} molecules.")
    mol_map = load_molecule_map(mol_ids)
    mol_map.to_csv(OUT / "s7_sampled_molecule_map.csv", index=False)
    chembl_ids = sorted(set(mol_map["chembl_id"].dropna().astype(str)))
    log(f"Fetching ChEMBL activities for {len(chembl_ids)} molecules.")
    activities = fetch_or_load_activities(chembl_ids)
    activities.to_csv(OUT / "s7_valid_chembl_pchembl_activities.csv", index=False)
    log("Building same-assay comparable activity pairs.")
    pairs_with_chembl, comparable = build_comparable_pairs(pairs, mol_map, activities)
    pairs_with_chembl.to_csv(OUT / "s7_sampled_pairs_with_chembl_ids.csv", index=False)
    comparable.to_csv(OUT / "s7_same_assay_activity_comparisons.csv", index=False)
    log("Adding Route-A ranks for comparable pairs.")
    enriched = add_routea_ranks(comparable)
    enriched.to_csv(OUT / "s7_target_aware_pair_rank_validation.csv", index=False)
    write_activity_rank_diagnostics(enriched)
    summary = summarize(enriched, pairs, mol_map, activities)
    (OUT / "s7_target_aware_activity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_verdict(summary)
    log(f"Wrote S7 outputs to {OUT}")


if __name__ == "__main__":
    main()
