#!/usr/bin/env python3
"""Route-A D4P1-Phase4 non-cherry-picked case-study package."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

SEED = 20260525

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN = PROJECT_ROOT / "plan_results"
OUT = PLAN / "routeA_chembl37k_d4p1_phase3_4_interpretability_cases"

PHASE3_QUERY = OUT / "d4p1_phase3_query_analysis_table.csv"
D4A4_CANON = PLAN / "routeA_chembl37k_d4a4_dual_mode_integration" / "d4a4_canonical_candidate_table.csv"
D4A4_CONS = PLAN / "routeA_chembl37k_d4a4_dual_mode_integration" / "d4a4_conservative_mode_top10.csv"
D4A4_EXPL = PLAN / "routeA_chembl37k_d4a4_dual_mode_integration" / "d4a4_exploration_mode_top10.csv"
STANDARDIZED_TEST = PLAN / "routeA_chembl37k_d4a2d2_de_hgb_ensemble" / "d4a2d2_standardized_predictions_test.jsonl"


def log(msg: str) -> None:
    print(msg, flush=True)


def split_pipe(value: object) -> List[str]:
    if pd.isna(value) or value is None or str(value) == "":
        return []
    return [part for part in str(value).split("|") if part]


def pipe_join(values: List[str]) -> str:
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            out.append(str(value))
            seen.add(value)
    return "|".join(out)


class FeatureCalculator:
    def __init__(self) -> None:
        self.fp_cache = {}
        self.ha_cache = {}

    def _mol(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
        stripped = smiles.replace("*", "")
        if stripped != smiles:
            return Chem.MolFromSmiles(stripped)
        return None

    def fp(self, smiles: str):
        if smiles not in self.fp_cache:
            mol = self._mol(smiles)
            self.fp_cache[smiles] = None if mol is None else AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return self.fp_cache[smiles]

    def heavy_atoms(self, smiles: str) -> int:
        if smiles not in self.ha_cache:
            mol = self._mol(smiles)
            self.ha_cache[smiles] = 0 if mol is None else int(mol.GetNumHeavyAtoms())
        return self.ha_cache[smiles]

    def similarity(self, old_smiles: str, candidate_smiles: str) -> float:
        ofp = self.fp(old_smiles)
        cfp = self.fp(candidate_smiles)
        if ofp is None or cfp is None:
            return 0.0
        return float(DataStructs.TanimotoSimilarity(ofp, cfp))

    def heavy_atom_delta(self, old_smiles: str, candidate_smiles: str) -> float:
        return float(abs(self.heavy_atoms(candidate_smiles) - self.heavy_atoms(old_smiles)))


def load_jsonl(path: Path):
    import json

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_exact_to_norm() -> Dict[str, str]:
    mapping = {}
    for row in load_jsonl(STANDARDIZED_TEST):
        mapping.setdefault(str(row["candidate"]), str(row["candidate_norm"]))
    return mapping


def build_candidate_diag_map() -> Dict[Tuple[str, str], dict]:
    canon_df = pd.read_csv(D4A4_CANON)
    mapping = {}
    for _, row in canon_df.iterrows():
        mapping[(str(row["query_id"]), str(row["candidate_norm"]))] = {
            "group_origin": str(row.get("group_origin", "other")),
            "a4c_tier": str(row.get("a4c_review_bucket", "A4C_UNKNOWN")),
            "hard_alert_flag": int(bool(row.get("hard_alert_flag", False))),
            "review_ready_flag": int(bool(row.get("review_ready_flag", False))),
            "alert_reason_codes": str(row.get("alert_reason_codes", "")),
            "gap_type": str(row.get("gap_type", "")),
        }
    return mapping


def build_mode_top1(path: Path) -> Dict[str, dict]:
    df = pd.read_csv(path)
    df = df[df["topK_rank"] == 1].copy()
    out = {}
    for _, row in df.iterrows():
        out[str(row["query_id"])] = row.to_dict()
    return out


def stratified_sample(pool: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(pool) <= n:
        out = pool.copy()
        out["sample_status"] = "SAMPLE_UNDERFILLED"
        return out
    rng = random.Random(SEED)
    grouped = defaultdict(list)
    for idx, row in pool.iterrows():
        grouped[str(row["attachment_signature"])].append(idx)
    for indices in grouped.values():
        rng.shuffle(indices)
    order = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    chosen = []
    while len(chosen) < n and any(indices for _, indices in order):
        for _, indices in order:
            if indices and len(chosen) < n:
                chosen.append(indices.pop(0))
    out = pool.loc[chosen].copy()
    out["sample_status"] = "SAMPLED"
    return out


def first_positive_from_method(row: pd.Series, method_col: str, exact_to_norm: Dict[str, str], diag_map: Dict[Tuple[str, str], dict], wanted_group: Optional[str] = None) -> Tuple[str, str, dict]:
    positives = set(split_pipe(row["positive_replacements_exact"]))
    candidates = split_pipe(row[method_col])
    qid = str(row["query_id"])
    for cand in candidates:
        norm = exact_to_norm.get(cand, cand.replace("*", ""))
        if cand not in positives:
            continue
        diag = diag_map.get((qid, norm), {})
        if wanted_group is not None and diag.get("group_origin") != wanted_group:
            continue
        return cand, norm, diag
    return "", "", {}


def norm_to_exact_lookup(row: pd.Series) -> Dict[str, str]:
    exacts = split_pipe(row["positive_replacements_exact"])
    norms = split_pipe(row["positive_replacements_norm"])
    out = {}
    for exact, norm in zip(exacts, norms):
        out.setdefault(norm, exact)
    return out


def main() -> None:
    log("Loading Phase3 query table and D4A4 diagnostics...")
    query_df = pd.read_csv(PHASE3_QUERY)
    query_df["query_id"] = query_df["query_id"].astype(str)
    exact_to_norm = load_exact_to_norm()
    diag_map = build_candidate_diag_map()
    cons_top1 = build_mode_top1(D4A4_CONS)
    expl_top1 = build_mode_top1(D4A4_EXPL)
    feat = FeatureCalculator()

    category_pools = {
        "DE_only_rescued": query_df[(query_df["DE_hit10"] == 1) & (query_df["HGB_hit10"] == 0)],
        "HGB_only_rescued": query_df[(query_df["HGB_hit10"] == 1) & (query_df["DE_hit10"] == 0)],
        "Borda_G2_high_risk": query_df[(query_df["borda_hit10_has_g2_positive"] == 1) & ((query_df["borda_hit10_hard_reject"] == 1) | (query_df["borda_hit10_has_tier2_positive"] == 1) | (query_df["borda_hit10_has_tier3_positive"] == 1))],
        "Borda_G3_acceptable_exploration": query_df[(query_df["borda_hit10_has_g3_positive"] == 1) & (query_df["borda_hit10_reviewable"] == 1) & (query_df["borda_hit10_hard_reject"] == 0)],
        "G4_shared_low_risk": query_df[(query_df["borda_hit10_has_g4_positive"] == 1) & (query_df["HGB_hit10"] == 1) & (query_df["borda_hit10_hard_reject"] == 0)],
        "C|O_negative_case": query_df[(query_df["attachment_signature"] == "C|O") & (query_df["borda_net_loss_flag"] == 1)],
        "cluster_09_failure_case": query_df[(query_df["old_fragment_cluster_id"] == "cluster_09") & (query_df["borda_net_loss_flag"] == 1)],
    }
    top1_diff_rows = []
    for _, row in query_df.iterrows():
        qid = str(row["query_id"])
        if qid not in cons_top1 or qid not in expl_top1:
            continue
        cons_norm = str(cons_top1[qid]["candidate_norm"])
        expl_norm = str(expl_top1[qid]["candidate_norm"])
        positive_norms = set(split_pipe(row["positive_replacements_norm"]))
        if cons_norm != expl_norm and cons_norm in positive_norms and expl_norm in positive_norms:
            top1_diff_rows.append(row.name)
    category_pools["Conservative_and_Exploration_top1_differ_both_hit"] = query_df.loc[top1_diff_rows].copy()

    samples = []
    for category, pool in category_pools.items():
        sampled = stratified_sample(pool, 5)
        for _, row in sampled.iterrows():
            qid = str(row["query_id"])
            pos_norm_to_exact = norm_to_exact_lookup(row)
            cons = cons_top1.get(qid, {})
            expl = expl_top1.get(qid, {})
            cons_norm = str(cons.get("candidate_norm", ""))
            expl_norm = str(expl.get("candidate_norm", ""))
            cons_exact = pos_norm_to_exact.get(cons_norm, cons_norm)
            expl_exact = pos_norm_to_exact.get(expl_norm, expl_norm)

            focal_exact = ""
            focal_norm = ""
            focal_diag = {}
            why = ""
            if category == "DE_only_rescued":
                focal_exact, focal_norm, focal_diag = first_positive_from_method(row, "DE_top10_exact", exact_to_norm, diag_map)
                why = "DE top10 hits a positive while HGB top10 misses."
            elif category == "HGB_only_rescued":
                focal_exact, focal_norm, focal_diag = first_positive_from_method(row, "HGB_top10_exact", exact_to_norm, diag_map)
                why = "HGB top10 hits a positive while DE top10 misses."
            elif category == "Borda_G2_high_risk":
                focal_exact, focal_norm, focal_diag = first_positive_from_method(row, "Borda_top10_exact", exact_to_norm, diag_map, wanted_group="G2_pure_borda_only")
                why = "Borda recovers a G2 positive that stays in expert/high-risk review territory."
            elif category == "Borda_G3_acceptable_exploration":
                focal_exact, focal_norm, focal_diag = first_positive_from_method(row, "Borda_top10_exact", exact_to_norm, diag_map, wanted_group="G3_de_retained_by_borda")
                why = "Borda recovers a G3 positive that looks reviewable under the audited proxy."
            elif category == "G4_shared_low_risk":
                focal_exact, focal_norm, focal_diag = first_positive_from_method(row, "Borda_top10_exact", exact_to_norm, diag_map, wanted_group="G4_shared")
                why = "Shared low-risk positive where Conservative and Exploration are aligned."
            elif category == "Conservative_and_Exploration_top1_differ_both_hit":
                focal_exact = expl_exact
                focal_norm = expl_norm
                focal_diag = diag_map.get((qid, expl_norm), {})
                why = "Conservative and Exploration top1 differ, but both candidates are positives."
            elif category == "C|O_negative_case":
                focal_exact, focal_norm, focal_diag = first_positive_from_method(row, "HGB_top10_exact", exact_to_norm, diag_map)
                why = "Negative C|O subspace example where HGB is right and Borda loses ground."
            elif category == "cluster_09_failure_case":
                focal_exact, focal_norm, focal_diag = first_positive_from_method(row, "HGB_top10_exact", exact_to_norm, diag_map)
                why = "Cluster_09 failure mode where HGB succeeds and DE/Borda collapse."

            if not focal_exact:
                focal_exact = cons_exact or expl_exact
                focal_norm = cons_norm or expl_norm
                focal_diag = diag_map.get((qid, focal_norm), {})

            sim = feat.similarity(str(row["old_fragment_smiles"]), focal_exact or focal_norm) if focal_exact or focal_norm else float("nan")
            delta = feat.heavy_atom_delta(str(row["old_fragment_smiles"]), focal_exact or focal_norm) if focal_exact or focal_norm else float("nan")
            samples.append(
                {
                    "case_id": f"{category}_{qid}",
                    "query_id": qid,
                    "category": category,
                    "sample_status": str(row.get("sample_status", "SAMPLED")),
                    "old_fragment_smiles": row["old_fragment_smiles"],
                    "attachment_signature": row["attachment_signature"],
                    "positive_replacement_set": row["positive_replacements_exact"],
                    "DE_top_candidates": row["DE_top10_exact"],
                    "HGB_top_candidates": row["HGB_top10_exact"],
                    "Borda_top_candidates": row["Borda_top10_exact"],
                    "Conservative_top1": cons_exact,
                    "Exploration_top1": expl_exact,
                    "focal_candidate": focal_exact,
                    "A4C_tier_if_available": focal_diag.get("a4c_tier", ""),
                    "group_origin_if_available": focal_diag.get("group_origin", ""),
                    "alert_reason_if_available": focal_diag.get("alert_reason_codes", "") or str(expl.get("exploration_warning", "")) or str(cons.get("reason_codes", "")),
                    "Morgan_similarity_if_available": sim,
                    "property_delta_if_available": delta,
                    "short_interpretation": why,
                    "why_included": "Pre-registered category with fixed-seed stratified sampling by attachment signature.",
                }
            )

    case_df = pd.DataFrame(samples)
    case_df.to_csv(OUT / "d4p1_phase4_case_study_table.csv", index=False)

    panel_df = case_df[
        [
            "case_id",
            "category",
            "old_fragment_smiles",
            "attachment_signature",
            "positive_replacement_set",
            "Conservative_top1",
            "Exploration_top1",
            "focal_candidate",
            "A4C_tier_if_available",
            "group_origin_if_available",
            "Morgan_similarity_if_available",
            "property_delta_if_available",
            "short_interpretation",
        ]
    ].copy()
    panel_df.to_csv(OUT / "d4p1_phase4_fig_case_panel_data.csv", index=False)

    report_lines = [
        "# D4P1-Phase4 Case Study Report",
        "",
        "## Sampling Rule",
        f"- Fixed random seed = {SEED}",
        "- Stratified by attachment_signature when pool size exceeded 5.",
        "- No manual substitution when a pool was underfilled.",
        "",
        "## Category Coverage",
    ]
    for category, sub_df in case_df.groupby("category"):
        report_lines.append(f"- `{category}`: {len(sub_df)} cases, statuses = {pipe_join(sorted(sub_df['sample_status'].unique().tolist()))}")
    report_lines.extend(
        [
            "",
            "## Skeptical Review",
            "- These cases are sampled examples, not proof of mechanism on their own.",
            "- Some categories are intrinsically diagnostic because A4C coverage is incomplete outside audited regions.",
            "- Old-fragment clusters remain descriptive rather than fully chemical taxonomies.",
            "- Weak MMP positives still do not prove activity-preserving bioisosteres.",
        ]
    )
    (OUT / "D4P1_PHASE4_CASE_STUDY_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    log("Phase4 case-study package complete.")


if __name__ == "__main__":
    main()
