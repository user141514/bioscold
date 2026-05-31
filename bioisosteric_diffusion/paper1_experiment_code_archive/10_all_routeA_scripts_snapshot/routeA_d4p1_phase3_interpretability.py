#!/usr/bin/env python3
"""Route-A D4P1-Phase3 interpretability package under the locked canonical protocol."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

SEED = 20260525
TOPK = 10
TOPR = 50

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN = PROJECT_ROOT / "plan_results"
OUT = PLAN / "routeA_chembl37k_d4p1_phase3_4_interpretability_cases"
OUT.mkdir(parents=True, exist_ok=True)

PHASE0 = PLAN / "routeA_chembl37k_d4p1_phase0_metric_lock"
PHASE1 = PLAN / "routeA_chembl37k_d4p1_phase1_subset_robustness"
PHASE2 = PLAN / "routeA_chembl37k_d4p1_phase2_component_contribution"
D4A0 = PLAN / "routeA_chembl37k_d0d3_engineering_safe" / "07_d4a0_matrix_freeze"
D4A1 = PLAN / "routeA_chembl37k_d4a1_learned_ranker"
D4A1R = PLAN / "routeA_chembl37k_d4a1r_ranker_audit"
D4A2D1R = PLAN / "routeA_chembl37k_d4a2d1r_dual_encoder_robustness"
D4A2D2 = PLAN / "routeA_chembl37k_d4a2d2_de_hgb_ensemble"
D4A3R = PLAN / "routeA_chembl37k_d4a3r_a4c_borda_review"
D4A3S = PLAN / "routeA_chembl37k_d4a3s_a4c_coverage_expansion"
D4A3T = PLAN / "routeA_chembl37k_d4a3t_exploration_calibration"
D4A4 = PLAN / "routeA_chembl37k_d4a4_dual_mode_integration"

PHASE0_CANONICAL = PHASE0 / "d4p1_phase0_canonical_proposal_table.csv"
PHASE0_FIG = PHASE0 / "d4p1_phase0_fig_component_curve_data.csv"
PHASE0_VERDICT = PHASE0 / "D4P1_PHASE0_CANONICAL_METRIC_LOCK_VERDICT.md"

PHASE1_QUERY = PHASE1 / "d4p1_phase1_query_level_canonical_table.csv"
PHASE1_SUBSETS = PHASE1 / "d4p1_phase1_subset_definitions.csv"
PHASE1_METRICS = PHASE1 / "d4p1_phase1_subset_robustness_metrics.csv"
PHASE1_CLUSTER = PHASE1 / "d4p1_phase1_old_fragment_cluster_metrics.csv"
PHASE1_VERDICT = PHASE1 / "D4P1_PHASE1_SUBSET_ROBUSTNESS_VERDICT.md"

PHASE2_COMPONENT = PHASE2 / "d4p1_phase2_component_contribution_metrics.csv"
PHASE2_SUBSET_MATRIX = PHASE2 / "d4p1_phase2_subset_component_matrix.csv"
PHASE2_MECH = PHASE2 / "d4p1_phase2_mechanism_hypothesis_tests.csv"
PHASE2_NEG = PHASE2 / "d4p1_phase2_negative_subspace_analysis.csv"
PHASE2_VERDICT = PHASE2 / "D4P1_PHASE2_COMPONENT_CONTRIBUTION_VERDICT.md"
PHASE2_INTERP = PHASE2 / "D4P1_PHASE2_COMPONENT_INTERPRETATION.md"

MANIFEST = D4A0 / "d4a0_query_split_manifest.jsonl"
DE_STD = D4A2D1R / "d4a2d1r_standardized_predictions.jsonl"
HGB_PREDS = D4A1 / "d4a1_test_predictions.jsonl"
STANDARDIZED_TEST = D4A2D2 / "d4a2d2_standardized_predictions_test.jsonl"
D4A1_MANIFEST = D4A1 / "d4a1_model_artifacts_manifest.json"
D4A1_TRAINING = D4A1 / "d4a1_model_training_summary.csv"
D4A1_SELECTION = D4A1 / "D4A1_MODEL_SELECTION_VERDICT.md"
D4A1R_FI = D4A1R / "d4a1r_feature_importance.csv"
D4A1R_FG = D4A1R / "d4a1r_feature_group_importance.csv"
D4A1R_ABL = D4A1R / "d4a1r_feature_ablation_results.csv"
D4A1R_SUBSET_ANNOT = D4A1R / "d4a1r_test_query_subset_annotations.csv"

G2_CSV = D4A3S / "d4a3s_G2_candidates.csv"
G3_CSV = D4A3S / "d4a3s_G3_candidates.csv"
G4_CSV = D4A3S / "d4a3s_G4_candidates.csv"
G1_LABELS = D4A3T / "d4a3t_candidate_labels_g1.csv"
D4A3R_TOPK = D4A3R / "d4a3r_method_topk_a4c_table.csv"

D4A4_CANON = D4A4 / "d4a4_canonical_candidate_table.csv"
D4A4_TIERS = D4A4 / "d4a4_candidate_final_tiers.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_jsonl(path: Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_pipe(value: object) -> List[str]:
    if pd.isna(value) or value is None or str(value) == "":
        return []
    return [part for part in str(value).split("|") if part]


def pipe_join(values: Iterable[str]) -> str:
    ordered: List[str] = []
    seen = set()
    for value in values:
        if value is None:
            continue
        value = str(value)
        if value == "" or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return "|".join(ordered)


def safe_mean(values: List[float]) -> float:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(mean(clean)) if clean else float("nan")


def safe_rate(values: Iterable[int]) -> float:
    arr = [int(v) for v in values]
    return float(np.mean(arr)) if arr else float("nan")


def canonicalize_fallback(candidate_exact: str) -> str:
    stripped = candidate_exact.replace("*", "")
    mol = Chem.MolFromSmiles(stripped)
    if mol is None:
        return stripped
    return Chem.MolToSmiles(mol, canonical=True)


class FeatureCalculator:
    def __init__(self) -> None:
        self.fp_cache: Dict[str, object] = {}
        self.heavy_cache: Dict[str, int] = {}

    def _mol(self, smiles: str) -> Optional[Chem.Mol]:
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
        if smiles not in self.heavy_cache:
            mol = self._mol(smiles)
            self.heavy_cache[smiles] = 0 if mol is None else int(mol.GetNumHeavyAtoms())
        return self.heavy_cache[smiles]

    def similarity(self, old_smiles: str, candidate_smiles: str) -> float:
        ofp = self.fp(old_smiles)
        cfp = self.fp(candidate_smiles)
        if ofp is None or cfp is None:
            return 0.0
        return float(DataStructs.TanimotoSimilarity(ofp, cfp))

    def heavy_atom_delta(self, old_smiles: str, candidate_smiles: str) -> float:
        return float(abs(self.heavy_atoms(candidate_smiles) - self.heavy_atoms(old_smiles)))


def compute_borda_for_query(
    de_top50: List[Tuple[int, str, float, int]],
    hgb_top50: List[Tuple[int, str, float, int, float, float]],
) -> List[Tuple[int, str, float, int, int]]:
    hgb_rank = {cand: rank for rank, cand, *_ in hgb_top50}
    scored = []
    for de_rank, cand, _, label in de_top50:
        borda_score = (TOPR - de_rank + 1) + (TOPR - hgb_rank.get(cand, TOPR + 1) + 1)
        scored.append((cand, float(borda_score), de_rank, label))
    scored.sort(key=lambda item: -item[1])
    ranked: List[Tuple[int, str, float, int, int]] = []
    for idx, (cand, borda_score, de_rank, label) in enumerate(scored, start=1):
        ranked.append((idx, cand, borda_score, de_rank, label))
    return ranked


def build_input_discovery() -> pd.DataFrame:
    specs = [
        (PHASE0_CANONICAL, "phase0", "canonical_table", "required"),
        (PHASE0_FIG, "phase0", "phase0_component_curve", "required"),
        (PHASE0_VERDICT, "phase0", "phase0_verdict", "required"),
        (PHASE1_QUERY, "phase1", "phase1_query_table", "required"),
        (PHASE1_SUBSETS, "phase1", "phase1_subset_defs", "required"),
        (PHASE1_METRICS, "phase1", "phase1_subset_metrics", "required"),
        (PHASE1_CLUSTER, "phase1", "phase1_cluster_metrics", "required"),
        (PHASE1_VERDICT, "phase1", "phase1_verdict", "required"),
        (PHASE2_COMPONENT, "phase2", "phase2_component_metrics", "required"),
        (PHASE2_SUBSET_MATRIX, "phase2", "phase2_subset_matrix", "required"),
        (PHASE2_MECH, "phase2", "phase2_mechanism_tests", "required"),
        (PHASE2_NEG, "phase2", "phase2_negative_subspaces", "required"),
        (PHASE2_VERDICT, "phase2", "phase2_verdict", "required"),
        (PHASE2_INTERP, "phase2", "phase2_interpretation", "required"),
        (DE_STD, "upstream", "standardized_attach_de_predictions", "required"),
        (HGB_PREDS, "upstream", "hgb_test_predictions", "required"),
        (STANDARDIZED_TEST, "upstream", "candidate_norm_reference", "required"),
        (MANIFEST, "upstream", "query_manifest", "required"),
        (D4A1_MANIFEST, "upstream", "hgb_feature_schema", "optional"),
        (D4A1_TRAINING, "upstream", "hgb_training_summary", "optional"),
        (D4A1_SELECTION, "upstream", "hgb_selection_verdict", "optional"),
        (D4A1R_FI, "upstream", "hgb_feature_importance", "optional"),
        (D4A1R_FG, "upstream", "hgb_feature_group_importance", "optional"),
        (D4A1R_ABL, "upstream", "hgb_feature_ablation", "optional"),
        (D4A1R_SUBSET_ANNOT, "upstream", "hgb_subset_annotations", "optional"),
        (G2_CSV, "upstream", "g2_provenance", "optional"),
        (G3_CSV, "upstream", "g3_provenance", "optional"),
        (G4_CSV, "upstream", "g4_provenance", "optional"),
        (G1_LABELS, "upstream", "g1_a4c_labels", "optional"),
        (D4A3R_TOPK, "upstream", "method_topk_a4c_proxy", "optional"),
        (D4A4_CANON, "upstream", "d4a4_canonical_candidate_hub", "optional"),
        (D4A4_TIERS, "upstream", "d4a4_final_tiers", "optional"),
    ]
    rows = []
    for path, stage, role, req in specs:
        record = {
            "file_path": str(path),
            "stage": stage,
            "role": role,
            "required_or_optional": req,
            "has_query_id": False,
            "has_method": False,
            "has_rank": False,
            "has_score": False,
            "has_candidate_smiles": False,
            "has_old_fragment_smiles": False,
            "has_a4c_tier": False,
            "status": "FOUND" if path.exists() else "MISSING",
            "notes": "",
        }
        if not path.exists():
            rows.append(record)
            continue
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path, nrows=2)
            cols = set(df.columns)
            record["has_query_id"] = any(c in cols for c in ["query_id", "qid"])
            record["has_method"] = "method" in cols
            record["has_rank"] = any(c in cols for c in ["rank", "topK_rank", "rank_HGB", "rank_DE", "rank_Borda"])
            record["has_score"] = any(c in cols for c in ["score", "score_HGB", "score_DE", "score_Borda", "metric_value"])
            record["has_candidate_smiles"] = any(c in cols for c in ["candidate", "candidate_norm", "replacement_smiles"])
            record["has_old_fragment_smiles"] = "old_fragment_smiles" in cols or "old_fragment" in cols
            record["has_a4c_tier"] = any(c in cols for c in ["final_action_tier", "a4c_review_bucket", "review_bucket"])
        elif suffix == ".jsonl":
            first = next(load_jsonl(path))
            keys = set(first.keys())
            record["has_query_id"] = any(k in keys for k in ["query_id", "qid", "q"])
            record["has_method"] = any(k in keys for k in ["method", "m"])
            record["has_rank"] = any(k in keys for k in ["rank", "r"])
            record["has_score"] = any(k in keys for k in ["score", "s"])
            record["has_candidate_smiles"] = any(k in keys for k in ["candidate", "c"])
            record["has_old_fragment_smiles"] = "old_fragment_smiles" in keys
            record["has_a4c_tier"] = any(k in keys for k in ["final_action_tier", "a4c_review_bucket"])
        else:
            record["notes"] = "Reference markdown/json file."
        rows.append(record)
    return pd.DataFrame(rows)


def load_query_analysis_table() -> pd.DataFrame:
    query_df = pd.read_csv(PHASE1_QUERY)
    query_df["query_id"] = query_df["query_id"].astype(str)
    query_df["num_positives"] = query_df["positive_replacement_set_size"].astype(int)
    query_df["single_pos_flag"] = (query_df["num_positives"] == 1).astype(int)
    query_df["multi_pos_flag"] = (query_df["num_positives"] > 1).astype(int)
    query_df["hard_top10_miss_flag"] = (query_df["attach_hit10"].fillna(0).astype(int) == 0).astype(int)
    query_df["easy_top10_hit_flag"] = (query_df["attach_hit10"].fillna(0).astype(int) == 1).astype(int)
    query_df["rare_replacement_flag"] = (query_df["target_replacement_frequency_bin"] == "rare_replacement").astype(int)
    query_df["frequent_replacement_flag"] = (query_df["target_replacement_frequency_bin"] == "frequent_replacement").astype(int)
    query_df["negative_subspace_flag"] = (
        (query_df["attachment_signature"] == "C|O")
        | query_df["old_fragment_cluster_id"].isin(["cluster_05", "cluster_09"])
    ).astype(int)
    query_df["EMBEDDING_ANALYSIS_PARTIAL"] = 1
    return query_df


def load_exact_to_norm_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in load_jsonl(STANDARDIZED_TEST):
        mapping.setdefault(str(row["candidate"]), str(row["candidate_norm"]))
    return mapping


def load_attach_de_top50(query_ids: set[str]) -> Dict[str, Dict[str, List[Tuple[int, str, float, int]]]]:
    methods = {"M1_attach": "attach", "M2_DE": "de"}
    store = {"attach": defaultdict(list), "de": defaultdict(list)}
    for row in load_jsonl(DE_STD):
        qid = str(row["q"])
        method = methods.get(str(row["m"]))
        if qid not in query_ids or method is None:
            continue
        rank = int(row["r"])
        if rank > TOPR:
            continue
        store[method][qid].append((rank, str(row["c"]), float(row.get("s", 0.0)), int(row.get("l", 0))))
    for method in store:
        for qid in store[method]:
            store[method][qid].sort(key=lambda item: item[0])
    return store


def load_group_map() -> Dict[Tuple[str, str], str]:
    mapping: Dict[Tuple[str, str], str] = {}
    for path, label in (
        (G2_CSV, "G2_pure_borda_only"),
        (G3_CSV, "G3_de_retained_by_borda"),
        (G4_CSV, "G4_shared"),
    ):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            mapping[(str(row["qid"]), str(row["candidate_norm"]))] = label
    return mapping


def load_candidate_diag_map(group_map: Dict[Tuple[str, str], str]) -> Dict[Tuple[str, str], dict]:
    mapping: Dict[Tuple[str, str], dict] = {}
    if D4A4_CANON.exists():
        canon_df = pd.read_csv(D4A4_CANON)
        for _, row in canon_df.iterrows():
            key = (str(row["query_id"]), str(row["candidate_norm"]))
            mapping[key] = {
                "group_origin": str(row.get("group_origin", group_map.get(key, "other"))),
                "final_action_tier": str(row.get("a4c_review_bucket", "A4C_UNKNOWN")),
                "a4c_review_bucket": str(row.get("a4c_review_bucket", "A4C_UNKNOWN")),
                "hard_alert_flag": bool(row.get("hard_alert_flag", False)),
                "review_ready_flag": bool(row.get("review_ready_flag", False)),
                "alert_type": str(row.get("alert_type", "")),
                "alert_reason_codes": str(row.get("alert_reason_codes", "")),
                "property_warning_flag": bool(row.get("property_warning_flag", False)),
                "gap_type": str(row.get("gap_type", "")),
            }
    if D4A4_TIERS.exists():
        tier_df = pd.read_csv(D4A4_TIERS)
        for _, row in tier_df.iterrows():
            key = (str(row["query_id"]), str(row["candidate_norm"]))
            entry = mapping.setdefault(
                key,
                {
                    "group_origin": group_map.get(key, "other"),
                    "final_action_tier": "Tier0_DATA_PENDING",
                    "a4c_review_bucket": "A4C_UNKNOWN",
                    "hard_alert_flag": False,
                    "review_ready_flag": False,
                    "alert_type": "",
                    "alert_reason_codes": "",
                    "property_warning_flag": False,
                    "gap_type": "",
                },
            )
            entry["group_origin"] = str(row.get("group_origin", entry["group_origin"]))
            entry["final_action_tier"] = str(row.get("final_action_tier", entry["final_action_tier"]))
            entry["a4c_review_bucket"] = str(row.get("a4c_review_bucket", entry["a4c_review_bucket"]))
            entry["hard_alert_flag"] = bool(row.get("hard_alert_flag", entry["hard_alert_flag"]))
            entry["review_ready_flag"] = bool(row.get("review_ready_flag", entry["review_ready_flag"]))
    if G1_LABELS.exists():
        g1_df = pd.read_csv(G1_LABELS)
        for _, row in g1_df.iterrows():
            key = (str(row["qid"]), str(row["candidate_norm"]))
            entry = mapping.setdefault(
                key,
                {
                    "group_origin": group_map.get(key, "other"),
                    "final_action_tier": "Tier0_DATA_PENDING",
                    "a4c_review_bucket": "A4C_UNKNOWN",
                    "hard_alert_flag": False,
                    "review_ready_flag": False,
                    "alert_type": "",
                    "alert_reason_codes": "",
                    "property_warning_flag": False,
                    "gap_type": "",
                },
            )
            if str(row.get("a4c_label", "")):
                entry["a4c_review_bucket"] = str(row.get("a4c_label", entry["a4c_review_bucket"]))
            if str(row.get("gap_type", "")):
                entry["gap_type"] = str(row.get("gap_type", entry["gap_type"]))
    return mapping


def gather_subset_info(sub_df: pd.DataFrame) -> dict:
    return {
        "N_queries": int(len(sub_df)),
        "rare_replacement_rate": safe_rate(sub_df["rare_replacement_flag"].tolist()),
        "hard_top10_miss_rate": safe_rate(sub_df["hard_top10_miss_flag"].tolist()),
        "single_pos_rate": safe_rate(sub_df["single_pos_flag"].tolist()),
        "DE_hit10_rate": float(sub_df["DE_hit10"].mean()) if len(sub_df) else float("nan"),
        "HGB_hit10_rate": float(sub_df["HGB_hit10"].mean()) if len(sub_df) else float("nan"),
        "Borda_hit10_rate": float(sub_df["Borda_hit10"].mean()) if len(sub_df) else float("nan"),
    }


def build_candidate_and_method_tables(
    query_df: pd.DataFrame,
    exact_to_norm: Dict[str, str],
    attach_de_top50: Dict[str, Dict[str, List[Tuple[int, str, float, int]]]],
    candidate_diag_map: Dict[Tuple[str, str], dict],
    group_map: Dict[Tuple[str, str], str],
) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, dict], Dict[str, List[Tuple[int, str, float, int, int]]], dict]:
    fieldnames = [
        "query_id",
        "old_fragment_smiles",
        "attachment_signature",
        "candidate_smiles",
        "candidate_norm",
        "method",
        "rank",
        "score",
        "is_positive_eval_only",
        "group_origin",
        "A4C_tier_if_available",
        "a4c_review_bucket_if_available",
        "alert_flag_if_available",
        "alert_type_if_available",
        "alert_reason_if_available",
        "review_ready_flag_if_available",
        "property_delta_if_available",
        "morgan_similarity_old_candidate_if_available",
        "global_freq_if_available",
        "attach_freq_if_available",
    ]
    candidate_path = OUT / "d4p1_phase3_candidate_analysis_table.csv"
    feature_calc = FeatureCalculator()
    query_map = {str(row["query_id"]): row for _, row in query_df.iterrows()}
    query_ids = set(query_map)
    method_best: Dict[str, Dict[str, dict]] = defaultdict(dict)
    pool_stats: Dict[str, dict] = {}
    borda_top50_map: Dict[str, List[Tuple[int, str, float, int, int]]] = {}
    reconstruction_check = {"borda_best_rank_mismatches": 0}

    with open(candidate_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        current_qid: Optional[str] = None
        current_rows: List[dict] = []
        processed = 0

        def process_query(qid: str, rows: List[dict]) -> None:
            nonlocal processed
            if qid not in query_ids:
                return
            qrow = query_map[qid]
            old_smiles = str(qrow["old_fragment_smiles"])
            attach_sig = str(qrow["attachment_signature"])
            positive_exact = set(split_pipe(qrow["positive_replacements_exact"]))
            positive_norm = set(split_pipe(qrow["positive_replacements_norm"]))

            scored_rows = []
            for row in rows:
                scored_rows.append(
                    (
                        float(row.get("score", 0.0)),
                        str(row["candidate"]),
                        int(row.get("label", 0)),
                        float(row.get("global_freq", 0.0)),
                        float(row.get("attach_freq", 0.0)),
                    )
                )
            scored_rows.sort(key=lambda item: (-item[0], item[1]))
            hgb_all = []
            for rank, (score, cand, label, gf, af) in enumerate(scored_rows, start=1):
                hgb_all.append((rank, cand, score, label, gf, af))
            hgb_top50 = hgb_all[:TOPR]

            full_feature_map = {}
            pool_gf = []
            pool_af = []
            pool_sim = []
            pool_delta = []
            freq_lookup = {}
            for rank, cand, score, label, gf, af in hgb_all:
                freq_lookup[cand] = (gf, af)
                sim = feature_calc.similarity(old_smiles, cand)
                delta = feature_calc.heavy_atom_delta(old_smiles, cand)
                full_feature_map[cand] = {"similarity": sim, "heavy_atom_delta": delta}
                pool_gf.append(gf)
                pool_af.append(af)
                pool_sim.append(sim)
                pool_delta.append(delta)
            pool_stats[qid] = {
                "candidate_pool_mean_global_freq": safe_mean(pool_gf),
                "candidate_pool_mean_attach_freq": safe_mean(pool_af),
                "candidate_pool_mean_similarity": safe_mean(pool_sim),
                "candidate_pool_mean_heavy_atom_delta": safe_mean(pool_delta),
            }

            attach_top50 = attach_de_top50["attach"][qid]
            de_top50 = attach_de_top50["de"][qid]
            borda_top50 = compute_borda_for_query(de_top50, hgb_top50)
            borda_top50_map[qid] = borda_top50

            expected_borda = qrow["Borda_best_rank"]
            got_borda = math.inf
            for rank, cand, _, _, label in borda_top50:
                if label == 1 or cand in positive_exact:
                    got_borda = float(rank)
                    break
            expected_borda = float(expected_borda) if not pd.isna(expected_borda) else math.inf
            if (math.isfinite(expected_borda) != math.isfinite(got_borda)) or (
                math.isfinite(expected_borda) and int(expected_borda) != int(got_borda)
            ):
                reconstruction_check["borda_best_rank_mismatches"] += 1

            def diag_for(cand: str) -> Tuple[str, dict]:
                norm = exact_to_norm.get(cand, canonicalize_fallback(cand))
                diag = candidate_diag_map.get((qid, norm))
                if diag is None:
                    diag = {
                        "group_origin": group_map.get((qid, norm), "other"),
                        "final_action_tier": "Tier0_DATA_PENDING",
                        "a4c_review_bucket": "A4C_UNKNOWN",
                        "hard_alert_flag": False,
                        "review_ready_flag": False,
                        "alert_type": "",
                        "alert_reason_codes": "",
                        "property_warning_flag": False,
                        "gap_type": "",
                    }
                return norm, diag

            def write_rows(method: str, ranked_rows: List[tuple], score_idx: int = 2) -> None:
                positive_best = None
                for record in ranked_rows[:TOPR]:
                    rank = int(record[0])
                    cand = str(record[1])
                    score = float(record[score_idx])
                    norm, diag = diag_for(cand)
                    feat = full_feature_map.get(cand)
                    if feat is None:
                        feat = {
                            "similarity": feature_calc.similarity(old_smiles, cand),
                            "heavy_atom_delta": feature_calc.heavy_atom_delta(old_smiles, cand),
                        }
                    gf, af = freq_lookup.get(cand, (float("nan"), float("nan")))
                    is_pos = int(cand in positive_exact or norm in positive_norm)
                    writer.writerow(
                        {
                            "query_id": qid,
                            "old_fragment_smiles": old_smiles,
                            "attachment_signature": attach_sig,
                            "candidate_smiles": cand,
                            "candidate_norm": norm,
                            "method": method,
                            "rank": rank,
                            "score": score,
                            "is_positive_eval_only": is_pos,
                            "group_origin": diag["group_origin"],
                            "A4C_tier_if_available": diag["final_action_tier"],
                            "a4c_review_bucket_if_available": diag["a4c_review_bucket"],
                            "alert_flag_if_available": int(bool(diag["hard_alert_flag"])),
                            "alert_type_if_available": diag["alert_type"],
                            "alert_reason_if_available": diag["alert_reason_codes"],
                            "review_ready_flag_if_available": int(bool(diag["review_ready_flag"])),
                            "property_delta_if_available": feat["heavy_atom_delta"],
                            "morgan_similarity_old_candidate_if_available": feat["similarity"],
                            "global_freq_if_available": gf,
                            "attach_freq_if_available": af,
                        }
                    )
                    if positive_best is None and is_pos:
                        positive_best = {
                            "rank": rank,
                            "candidate_smiles": cand,
                            "candidate_norm": norm,
                            "score": score,
                            "global_freq": gf,
                            "attach_freq": af,
                            "morgan_similarity": feat["similarity"],
                            "property_delta": feat["heavy_atom_delta"],
                            "group_origin": diag["group_origin"],
                            "final_action_tier": diag["final_action_tier"],
                            "a4c_review_bucket": diag["a4c_review_bucket"],
                            "hard_alert_flag": int(bool(diag["hard_alert_flag"])),
                            "review_ready_flag": int(bool(diag["review_ready_flag"])),
                            "alert_reason_codes": diag["alert_reason_codes"],
                        }
                if positive_best is None:
                    positive_best = {
                        "rank": float("inf"),
                        "candidate_smiles": "",
                        "candidate_norm": "",
                        "score": float("nan"),
                        "global_freq": float("nan"),
                        "attach_freq": float("nan"),
                        "morgan_similarity": float("nan"),
                        "property_delta": float("nan"),
                        "group_origin": "NO_HIT",
                        "final_action_tier": "NO_HIT",
                        "a4c_review_bucket": "NO_HIT",
                        "hard_alert_flag": 0,
                        "review_ready_flag": 0,
                        "alert_reason_codes": "",
                    }
                method_best[qid][method] = positive_best

            write_rows("M1_attach", attach_top50, score_idx=2)
            write_rows("M2_DE", de_top50, score_idx=2)
            write_rows("M3_HGB", hgb_top50, score_idx=2)
            write_rows("M6_Borda_DE_HGB", borda_top50, score_idx=2)
            processed += 1
            if processed % 2000 == 0:
                log(f"  processed {processed} queries for candidate analysis")

        for row in load_jsonl(HGB_PREDS):
            qid = str(row["query_id"])
            if qid not in query_ids:
                continue
            if current_qid is None:
                current_qid = qid
            if qid != current_qid:
                process_query(current_qid, current_rows)
                current_qid = qid
                current_rows = [row]
            else:
                current_rows.append(row)
        if current_qid is not None:
            process_query(current_qid, current_rows)
    return method_best, pool_stats, borda_top50_map, reconstruction_check


def build_de_outputs(query_df: pd.DataFrame, method_best: Dict[str, Dict[str, dict]], pool_stats: Dict[str, dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_specs = [
        ("DE_only_hit", query_df[(query_df["DE_hit10"] == 1) & (query_df["HGB_hit10"] == 0)]),
        ("HGB_only_hit", query_df[(query_df["HGB_hit10"] == 1) & (query_df["DE_hit10"] == 0)]),
        ("all_hit", query_df[(query_df["DE_hit10"] == 1) & (query_df["HGB_hit10"] == 1) & (query_df["Borda_hit10"] == 1)]),
        ("Borda_gain_queries", query_df[(query_df["Borda_hit10"] == 1) & (query_df["HGB_hit10"] == 0)]),
        ("G3_positive_queries", query_df[query_df["borda_hit10_has_g3_positive"] == 1]),
        ("G4_shared_queries", query_df[query_df["borda_hit10_has_g4_positive"] == 1]),
        ("Borda_miss_queries", query_df[query_df["Borda_hit10"] == 0]),
    ]
    summary_rows = []
    for group_name, sub_df in group_specs:
        qids = sub_df["query_id"].astype(str).tolist()
        de_ranks = []
        hgb_ranks = []
        borda_ranks = []
        de_sim = []
        hgb_sim = []
        de_gf = []
        hgb_gf = []
        de_af = []
        hgb_af = []
        a4c_cov = []
        hard_alert = []
        for qid in qids:
            de_info = method_best[qid]["M2_DE"]
            hgb_info = method_best[qid]["M3_HGB"]
            borda_info = method_best[qid]["M6_Borda_DE_HGB"]
            if math.isfinite(float(de_info["rank"])):
                de_ranks.append(float(de_info["rank"]))
            if math.isfinite(float(hgb_info["rank"])):
                hgb_ranks.append(float(hgb_info["rank"]))
            if math.isfinite(float(borda_info["rank"])):
                borda_ranks.append(float(borda_info["rank"]))
            if not math.isnan(float(de_info["morgan_similarity"])):
                de_sim.append(float(de_info["morgan_similarity"]))
            if not math.isnan(float(hgb_info["morgan_similarity"])):
                hgb_sim.append(float(hgb_info["morgan_similarity"]))
            if not math.isnan(float(de_info["global_freq"])):
                de_gf.append(float(de_info["global_freq"]))
            if not math.isnan(float(hgb_info["global_freq"])):
                hgb_gf.append(float(hgb_info["global_freq"]))
            if not math.isnan(float(de_info["attach_freq"])):
                de_af.append(float(de_info["attach_freq"]))
            if not math.isnan(float(hgb_info["attach_freq"])):
                hgb_af.append(float(hgb_info["attach_freq"]))
            a4c_cov.append(int(de_info["final_action_tier"] not in {"Tier0_DATA_PENDING", "NO_HIT"}))
            hard_alert.append(int(de_info["hard_alert_flag"]))
        summary_rows.append(
            {
                "analysis_group": group_name,
                **gather_subset_info(sub_df),
                "mean_DE_best_rank": safe_mean(de_ranks),
                "mean_HGB_best_rank": safe_mean(hgb_ranks),
                "mean_Borda_best_rank": safe_mean(borda_ranks),
                "mean_DE_positive_morgan_similarity": safe_mean(de_sim),
                "mean_HGB_positive_morgan_similarity": safe_mean(hgb_sim),
                "mean_DE_positive_global_freq": safe_mean(de_gf),
                "mean_HGB_positive_global_freq": safe_mean(hgb_gf),
                "mean_DE_positive_attach_freq": safe_mean(de_af),
                "mean_HGB_positive_attach_freq": safe_mean(hgb_af),
                "DE_positive_a4c_coverage_rate": safe_rate(a4c_cov),
                "DE_positive_hard_alert_rate": safe_rate(hard_alert),
                "interpretability_scope": "EMBEDDING_ANALYSIS_PARTIAL",
            }
        )

    subset_specs = [
        ("all_test", query_df),
        ("rare_replacement", query_df[query_df["rare_replacement_flag"] == 1]),
        ("frequent_replacement", query_df[query_df["frequent_replacement_flag"] == 1]),
        ("hard_top10_miss", query_df[query_df["hard_top10_miss_flag"] == 1]),
        ("easy_top10_hit", query_df[query_df["easy_top10_hit_flag"] == 1]),
        ("single_pos", query_df[query_df["single_pos_flag"] == 1]),
        ("multi_pos", query_df[query_df["multi_pos_flag"] == 1]),
        ("G3_positive_queries", query_df[query_df["borda_hit10_has_g3_positive"] == 1]),
    ]
    subset_rows = []
    for subset_name, sub_df in subset_specs:
        qids = sub_df["query_id"].astype(str).tolist()
        de_sim = []
        de_gf = []
        de_af = []
        pool_gf = []
        pool_af = []
        pool_sim = []
        for qid in qids:
            info = method_best[qid]["M2_DE"]
            if not math.isnan(float(info["morgan_similarity"])):
                de_sim.append(float(info["morgan_similarity"]))
            if not math.isnan(float(info["global_freq"])):
                de_gf.append(float(info["global_freq"]))
            if not math.isnan(float(info["attach_freq"])):
                de_af.append(float(info["attach_freq"]))
            pool = pool_stats.get(qid, {})
            pool_gf.append(float(pool.get("candidate_pool_mean_global_freq", float("nan"))))
            pool_af.append(float(pool.get("candidate_pool_mean_attach_freq", float("nan"))))
            pool_sim.append(float(pool.get("candidate_pool_mean_similarity", float("nan"))))
        subset_rows.append(
            {
                "subset_name": subset_name,
                "N_queries": int(len(sub_df)),
                "Attach_Top10": float(sub_df["attach_hit10"].mean()) if len(sub_df) else float("nan"),
                "DE_Top10": float(sub_df["DE_hit10"].mean()) if len(sub_df) else float("nan"),
                "HGB_Top10": float(sub_df["HGB_hit10"].mean()) if len(sub_df) else float("nan"),
                "Borda_Top10": float(sub_df["Borda_hit10"].mean()) if len(sub_df) else float("nan"),
                "DE_minus_attach": float(sub_df["DE_hit10"].mean() - sub_df["attach_hit10"].mean()) if len(sub_df) else float("nan"),
                "DE_minus_HGB": float(sub_df["DE_hit10"].mean() - sub_df["HGB_hit10"].mean()) if len(sub_df) else float("nan"),
                "mean_DE_positive_morgan_similarity": safe_mean(de_sim),
                "mean_DE_positive_global_freq": safe_mean(de_gf),
                "mean_DE_positive_attach_freq": safe_mean(de_af),
                "candidate_pool_mean_global_freq": safe_mean(pool_gf),
                "candidate_pool_mean_attach_freq": safe_mean(pool_af),
                "candidate_pool_mean_similarity": safe_mean(pool_sim),
                "interpretability_scope": "EMBEDDING_ANALYSIS_PARTIAL",
            }
        )

    comparison_specs = [
        ("DE_only_vs_HGB_only", "DE_only_hit", "HGB_only_hit", "summary"),
        ("G3_vs_G4", "G3_positive_queries", "G4_shared_queries", "summary"),
        ("rare_vs_frequent_among_DE_hits", "rare_replacement", "frequent_replacement", "subset"),
        ("hard_vs_easy_among_DE_hits", "hard_top10_miss", "easy_top10_hit", "subset"),
    ]
    summary_df = pd.DataFrame(summary_rows)
    subset_df = pd.DataFrame(subset_rows)
    lookup_summary = {row["analysis_group"]: row for row in summary_rows}
    lookup_subset = {row["subset_name"]: row for row in subset_rows}
    comp_rows = []
    metrics = [
        "mean_DE_positive_global_freq",
        "mean_HGB_positive_global_freq",
        "mean_DE_positive_attach_freq",
        "mean_HGB_positive_attach_freq",
        "mean_DE_positive_morgan_similarity",
        "mean_HGB_positive_morgan_similarity",
    ]
    for comp_name, left, right, source in comparison_specs:
        left_row = lookup_summary[left] if source == "summary" else lookup_subset[left]
        right_row = lookup_summary[right] if source == "summary" else lookup_subset[right]
        for metric in metrics:
            if metric not in left_row or metric not in right_row:
                continue
            comp_rows.append(
                {
                    "comparison_name": comp_name,
                    "source_table": source,
                    "metric": metric,
                    "group_a": left,
                    "group_b": right,
                    "group_a_value": left_row[metric],
                    "group_b_value": right_row[metric],
                    "delta_a_minus_b": left_row[metric] - right_row[metric],
                }
            )
    de_only_qids = query_df[(query_df["DE_hit10"] == 1) & (query_df["HGB_hit10"] == 0)]["query_id"].astype(str).tolist()
    de_only_pool_gf = safe_mean([pool_stats[qid]["candidate_pool_mean_global_freq"] for qid in de_only_qids])
    de_only_pool_af = safe_mean([pool_stats[qid]["candidate_pool_mean_attach_freq"] for qid in de_only_qids])
    de_only_pool_sim = safe_mean([pool_stats[qid]["candidate_pool_mean_similarity"] for qid in de_only_qids])
    left_row = lookup_summary["DE_only_hit"]
    comp_rows.extend(
        [
            {
                "comparison_name": "DE_only_vs_random_pool_same_queries",
                "source_table": "summary_vs_pool",
                "metric": "global_freq",
                "group_a": "DE_only_hit_positive",
                "group_b": "candidate_pool_same_queries",
                "group_a_value": left_row["mean_DE_positive_global_freq"],
                "group_b_value": de_only_pool_gf,
                "delta_a_minus_b": left_row["mean_DE_positive_global_freq"] - de_only_pool_gf,
            },
            {
                "comparison_name": "DE_only_vs_random_pool_same_queries",
                "source_table": "summary_vs_pool",
                "metric": "attach_freq",
                "group_a": "DE_only_hit_positive",
                "group_b": "candidate_pool_same_queries",
                "group_a_value": left_row["mean_DE_positive_attach_freq"],
                "group_b_value": de_only_pool_af,
                "delta_a_minus_b": left_row["mean_DE_positive_attach_freq"] - de_only_pool_af,
            },
            {
                "comparison_name": "DE_only_vs_random_pool_same_queries",
                "source_table": "summary_vs_pool",
                "metric": "morgan_similarity",
                "group_a": "DE_only_hit_positive",
                "group_b": "candidate_pool_same_queries",
                "group_a_value": left_row["mean_DE_positive_morgan_similarity"],
                "group_b_value": de_only_pool_sim,
                "delta_a_minus_b": left_row["mean_DE_positive_morgan_similarity"] - de_only_pool_sim,
            },
        ]
    )
    return summary_df, subset_df, pd.DataFrame(comp_rows)


def build_hgb_outputs(query_df: pd.DataFrame, method_best: Dict[str, Dict[str, dict]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    fi_df = pd.read_csv(D4A1R_FI)
    hgb_manifest = json.loads(D4A1_MANIFEST.read_text(encoding="utf-8"))
    perm_abs = fi_df["permutation_importance_mean"].abs().sum()
    fi_rows = []
    for _, row in fi_df.iterrows():
        fi_rows.append(
            {
                "feature": row["feature"],
                "feature_group": row["feature_group"],
                "native_importance_raw": float(row["importance"]),
                "native_importance_reliable": int(not math.isclose(float(row["importance"]), 1.0 / len(fi_df), rel_tol=1e-6)),
                "permutation_importance_mean": float(row["permutation_importance_mean"]),
                "permutation_importance_std": float(row["permutation_importance_std"]),
                "permutation_importance_abs_pct": float(abs(row["permutation_importance_mean"]) / perm_abs) if perm_abs else float("nan"),
                "interpretability_source": "D4A1R_permutation_fallback",
                "notes": "Primary explanation signal uses permutation importance because native importances are flat/unreliable.",
            }
        )
    fi_out = pd.DataFrame(fi_rows)

    ablation_df = pd.read_csv(D4A1R_ABL)
    group_rows = []
    for group_name, sub in fi_out.groupby("feature_group"):
        group_rows.append(
            {
                "feature_group": group_name,
                "permutation_importance_sum": float(sub["permutation_importance_mean"].sum()),
                "permutation_importance_abs_sum": float(sub["permutation_importance_mean"].abs().sum()),
                "permutation_importance_abs_pct": float(sub["permutation_importance_mean"].abs().sum() / fi_out["permutation_importance_mean"].abs().sum()),
                "schema_features": pipe_join([f for f in hgb_manifest["feature_names"] if (("freq" in f and group_name == "frequency") or ("tanimoto" in f and group_name == "similarity") or ("atom" in f and group_name == "property") or ("match" in f and group_name == "attachment"))]),
                "supporting_ablation_rows": pipe_join(ablation_df["ablation"].tolist()),
                "interpretation": "frequency priors" if group_name == "frequency" else "structural similarity" if group_name == "similarity" else "simple size/property deltas" if group_name == "property" else "attachment/context",
            }
        )
    group_out = pd.DataFrame(group_rows).sort_values("permutation_importance_abs_sum", ascending=False)

    negative_rows = []
    for subset_name, mask in [
        ("attachment_signature_C|O", query_df["attachment_signature"] == "C|O"),
        ("old_fragment_cluster_05", query_df["old_fragment_cluster_id"] == "cluster_05"),
        ("old_fragment_cluster_09", query_df["old_fragment_cluster_id"] == "cluster_09"),
    ]:
        sub_df = query_df[mask]
        qids = sub_df["query_id"].astype(str).tolist()
        hgb_gf = []
        hgb_af = []
        hgb_sim = []
        hgb_delta = []
        for qid in qids:
            info = method_best[qid]["M3_HGB"]
            if math.isfinite(float(info["rank"])):
                hgb_gf.append(float(info["global_freq"]))
                hgb_af.append(float(info["attach_freq"]))
                hgb_sim.append(float(info["morgan_similarity"]))
                hgb_delta.append(float(info["property_delta"]))
        negative_rows.append(
            {
                "subset_name": subset_name,
                "N_queries": int(len(sub_df)),
                "HGB_Top10": float(sub_df["HGB_hit10"].mean()) if len(sub_df) else float("nan"),
                "Borda_Top10": float(sub_df["Borda_hit10"].mean()) if len(sub_df) else float("nan"),
                "Borda_minus_HGB": float(sub_df["Borda_hit10"].mean() - sub_df["HGB_hit10"].mean()) if len(sub_df) else float("nan"),
                "mean_HGB_positive_global_freq": safe_mean(hgb_gf),
                "mean_HGB_positive_attach_freq": safe_mean(hgb_af),
                "mean_HGB_positive_morgan_similarity": safe_mean(hgb_sim),
                "mean_HGB_positive_property_delta": safe_mean(hgb_delta),
                "dominant_replacement_frequency_bin": pipe_join([f"{k} ({v})" for k, v in Counter(sub_df["target_replacement_frequency_bin"]).most_common(3)]),
                "dominant_attachment_signatures": pipe_join([f"{k} ({v})" for k, v in Counter(sub_df["attachment_signature"]).most_common(3)]),
                "interpretation": "HGB appears comparatively strong in this negative subspace; compare to DE/Borda in the deep-dive report.",
            }
        )
    negative_out = pd.DataFrame(negative_rows)

    shap_blocked = f"""# D4P1-Phase3 HGB SHAP Blocked

## Status
SHAP_BLOCKED

## Evidence
- No serialized D4A1/D4A1R HistGradientBoosting model artifact is available under `{D4A1}` or `{D4A1R}`.
- The reusable explainability artifacts that do exist are `{D4A1R_FI.name}`, `{D4A1R_FG.name}`, and `{D4A1R_ABL.name}`.
- The `importance` column in `{D4A1R_FI.name}` is flat and therefore not a reliable primary explanation signal.

## Fallback Used
1. Permutation importance from `{D4A1R_FI.name}`.
2. Feature-family aggregation over the 7-feature schema from `{D4A1_MANIFEST.name}`.
    3. Diagnostic ablation support from `{D4A1R_ABL.name}`.
"""
    return fi_out, group_out, negative_out, shap_blocked


def build_borda_complementarity_outputs(query_df: pd.DataFrame, method_best: Dict[str, Dict[str, dict]]) -> Tuple[pd.DataFrame, str]:
    categories = {
        "DE_only_hit": query_df[(query_df["DE_hit10"] == 1) & (query_df["HGB_hit10"] == 0) & (query_df["Borda_hit10"] == 0)],
        "HGB_only_hit": query_df[(query_df["HGB_hit10"] == 1) & (query_df["DE_hit10"] == 0) & (query_df["Borda_hit10"] == 0)],
        "Borda_only_hit": query_df[(query_df["Borda_hit10"] == 1) & (query_df["DE_hit10"] == 0) & (query_df["HGB_hit10"] == 0)],
        "DE_and_HGB_hit": query_df[(query_df["DE_hit10"] == 1) & (query_df["HGB_hit10"] == 1) & (query_df["Borda_hit10"] == 0)],
        "Borda_and_HGB_hit": query_df[(query_df["Borda_hit10"] == 1) & (query_df["HGB_hit10"] == 1) & (query_df["DE_hit10"] == 0)],
        "all_hit": query_df[(query_df["DE_hit10"] == 1) & (query_df["HGB_hit10"] == 1) & (query_df["Borda_hit10"] == 1)],
        "all_miss": query_df[(query_df["DE_hit10"] == 0) & (query_df["HGB_hit10"] == 0) & (query_df["Borda_hit10"] == 0)],
    }
    rows = []
    for cat_name, sub_df in categories.items():
        qids = sub_df["query_id"].astype(str).tolist()
        de_ranks = []
        hgb_ranks = []
        borda_ranks = []
        tier_dist = []
        for qid in qids:
            de_info = method_best[qid]["M2_DE"]
            hgb_info = method_best[qid]["M3_HGB"]
            borda_info = method_best[qid]["M6_Borda_DE_HGB"]
            if math.isfinite(float(de_info["rank"])):
                de_ranks.append(float(de_info["rank"]))
            if math.isfinite(float(hgb_info["rank"])):
                hgb_ranks.append(float(hgb_info["rank"]))
            if math.isfinite(float(borda_info["rank"])):
                borda_ranks.append(float(borda_info["rank"]))
            tier_dist.append(borda_info["final_action_tier"])
        rows.append(
            {
                "category": cat_name,
                "N_queries": int(len(sub_df)),
                "replacement_frequency_distribution": pipe_join([f"{k} ({v})" for k, v in Counter(sub_df["target_replacement_frequency_bin"]).most_common(3)]),
                "hard_top10_miss_rate": safe_rate(sub_df["hard_top10_miss_flag"].tolist()),
                "rare_replacement_rate": safe_rate(sub_df["rare_replacement_flag"].tolist()),
                "old_fragment_cluster_distribution": pipe_join([f"{k} ({v})" for k, v in Counter(sub_df["old_fragment_cluster_id"]).most_common(3)]),
                "attachment_signature_distribution": pipe_join([f"{k} ({v})" for k, v in Counter(sub_df["attachment_signature"]).most_common(3)]),
                "A4C_tier_distribution_if_available": pipe_join([f"{k} ({v})" for k, v in Counter(tier_dist).most_common(3)]),
                "mean_DE_rank": safe_mean(de_ranks),
                "mean_HGB_rank": safe_mean(hgb_ranks),
                "mean_Borda_rank": safe_mean(borda_ranks),
            }
        )
    cat_df = pd.DataFrame(rows)
    borda_gain = query_df[(query_df["Borda_hit10"] == 1) & (query_df["HGB_hit10"] == 0)]
    gain_de_direct = int((borda_gain["DE_hit10"] == 1).sum())
    gain_midrank = int(((borda_gain["DE_hit10"] == 0) & (borda_gain["DE_hit50"] == 1)).sum()) if "DE_hit50" in borda_gain.columns else 0
    summary = f"""# D4P1-Phase3 Borda Complementarity Summary

## Evidence
- Canonical overall proposal gain remains Phase2-locked: Borda(DE,HGB) - HGB = +0.0425.
- `Borda_only_hit` queries exist and therefore the fusion is not only replaying direct DE or HGB top-10 wins.
- Among Borda net-gain queries (`Borda_hit10=1` and `HGB_hit10=0`):
  - direct DE-rescue cases (`DE_hit10=1`) = {gain_de_direct}
  - mid-rank DE promotion cases (`DE_hit10=0` but DE within top-50) = {gain_midrank}

## Inference
- Borda gain is primarily consistent with DE complementing HGB, not with a universal DE advantage.
- The existence of Borda-only gain cases indicates that simple score pooling can promote mid-rank structural candidates into the top-10.
- This also explains why hand-written routing underperformed simple Borda in D4A2D2R: the useful complementarity is not confined to a separable query subset.

## Unknowns
- Because DE embeddings are not available, this complementarity explanation remains `EMBEDDING_ANALYSIS_PARTIAL`.
- Candidate-level A4C coverage is incomplete outside the audited gain regions, so risk statements remain diagnostic only.
"""
    return cat_df, summary


def build_negative_subspace_outputs(query_df: pd.DataFrame, method_best: Dict[str, Dict[str, dict]]) -> Tuple[pd.DataFrame, str]:
    rows = []
    report_lines = ["# D4P1-Phase3 Negative Subspace Report", "", "## Evidence"]
    for subset_name, mask in [
        ("attachment_signature_C|O", query_df["attachment_signature"] == "C|O"),
        ("old_fragment_cluster_05", query_df["old_fragment_cluster_id"] == "cluster_05"),
        ("old_fragment_cluster_09", query_df["old_fragment_cluster_id"] == "cluster_09"),
    ]:
        sub_df = query_df[mask]
        qids = sub_df["query_id"].astype(str).tolist()
        de_ranks = []
        hgb_ranks = []
        borda_ranks = []
        de_hit = float(sub_df["DE_hit10"].mean()) if len(sub_df) else float("nan")
        hgb_hit = float(sub_df["HGB_hit10"].mean()) if len(sub_df) else float("nan")
        borda_hit = float(sub_df["Borda_hit10"].mean()) if len(sub_df) else float("nan")
        g_comp = []
        tiers = []
        hgb_gf = []
        de_gf = []
        for qid in qids:
            de_info = method_best[qid]["M2_DE"]
            hgb_info = method_best[qid]["M3_HGB"]
            borda_info = method_best[qid]["M6_Borda_DE_HGB"]
            if math.isfinite(float(de_info["rank"])):
                de_ranks.append(float(de_info["rank"]))
            if math.isfinite(float(hgb_info["rank"])):
                hgb_ranks.append(float(hgb_info["rank"]))
            if math.isfinite(float(borda_info["rank"])):
                borda_ranks.append(float(borda_info["rank"]))
            g_comp.append(borda_info["group_origin"])
            tiers.append(borda_info["final_action_tier"])
            if not math.isnan(float(hgb_info["global_freq"])):
                hgb_gf.append(float(hgb_info["global_freq"]))
            if not math.isnan(float(de_info["global_freq"])):
                de_gf.append(float(de_info["global_freq"]))
        dominant_old = pipe_join([f"{k} ({v})" for k, v in Counter(sub_df["old_fragment_smiles"]).most_common(5)])
        dominant_repl = pipe_join([f"{k} ({v})" for k, v in Counter([cand for value in sub_df["positive_replacements_exact"] for cand in split_pipe(value)]).most_common(5)])
        dominant_attach = pipe_join([f"{k} ({v})" for k, v in Counter(sub_df["attachment_signature"]).most_common(5)])
        explanation = []
        if hgb_hit > borda_hit and hgb_hit > de_hit:
            explanation.append("HGB uniquely strong relative to DE/Borda")
        if de_hit < hgb_hit:
            explanation.append("DE weakness appears primary")
        if borda_hit < de_hit:
            explanation.append("Borda fusion amplifies the weaker DE signal in this regime")
        if subset_name == "old_fragment_cluster_09":
            explanation.append("KNOWN_FAILURE_MODE_UNEXPLAINED only partially resolved; dominant pattern is benzylic O-linked aromatic chemistry")
        rows.append(
            {
                "subset_name": subset_name,
                "N_queries": int(len(sub_df)),
                "Attach_Top10": float(sub_df["attach_hit10"].mean()) if len(sub_df) else float("nan"),
                "DE_Top10": de_hit,
                "HGB_Top10": hgb_hit,
                "Borda_Top10": borda_hit,
                "Oracle_Top10_if_available": float(sub_df["Oracle_hit10_if_available"].mean()) if len(sub_df) else float("nan"),
                "Borda_minus_HGB": borda_hit - hgb_hit if len(sub_df) else float("nan"),
                "dominant_old_fragments": dominant_old,
                "dominant_replacement_fragments": dominant_repl,
                "dominant_attachment_signatures": dominant_attach,
                "replacement_frequency_distribution": pipe_join([f"{k} ({v})" for k, v in Counter(sub_df["target_replacement_frequency_bin"]).most_common(3)]),
                "mean_DE_best_rank": safe_mean(de_ranks),
                "mean_HGB_best_rank": safe_mean(hgb_ranks),
                "mean_Borda_best_rank": safe_mean(borda_ranks),
                "mean_DE_positive_global_freq": safe_mean(de_gf),
                "mean_HGB_positive_global_freq": safe_mean(hgb_gf),
                "G2_G3_G4_composition_if_available": pipe_join([f"{k} ({v})" for k, v in Counter(g_comp).most_common(5)]),
                "A4C_tier_distribution_if_available": pipe_join([f"{k} ({v})" for k, v in Counter(tiers).most_common(5)]),
                "plausible_explanation": "; ".join(explanation) if explanation else "KNOWN_FAILURE_MODE_UNEXPLAINED",
            }
        )
        report_lines.append(f"- `{subset_name}`: N={len(sub_df)}, HGB={hgb_hit:.4f}, Borda={borda_hit:.4f}, delta={borda_hit - hgb_hit:+.4f}")
        report_lines.append(f"  Explanation candidate: {'; '.join(explanation) if explanation else 'KNOWN_FAILURE_MODE_UNEXPLAINED'}")
    report_lines.extend(
        [
            "",
            "## Skeptical Review",
            "- These explanations are distributional and post-hoc; they do not prove causal chemical mechanisms.",
            "- Old-fragment clusters are descriptive partitions, not medicinal-chemistry ontologies.",
            "- A4C tier distributions remain diagnostic because coverage is incomplete outside audited regions.",
            "- Weak MMP labels should not be overclaimed as activity-preserving truth.",
            "",
            "## Unknowns",
            "- DE embedding geometry is not available, so latent-space explanations remain partial.",
            "- Cluster_09 remains the most severe unresolved failure mode even though its dominant chemistry is identifiable.",
        ]
    )
    return pd.DataFrame(rows), "\n".join(report_lines) + "\n"


def build_figures(fi_df: pd.DataFrame, cat_df: pd.DataFrame, negative_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comp_fig = cat_df[["category", "N_queries", "hard_top10_miss_rate", "rare_replacement_rate", "mean_DE_rank", "mean_HGB_rank", "mean_Borda_rank"]].copy()
    comp_fig = comp_fig.rename(columns={"category": "row_name"})
    fi_fig = fi_df[["feature", "feature_group", "permutation_importance_mean", "permutation_importance_std", "permutation_importance_abs_pct"]].copy()
    neg_fig = negative_df[["subset_name", "N_queries", "Attach_Top10", "DE_Top10", "HGB_Top10", "Borda_Top10", "Borda_minus_HGB", "mean_DE_best_rank", "mean_HGB_best_rank", "mean_Borda_best_rank"]].copy()
    return comp_fig, fi_fig, neg_fig


def main() -> None:
    log("Building Phase3 input discovery...")
    input_df = build_input_discovery()
    input_df.to_csv(OUT / "d4p1_phase3_4_input_discovery.csv", index=False)
    missing_required = input_df[(input_df["required_or_optional"] == "required") & (input_df["status"] != "FOUND")]
    if not missing_required.empty:
        raise SystemExit("INPUT_MISSING_CANONICAL_PHASE_FILES")

    log("Building Phase3 query analysis table...")
    query_df = load_query_analysis_table()
    query_df.to_csv(OUT / "d4p1_phase3_query_analysis_table.csv", index=False)

    log("Loading candidate mappings and diagnostics...")
    exact_to_norm = load_exact_to_norm_map()
    query_ids = set(query_df["query_id"].astype(str))
    attach_de_top50 = load_attach_de_top50(query_ids)
    group_map = load_group_map()
    candidate_diag_map = load_candidate_diag_map(group_map)

    log("Building candidate analysis table and method-best summaries...")
    method_best, pool_stats, _, reconstruction_check = build_candidate_and_method_tables(
        query_df=query_df,
        exact_to_norm=exact_to_norm,
        attach_de_top50=attach_de_top50,
        candidate_diag_map=candidate_diag_map,
        group_map=group_map,
    )
    log(f"Borda best-rank mismatches vs Phase1 query table: {reconstruction_check['borda_best_rank_mismatches']}")

    log("Building DE signal outputs...")
    de_summary, de_by_subset, de_comp = build_de_outputs(query_df, method_best, pool_stats)
    de_summary.to_csv(OUT / "d4p1_phase3_de_signal_summary.csv", index=False)
    de_by_subset.to_csv(OUT / "d4p1_phase3_de_signal_by_subset.csv", index=False)
    de_comp.to_csv(OUT / "d4p1_phase3_de_vs_hgb_feature_comparison.csv", index=False)

    log("Building HGB explainability outputs...")
    hgb_fi, hgb_fg, hgb_neg, shap_blocked = build_hgb_outputs(query_df, method_best)
    hgb_fi.to_csv(OUT / "d4p1_phase3_hgb_feature_importance.csv", index=False)
    hgb_fg.to_csv(OUT / "d4p1_phase3_hgb_feature_group_importance.csv", index=False)
    hgb_neg.to_csv(OUT / "d4p1_phase3_hgb_negative_subspace_profile.csv", index=False)
    (OUT / "D4P1_PHASE3_HGB_SHAP_BLOCKED.md").write_text(shap_blocked, encoding="utf-8")

    log("Building Borda complementarity outputs...")
    borda_cat, borda_md = build_borda_complementarity_outputs(query_df, method_best)
    borda_cat.to_csv(OUT / "d4p1_phase3_borda_complementarity_categories.csv", index=False)
    (OUT / "d4p1_phase3_borda_complementarity_summary.md").write_text(borda_md, encoding="utf-8")

    log("Building negative-subspace deep dive...")
    negative_df, negative_md = build_negative_subspace_outputs(query_df, method_best)
    negative_df.to_csv(OUT / "d4p1_phase3_negative_subspace_deep_dive.csv", index=False)
    (OUT / "D4P1_PHASE3_NEGATIVE_SUBSPACE_REPORT.md").write_text(negative_md, encoding="utf-8")

    log("Building figure-ready Phase3 data...")
    fig_comp, fig_hgb, fig_neg = build_figures(hgb_fi, borda_cat, negative_df)
    fig_comp.to_csv(OUT / "d4p1_phase3_fig_de_hgb_complementarity_data.csv", index=False)
    fig_hgb.to_csv(OUT / "d4p1_phase3_fig_hgb_feature_importance_data.csv", index=False)
    fig_neg.to_csv(OUT / "d4p1_phase3_fig_negative_subspace_data.csv", index=False)

    log("Phase3 interpretability package complete.")


if __name__ == "__main__":
    main()
