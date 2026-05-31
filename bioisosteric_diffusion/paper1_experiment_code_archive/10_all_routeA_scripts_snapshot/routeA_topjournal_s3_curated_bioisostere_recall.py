#!/usr/bin/env python3
"""Route-A top-journal S3: external curated bioisostere recall.

Primary external source:
Peter Ertl et al., "Ring replacement recommender: Ring modifications for
improving biological activity", with the public rrr-data.txt support file.

The script reports two distinct quantities:
1. zero-shot recall of Route-A rankers on curated ring replacements mapped to
   the Route-A replacement vocabulary;
2. a curated-prior score adapter tuned on external validation templates and
   evaluated on external held-out templates.

The direct RRR lookup score is reported only as a database upper bound.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from routeA_d4s2_common import (  # noqa: E402
    SIG_ORDER,
    feature_columns_for_sets,
    full_attach_arrays,
    full_borda,
    load_baseline_artifacts,
    metrics_from_score_matrix,
    score_de_full,
    score_hgb_full,
    score_matrix_to_ranks,
)
from routeA_topjournal_s1_router_and_review_pilot import (  # noqa: E402
    load_selected_predictor_local,
    row_zscore,
    score_selected_local,
)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("once", category=InconsistentVersionWarning)

SEED = 20260528
PROJECT = Path("E:/zuhui/bioisosteric_diffusion")
PLAN = PROJECT / "plan_results"
DATA = PROJECT / "data"
OUT = PLAN / "routeA_topjournal_s3_curated_bioisostere_recall"
OUT.mkdir(parents=True, exist_ok=True)

RRR_PATH = DATA / "external_rrr_data.txt"
VOCAB_PATH = PLAN / "routeA_chembl37k_d4s_b0_blind_split_baseline" / "d4s_b0_train_vocab.csv"

PRIMARY_MIN_PUBLICATIONS = 5
PRIMARY_MIN_LOG_IMPROVEMENT = 0.0
BETA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
SIM_THRESHOLD_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]


def install_sklearn_loss_pickle_compat() -> None:
    """Allow sklearn 1.3.x HGB artifacts to unpickle under newer sklearn builds."""
    try:
        import sklearn._loss._loss as loss_module
    except Exception:
        return
    if hasattr(loss_module, "__pyx_unpickle_CyHalfBinomialLoss"):
        return

    def _unpickle(cls, checksum, state):
        obj = cls.__new__(cls)
        if state:
            obj.__setstate__(state)
        return obj

    loss_module.__pyx_unpickle_CyHalfBinomialLoss = _unpickle


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5_bucket(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 10


def split_for_template(template: str) -> str:
    bucket = md5_bucket(template)
    if bucket < 6:
        return "external_train"
    if bucket < 8:
        return "external_val"
    return "external_test"


def r_to_star(smiles: str) -> str:
    return smiles.replace("[R]", "*")


def canonical_dummy_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def dummy_neighbor_root(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "C"
    dummy = None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummy = atom
            break
    if dummy is None or not dummy.GetNeighbors():
        return "C"
    nbr = dummy.GetNeighbors()[0]
    symbol = nbr.GetSymbol().upper()
    if symbol in {"C", "N", "O", "S"}:
        return symbol
    return "C"


def infer_attachment_signature(old_fragment: str) -> str:
    root = dummy_neighbor_root(old_fragment)
    if root == "C":
        return "C|C"
    signature = f"C|{root}"
    return signature if signature in SIG_ORDER else "C|C"


def fragment_fp(smiles: str, bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles.replace("*", ""))
    arr = np.zeros(bits, dtype=np.float32)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto_from_arrays(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.dot(a, b))
    denom = float(a.sum() + b.sum() - inter)
    if denom <= 0:
        return 0.0
    return inter / denom


def load_vocab_mapping() -> tuple[list[str], dict[str, int], dict[str, str]]:
    vocab_df = pd.read_csv(VOCAB_PATH)
    vocab = vocab_df["replacement_smiles"].astype(str).tolist()
    vocab_index = {smi: idx for idx, smi in enumerate(vocab)}
    canonical_to_vocab: dict[str, str] = {}
    for smi in vocab:
        can = canonical_dummy_smiles(smi)
        if can is not None:
            canonical_to_vocab.setdefault(can, smi)
    return vocab, vocab_index, canonical_to_vocab


def map_to_vocab(smiles: str, canonical_to_vocab: dict[str, str]) -> str | None:
    star = r_to_star(smiles)
    if star in canonical_to_vocab.values():
        return star
    can = canonical_dummy_smiles(star)
    if can is None:
        return None
    return canonical_to_vocab.get(can)


def load_rrr_pairs(canonical_to_vocab: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    templates = []
    current = None
    for line in RRR_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        smiles, avg_improvement, publications = parts[0], parts[1], parts[2]
        if publications == "*":
            current = smiles
            templates.append(
                {
                    "template_rrr_smiles": smiles,
                    "old_fragment_smiles": r_to_star(smiles),
                    "old_fragment_vocab": map_to_vocab(smiles, canonical_to_vocab),
                    "attachment_signature": infer_attachment_signature(r_to_star(smiles)),
                    "external_split": split_for_template(smiles),
                }
            )
            continue
        if current is None:
            continue
        rows.append(
            {
                "template_rrr_smiles": current,
                "replacement_rrr_smiles": smiles,
                "old_fragment_smiles": r_to_star(current),
                "replacement_fragment_smiles": r_to_star(smiles),
                "old_fragment_vocab": map_to_vocab(current, canonical_to_vocab),
                "replacement_vocab": map_to_vocab(smiles, canonical_to_vocab),
                "attachment_signature": infer_attachment_signature(r_to_star(current)),
                "avg_log10_activity_improvement": float(avg_improvement),
                "publication_count": int(publications),
                "external_split": split_for_template(current),
            }
        )
    return pd.DataFrame(templates), pd.DataFrame(rows)


def curated_weight(avg_improvement: float, publication_count: int) -> float:
    return max(0.0, float(avg_improvement)) * math.log1p(float(publication_count))


def build_benchmark(pairs: pd.DataFrame, vocab_index: dict[str, int]) -> tuple[pd.DataFrame, np.ndarray, list[list[int]], pd.DataFrame]:
    eligible = pairs.loc[
        pairs["replacement_vocab"].notna()
        & (pairs["publication_count"] >= PRIMARY_MIN_PUBLICATIONS)
        & (pairs["avg_log10_activity_improvement"] >= PRIMARY_MIN_LOG_IMPROVEMENT)
    ].copy()
    eligible["replacement_idx"] = eligible["replacement_vocab"].map(vocab_index).astype(int)
    eligible["edge_weight"] = [
        curated_weight(a, n)
        for a, n in zip(eligible["avg_log10_activity_improvement"], eligible["publication_count"])
    ]
    query_rows = []
    positive_indices = []
    for template, sub in eligible.groupby("template_rrr_smiles", sort=True):
        old_fragment = str(sub["old_fragment_smiles"].iloc[0])
        signature = str(sub["attachment_signature"].iloc[0])
        split = str(sub["external_split"].iloc[0])
        pos = sorted(set(int(x) for x in sub["replacement_idx"]))
        if not pos:
            continue
        query_rows.append(
            {
                "query_id": hashlib.md5(template.encode("utf-8")).hexdigest()[:16],
                "template_rrr_smiles": template,
                "old_fragment_smiles": old_fragment,
                "attachment_signature": signature,
                "external_split": split,
                "n_curated_positive_replacements": int(len(pos)),
                "max_publication_count": int(sub["publication_count"].max()),
                "mean_log10_activity_improvement": float(sub["avg_log10_activity_improvement"].mean()),
                "old_fragment_in_routeA_vocab": int(pd.notna(sub["old_fragment_vocab"].iloc[0])),
            }
        )
        positive_indices.append(pos)
    queries = pd.DataFrame(query_rows)
    labels = np.zeros((len(queries), len(vocab_index)), dtype=np.uint8)
    for i, pos in enumerate(positive_indices):
        labels[i, pos] = 1
    return queries, labels, positive_indices, eligible


def compute_routea_scores(queries: pd.DataFrame, labels: np.ndarray) -> dict[str, np.ndarray]:
    install_sklearn_loss_pickle_compat()
    artifacts = load_baseline_artifacts()
    de_scores, de_ranks = score_de_full(queries, artifacts)
    hgb_scores, hgb_ranks = score_hgb_full(queries, artifacts)
    attach_scores, attach_ranks = full_attach_arrays(queries, artifacts)
    borda_scores, borda_ranks = full_borda(de_ranks, hgb_ranks)

    feature_cols = feature_columns_for_sets()["F3_rank_score_frequency_chemistry"]
    col_idx = {name: idx for idx, name in enumerate(feature_cols)}
    x_matrix = np.zeros((len(queries), len(artifacts.vocab), len(feature_cols)), dtype=np.float32)
    x_matrix[:, :, col_idx["rank_attach"]] = attach_ranks
    x_matrix[:, :, col_idx["rank_de"]] = de_ranks
    x_matrix[:, :, col_idx["rank_hgb"]] = hgb_ranks
    x_matrix[:, :, col_idx["rank_borda"]] = borda_ranks
    x_matrix[:, :, col_idx["score_attach"]] = attach_scores
    x_matrix[:, :, col_idx["score_de"]] = de_scores
    x_matrix[:, :, col_idx["score_hgb"]] = hgb_scores
    x_matrix[:, :, col_idx["score_borda"]] = borda_scores

    payload = load_selected_predictor_local("M5_mlp_F0")
    mlp_scores = score_selected_local(payload, x_matrix)
    s1_scores = 0.95 * row_zscore(mlp_scores) + 0.05 * row_zscore(hgb_scores)

    return {
        "Attachment_frequency": attach_scores,
        "DE": de_scores,
        "HGB": hgb_scores,
        "Borda_DE_HGB": borda_scores,
        "M5_mlp_F0": mlp_scores,
        "S1_MLP_HGB_score_blend": s1_scores,
    }


def direct_rrr_prior(queries: pd.DataFrame, eligible_pairs: pd.DataFrame, vocab_index: dict[str, int]) -> np.ndarray:
    prior = np.zeros((len(queries), len(vocab_index)), dtype=np.float32)
    grouped = defaultdict(list)
    for row in eligible_pairs.itertuples(index=False):
        grouped[str(row.template_rrr_smiles)].append(
            (int(row.replacement_idx), curated_weight(row.avg_log10_activity_improvement, row.publication_count))
        )
    for i, row in enumerate(queries.itertuples(index=False)):
        for idx, weight in grouped.get(str(row.template_rrr_smiles), []):
            prior[i, idx] = max(prior[i, idx], float(weight))
    return prior


def smoothed_rrr_prior(
    queries: pd.DataFrame,
    train_pairs: pd.DataFrame,
    train_templates: pd.DataFrame,
    vocab_index: dict[str, int],
    sim_threshold: float,
) -> np.ndarray:
    train_templates = train_templates.loc[
        train_templates["template_rrr_smiles"].isin(set(train_pairs["template_rrr_smiles"]))
    ].copy()
    template_fps = {
        str(row.template_rrr_smiles): fragment_fp(str(row.old_fragment_smiles))
        for row in train_templates.itertuples(index=False)
    }
    edge_map = defaultdict(list)
    for row in train_pairs.itertuples(index=False):
        edge_map[str(row.template_rrr_smiles)].append(
            (int(row.replacement_idx), curated_weight(row.avg_log10_activity_improvement, row.publication_count))
        )
    prior = np.zeros((len(queries), len(vocab_index)), dtype=np.float32)
    for i, row in enumerate(queries.itertuples(index=False)):
        qfp = fragment_fp(str(row.old_fragment_smiles))
        for template, tfp in template_fps.items():
            sim = tanimoto_from_arrays(qfp, tfp)
            if sim < sim_threshold:
                continue
            for cand_idx, weight in edge_map.get(template, []):
                prior[i, cand_idx] += float(sim * weight)
    return prior


def query_metric_row(method: str, split: str, scores: np.ndarray, labels: np.ndarray) -> dict:
    metrics = metrics_from_score_matrix(scores, labels)
    return {
        "method": method,
        "split": split,
        "n_queries": int(labels.shape[0]),
        "Top1": metrics["Top1"],
        "Top5": metrics["Top5"],
        "Top10": metrics["Top10"],
        "Top20": metrics["Top20"],
        "Top50": metrics["Top50"],
        "MRR": metrics["MRR"],
        "NDCG@10": metrics["NDCG@10"],
    }


def pair_metric_row(method: str, split: str, scores: np.ndarray, queries: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    if pairs.empty:
        return {
            "method": method,
            "split": split,
            "n_pairs": 0,
            "PairTop1": np.nan,
            "PairTop5": np.nan,
            "PairTop10": np.nan,
            "PairTop20": np.nan,
            "PairTop50": np.nan,
            "PairMRR": np.nan,
            "PublicationWeightedPairTop10": np.nan,
        }
    q_index = {str(row.template_rrr_smiles): i for i, row in queries.iterrows()}
    ranks = score_matrix_to_ranks(scores)
    pair_ranks = []
    weights = []
    for row in pairs.itertuples(index=False):
        template = str(row.template_rrr_smiles)
        if template not in q_index:
            continue
        pair_ranks.append(int(ranks[q_index[template], int(row.replacement_idx)]))
        weights.append(float(row.publication_count))
    arr = np.asarray(pair_ranks, dtype=float)
    w = np.asarray(weights, dtype=float)
    return {
        "method": method,
        "split": split,
        "n_pairs": int(len(arr)),
        "PairTop1": float(np.mean(arr <= 1)),
        "PairTop5": float(np.mean(arr <= 5)),
        "PairTop10": float(np.mean(arr <= 10)),
        "PairTop20": float(np.mean(arr <= 20)),
        "PairTop50": float(np.mean(arr <= 50)),
        "PairMRR": float(np.mean(1.0 / arr)),
        "PublicationWeightedPairTop10": float(np.average((arr <= 10).astype(float), weights=w)) if w.sum() else np.nan,
    }


def tune_external_adapter(
    queries: pd.DataFrame,
    labels: np.ndarray,
    eligible_pairs: pd.DataFrame,
    templates: pd.DataFrame,
    vocab_index: dict[str, int],
    base_scores: np.ndarray,
) -> tuple[dict, np.ndarray]:
    train_pairs = eligible_pairs.loc[eligible_pairs["external_split"] == "external_train"].copy()
    val_mask = queries["external_split"].eq("external_val").to_numpy()
    val_queries = queries.loc[val_mask].reset_index(drop=True)
    val_pairs = eligible_pairs.loc[eligible_pairs["external_split"] == "external_val"].copy()
    base_val_metrics = metrics_from_score_matrix(base_scores[val_mask], labels[val_mask])
    best = None
    all_rows = []
    for sim_threshold in SIM_THRESHOLD_GRID:
        prior = smoothed_rrr_prior(
            queries,
            train_pairs=train_pairs,
            train_templates=templates.loc[templates["external_split"] == "external_train"],
            vocab_index=vocab_index,
            sim_threshold=sim_threshold,
        )
        prior_z = row_zscore(prior)
        base_z = row_zscore(base_scores)
        for beta in BETA_GRID:
            scores = base_z + beta * prior_z
            metrics = metrics_from_score_matrix(scores[val_mask], labels[val_mask])
            pair_metrics = pair_metric_row("candidate", "external_val", scores[val_mask], val_queries, val_pairs)
            passes_constraints = (
                metrics["Top10"] >= base_val_metrics["Top10"]
                and metrics["Top5"] >= base_val_metrics["Top5"]
            )
            row = {
                "sim_threshold": float(sim_threshold),
                "beta": float(beta),
                "selection_objective": "max_validation_NDCG@10_with_no_Top5_Top10_loss_vs_S1",
                "passes_no_loss_constraint": int(passes_constraints),
                "validation_n_queries": int(val_mask.sum()),
                "validation_Top10": metrics["Top10"],
                "validation_MRR": metrics["MRR"],
                "validation_Top5": metrics["Top5"],
                "validation_NDCG@10": metrics["NDCG@10"],
                "validation_PairTop10": pair_metrics["PairTop10"],
                "validation_PairMRR": pair_metrics["PairMRR"],
                "validation_PublicationWeightedPairTop10": pair_metrics["PublicationWeightedPairTop10"],
            }
            all_rows.append(row)
            key = (
                int(passes_constraints),
                metrics["NDCG@10"],
                metrics["Top10"],
                metrics["MRR"],
                pair_metrics["PublicationWeightedPairTop10"],
                pair_metrics["PairTop10"],
                -beta,
            )
            if best is None or key > best[0]:
                best = (key, row, prior)
    grid = pd.DataFrame(all_rows)
    grid["selected"] = 0
    selected = best[1]
    grid.loc[
        (grid["sim_threshold"] == selected["sim_threshold"]) & (grid["beta"] == selected["beta"]),
        "selected",
    ] = 1
    grid.to_csv(OUT / "s3_curated_adapter_validation_grid.csv", index=False)
    selected_prior = best[2]
    adapted = row_zscore(base_scores) + float(selected["beta"]) * row_zscore(selected_prior)
    return selected, adapted


def bootstrap_top10_delta(a_scores: np.ndarray, b_scores: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict:
    a_best = metrics_from_score_matrix(a_scores, labels)["best_rank"]
    b_best = metrics_from_score_matrix(b_scores, labels)["best_rank"]
    a_hit = (a_best <= 10).astype(float)
    b_hit = (b_best <= 10).astype(float)
    n = len(a_hit)
    deltas = np.empty(5000, dtype=float)
    for i in range(len(deltas)):
        idx = rng.integers(0, n, size=n)
        deltas[i] = float(np.mean(a_hit[idx] - b_hit[idx]))
    return {
        "Top10_delta": float(np.mean(a_hit - b_hit)),
        "Top10_ci_low": float(np.quantile(deltas, 0.025)),
        "Top10_ci_high": float(np.quantile(deltas, 0.975)),
    }


def bootstrap_pair_delta(
    a_scores: np.ndarray,
    b_scores: np.ndarray,
    queries: pd.DataFrame,
    pairs: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    a_ranks = score_matrix_to_ranks(a_scores)
    b_ranks = score_matrix_to_ranks(b_scores)
    pair_groups = []
    for i, row in queries.iterrows():
        sub = pairs.loc[pairs["template_rrr_smiles"].eq(row["template_rrr_smiles"])]
        if sub.empty:
            continue
        idx = sub["replacement_idx"].astype(int).to_numpy()
        weights = sub["publication_count"].astype(float).to_numpy()
        pair_groups.append(
            {
                "a_hit10": (a_ranks[i, idx] <= 10).astype(float),
                "b_hit10": (b_ranks[i, idx] <= 10).astype(float),
                "a_mrr": 1.0 / a_ranks[i, idx].astype(float),
                "b_mrr": 1.0 / b_ranks[i, idx].astype(float),
                "weights": weights,
            }
        )

    def summarize(sampled_groups: list[dict]) -> tuple[float, float, float]:
        a_hit = np.concatenate([g["a_hit10"] for g in sampled_groups])
        b_hit = np.concatenate([g["b_hit10"] for g in sampled_groups])
        a_mrr = np.concatenate([g["a_mrr"] for g in sampled_groups])
        b_mrr = np.concatenate([g["b_mrr"] for g in sampled_groups])
        weights = np.concatenate([g["weights"] for g in sampled_groups])
        pair_top10_delta = float(np.mean(a_hit - b_hit))
        pair_mrr_delta = float(np.mean(a_mrr - b_mrr))
        weighted_delta = float(np.average(a_hit - b_hit, weights=weights)) if weights.sum() else np.nan
        return pair_top10_delta, weighted_delta, pair_mrr_delta

    observed = summarize(pair_groups)
    n = len(pair_groups)
    boot = np.empty((5000, 3), dtype=float)
    for i in range(len(boot)):
        idx = rng.integers(0, n, size=n)
        boot[i] = summarize([pair_groups[j] for j in idx])
    return {
        "PairTop10_delta": observed[0],
        "PairTop10_ci_low": float(np.quantile(boot[:, 0], 0.025)),
        "PairTop10_ci_high": float(np.quantile(boot[:, 0], 0.975)),
        "PublicationWeightedPairTop10_delta": observed[1],
        "PublicationWeightedPairTop10_ci_low": float(np.quantile(boot[:, 1], 0.025)),
        "PublicationWeightedPairTop10_ci_high": float(np.quantile(boot[:, 1], 0.975)),
        "PairMRR_delta": observed[2],
        "PairMRR_ci_low": float(np.quantile(boot[:, 2], 0.025)),
        "PairMRR_ci_high": float(np.quantile(boot[:, 2], 0.975)),
    }


def main() -> None:
    log("Loading vocabulary and RRR source data.")
    vocab, vocab_index, canonical_to_vocab = load_vocab_mapping()
    templates, pairs = load_rrr_pairs(canonical_to_vocab)
    queries, labels, _positive_indices, eligible_pairs = build_benchmark(pairs, vocab_index)

    source_summary = {
        "source_name": "Ertl_Ring_Replacement_Recommender_rrr_data",
        "source_path": str(RRR_PATH),
        "sklearn_runtime_version": sklearn.__version__,
        "raw_templates": int(len(templates)),
        "raw_pairs": int(len(pairs)),
        "routeA_candidate_vocab_size": int(len(vocab)),
        "pairs_with_replacement_in_routeA_vocab": int(pairs["replacement_vocab"].notna().sum()),
        "pairs_passing_primary_support_filter": int(len(eligible_pairs)),
        "templates_with_at_least_one_primary_positive": int(len(queries)),
        "primary_min_publications": PRIMARY_MIN_PUBLICATIONS,
        "primary_min_log10_activity_improvement": PRIMARY_MIN_LOG_IMPROVEMENT,
        "query_splits": queries["external_split"].value_counts().to_dict(),
    }
    (OUT / "s3_curated_source_coverage_summary.json").write_text(
        json.dumps(source_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pairs.to_csv(OUT / "s3_rrr_pair_mapping_audit.csv", index=False)
    queries.to_csv(OUT / "s3_curated_query_manifest.csv", index=False)
    eligible_pairs.to_csv(OUT / "s3_curated_pair_manifest.csv", index=False)

    log(f"Scoring {len(queries)} curated external queries.")
    scores = compute_routea_scores(queries, labels)
    direct_prior = direct_rrr_prior(queries, eligible_pairs, vocab_index)
    scores["RRR_direct_lookup_upper_bound"] = direct_prior

    selected_adapter, adapted_scores = tune_external_adapter(
        queries,
        labels,
        eligible_pairs,
        templates,
        vocab_index,
        scores["S1_MLP_HGB_score_blend"],
    )
    scores["S3_template_heldout_curated_prior_adapter"] = adapted_scores

    query_rows = []
    pair_rows = []
    for split in ["all_external", "external_val", "external_test"]:
        if split == "all_external":
            mask = np.ones(len(queries), dtype=bool)
        else:
            mask = queries["external_split"].eq(split).to_numpy()
        q_sub = queries.loc[mask].reset_index(drop=True)
        y_sub = labels[mask]
        pair_sub = eligible_pairs.loc[eligible_pairs["external_split"].eq(split.replace("all_external", ""))]
        if split == "all_external":
            pair_sub = eligible_pairs
        for method, matrix in scores.items():
            query_rows.append(query_metric_row(method, split, matrix[mask], y_sub))
            pair_rows.append(pair_metric_row(method, split, matrix[mask], q_sub, pair_sub))
    query_metrics = pd.DataFrame(query_rows)
    pair_metrics = pd.DataFrame(pair_rows)
    query_metrics.to_csv(OUT / "s3_curated_query_level_recall_metrics.csv", index=False)
    pair_metrics.to_csv(OUT / "s3_curated_pair_level_recall_metrics.csv", index=False)

    test_mask = queries["external_split"].eq("external_test").to_numpy()
    test_queries = queries.loc[test_mask].reset_index(drop=True)
    test_pairs = eligible_pairs.loc[eligible_pairs["external_split"] == "external_test"].copy()
    rng = np.random.default_rng(SEED)
    boot_rows = []
    pair_boot_rows = []
    for baseline in ["Borda_DE_HGB", "M5_mlp_F0", "S1_MLP_HGB_score_blend"]:
        row = bootstrap_top10_delta(
            scores["S3_template_heldout_curated_prior_adapter"][test_mask],
            scores[baseline][test_mask],
            labels[test_mask],
            rng,
        )
        row["comparison"] = f"S3_adapter_minus_{baseline}"
        row["split"] = "external_test"
        boot_rows.append(row)
        pair_row = bootstrap_pair_delta(
            scores["S3_template_heldout_curated_prior_adapter"][test_mask],
            scores[baseline][test_mask],
            test_queries,
            test_pairs,
            rng,
        )
        pair_row["comparison"] = f"S3_adapter_minus_{baseline}"
        pair_row["split"] = "external_test"
        pair_boot_rows.append(pair_row)
    query_bootstrap = pd.DataFrame(boot_rows)
    pair_bootstrap = pd.DataFrame(pair_boot_rows)
    query_bootstrap.to_csv(OUT / "s3_curated_external_test_bootstrap.csv", index=False)
    pair_bootstrap.to_csv(OUT / "s3_curated_external_test_pair_bootstrap.csv", index=False)

    # Per-query diagnostic ranks for the held-out test split.
    diagnostic_rows = []
    for method, matrix in scores.items():
        ranks = metrics_from_score_matrix(matrix, labels)["best_rank"]
        for i, row in queries.iterrows():
            diagnostic_rows.append(
                {
                    "query_id": row["query_id"],
                    "template_rrr_smiles": row["template_rrr_smiles"],
                    "old_fragment_smiles": row["old_fragment_smiles"],
                    "attachment_signature": row["attachment_signature"],
                    "external_split": row["external_split"],
                    "method": method,
                    "best_curated_rank": float(ranks[i]) if np.isfinite(ranks[i]) else np.nan,
                    "hit10": int(ranks[i] <= 10),
                    "n_curated_positive_replacements": int(row["n_curated_positive_replacements"]),
                    "max_publication_count": int(row["max_publication_count"]),
                }
            )
    pd.DataFrame(diagnostic_rows).to_csv(OUT / "s3_curated_query_rank_diagnostics.csv", index=False)

    q_index_all = {str(row.template_rrr_smiles): i for i, row in queries.iterrows()}
    ranks_by_method = {method: score_matrix_to_ranks(matrix) for method, matrix in scores.items()}
    pair_rank_rows = []
    rescue_rows = []
    for row in eligible_pairs.itertuples(index=False):
        template = str(row.template_rrr_smiles)
        query_idx = q_index_all.get(template)
        if query_idx is None:
            continue
        cand_idx = int(row.replacement_idx)
        rank_map = {method: int(ranks[query_idx, cand_idx]) for method, ranks in ranks_by_method.items()}
        for method, rank in rank_map.items():
            pair_rank_rows.append(
                {
                    "template_rrr_smiles": template,
                    "old_fragment_smiles": row.old_fragment_smiles,
                    "replacement_rrr_smiles": row.replacement_rrr_smiles,
                    "replacement_fragment_smiles": row.replacement_fragment_smiles,
                    "replacement_vocab": row.replacement_vocab,
                    "attachment_signature": row.attachment_signature,
                    "external_split": row.external_split,
                    "avg_log10_activity_improvement": float(row.avg_log10_activity_improvement),
                    "publication_count": int(row.publication_count),
                    "method": method,
                    "pair_rank": rank,
                    "hit10": int(rank <= 10),
                }
            )
        s1_rank = rank_map["S1_MLP_HGB_score_blend"]
        s3_rank = rank_map["S3_template_heldout_curated_prior_adapter"]
        if row.external_split == "external_test" and s1_rank > 10 and s3_rank <= 10:
            rescue_rows.append(
                {
                    "template_rrr_smiles": template,
                    "old_fragment_smiles": row.old_fragment_smiles,
                    "replacement_rrr_smiles": row.replacement_rrr_smiles,
                    "replacement_fragment_smiles": row.replacement_fragment_smiles,
                    "replacement_vocab": row.replacement_vocab,
                    "attachment_signature": row.attachment_signature,
                    "avg_log10_activity_improvement": float(row.avg_log10_activity_improvement),
                    "publication_count": int(row.publication_count),
                    "S1_rank": s1_rank,
                    "S3_adapter_rank": s3_rank,
                    "Borda_rank": rank_map["Borda_DE_HGB"],
                    "HGB_rank": rank_map["HGB"],
                    "M5_mlp_F0_rank": rank_map["M5_mlp_F0"],
                }
            )
    pd.DataFrame(pair_rank_rows).to_csv(OUT / "s3_curated_pair_rank_diagnostics.csv", index=False)
    pd.DataFrame(rescue_rows).sort_values(
        ["publication_count", "avg_log10_activity_improvement", "S3_adapter_rank"],
        ascending=[False, False, True],
    ).to_csv(OUT / "s3_external_test_s3_adapter_rescued_pairs.csv", index=False)

    def fetch_metric(df: pd.DataFrame, method: str, split: str, col: str) -> float:
        sub = df.loc[(df["method"] == method) & (df["split"] == split)]
        return float(sub[col].iloc[0])

    def fetch_bootstrap(df: pd.DataFrame, comparison: str, col: str) -> float:
        sub = df.loc[df["comparison"] == comparison]
        return float(sub[col].iloc[0])

    verdict = f"""# S3 Curated Bioisostere Recall Verdict

**Verdict:** B. CURATED_RING_RECALL_READY_WITH_COVERAGE_LIMITS

## Source and Coverage

External source: Peter Ertl et al. ring replacement recommender support file (`rrr-data.txt`).

- Raw RRR templates: {source_summary['raw_templates']}
- Raw directed replacement rows: {source_summary['raw_pairs']}
- Rows with replacement in Route-A blind vocabulary: {source_summary['pairs_with_replacement_in_routeA_vocab']}
- Rows passing primary support filter (publications >= {PRIMARY_MIN_PUBLICATIONS}, mean log10 improvement >= {PRIMARY_MIN_LOG_IMPROVEMENT}): {source_summary['pairs_passing_primary_support_filter']}
- Templates with at least one mapped curated positive: {source_summary['templates_with_at_least_one_primary_positive']}
- External query split: {source_summary['query_splits']}

## Zero-Shot External Recall

On all mapped curated RRR queries, zero-shot S1 score blend query Top10 = {fetch_metric(query_metrics, 'S1_MLP_HGB_score_blend', 'all_external', 'Top10'):.4f}; Borda query Top10 = {fetch_metric(query_metrics, 'Borda_DE_HGB', 'all_external', 'Top10'):.4f}; HGB query Top10 = {fetch_metric(query_metrics, 'HGB', 'all_external', 'Top10'):.4f}.

## Curated-Prior Adapter

The S3 adapter uses only external-train RRR templates to build a fingerprint-smoothed curated prior. Beta and similarity threshold are selected on external-val templates, then evaluated on external-test templates.

Selected adapter: similarity threshold = {selected_adapter['sim_threshold']:.2f}, beta = {selected_adapter['beta']:.2f}. Validation objective: {selected_adapter['selection_objective']}; no-loss constraint passed = {selected_adapter['passes_no_loss_constraint']}.

External-test query Top10:

- S1 score blend: Top10 {fetch_metric(query_metrics, 'S1_MLP_HGB_score_blend', 'external_test', 'Top10'):.4f}, MRR {fetch_metric(query_metrics, 'S1_MLP_HGB_score_blend', 'external_test', 'MRR'):.4f}, NDCG@10 {fetch_metric(query_metrics, 'S1_MLP_HGB_score_blend', 'external_test', 'NDCG@10'):.4f}
- S3 curated-prior adapter: Top10 {fetch_metric(query_metrics, 'S3_template_heldout_curated_prior_adapter', 'external_test', 'Top10'):.4f}, MRR {fetch_metric(query_metrics, 'S3_template_heldout_curated_prior_adapter', 'external_test', 'MRR'):.4f}, NDCG@10 {fetch_metric(query_metrics, 'S3_template_heldout_curated_prior_adapter', 'external_test', 'NDCG@10'):.4f}
- RRR direct lookup upper bound: {fetch_metric(query_metrics, 'RRR_direct_lookup_upper_bound', 'external_test', 'Top10'):.4f}

External-test curated pair recall:

- S1 score blend: PairTop10 {fetch_metric(pair_metrics, 'S1_MLP_HGB_score_blend', 'external_test', 'PairTop10'):.4f}, publication-weighted PairTop10 {fetch_metric(pair_metrics, 'S1_MLP_HGB_score_blend', 'external_test', 'PublicationWeightedPairTop10'):.4f}
- S3 curated-prior adapter: PairTop10 {fetch_metric(pair_metrics, 'S3_template_heldout_curated_prior_adapter', 'external_test', 'PairTop10'):.4f}, publication-weighted PairTop10 {fetch_metric(pair_metrics, 'S3_template_heldout_curated_prior_adapter', 'external_test', 'PublicationWeightedPairTop10'):.4f}

External-test S3 minus S1 bootstrap deltas:

- Query Top10 delta {fetch_bootstrap(query_bootstrap, 'S3_adapter_minus_S1_MLP_HGB_score_blend', 'Top10_delta'):.4f}, 95% CI [{fetch_bootstrap(query_bootstrap, 'S3_adapter_minus_S1_MLP_HGB_score_blend', 'Top10_ci_low'):.4f}, {fetch_bootstrap(query_bootstrap, 'S3_adapter_minus_S1_MLP_HGB_score_blend', 'Top10_ci_high'):.4f}]
- PairTop10 delta {fetch_bootstrap(pair_bootstrap, 'S3_adapter_minus_S1_MLP_HGB_score_blend', 'PairTop10_delta'):.4f}, 95% CI [{fetch_bootstrap(pair_bootstrap, 'S3_adapter_minus_S1_MLP_HGB_score_blend', 'PairTop10_ci_low'):.4f}, {fetch_bootstrap(pair_bootstrap, 'S3_adapter_minus_S1_MLP_HGB_score_blend', 'PairTop10_ci_high'):.4f}]
- Publication-weighted PairTop10 delta {fetch_bootstrap(pair_bootstrap, 'S3_adapter_minus_S1_MLP_HGB_score_blend', 'PublicationWeightedPairTop10_delta'):.4f}, 95% CI [{fetch_bootstrap(pair_bootstrap, 'S3_adapter_minus_S1_MLP_HGB_score_blend', 'PublicationWeightedPairTop10_ci_low'):.4f}, {fetch_bootstrap(pair_bootstrap, 'S3_adapter_minus_S1_MLP_HGB_score_blend', 'PublicationWeightedPairTop10_ci_high'):.4f}]

The direct lookup row is not an independent ML result; it is a database-augmented upper bound showing what is recoverable when the curated source itself is used as a lookup table.

## Claim Boundary

This is an external curated ring-replacement recall benchmark. It improves evidence beyond internal MMP recovery, but it is still not wet-lab validation and does not prove activity preservation for generated proposals outside the curated RRR scope.
"""
    (OUT / "S3_CURATED_BIOISOSTERE_RECALL_VERDICT.md").write_text(verdict, encoding="utf-8")
    log(f"Wrote S3 outputs to {OUT}")


if __name__ == "__main__":
    main()
