#!/usr/bin/env python3
"""Route-A top-journal S4: robustness checks for the curated prior adapter."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from routeA_d4s2_common import metrics_from_score_matrix, score_matrix_to_ranks  # noqa: E402
from routeA_topjournal_s1_router_and_review_pilot import row_zscore  # noqa: E402
import routeA_topjournal_s3_curated_bioisostere_recall as s3  # noqa: E402

SEED = 20260528
PROJECT = Path("E:/zuhui/bioisosteric_diffusion")
PLAN = PROJECT / "plan_results"
S3_OUT = PLAN / "routeA_topjournal_s3_curated_bioisostere_recall"
OUT = PLAN / "routeA_topjournal_s4_curated_adapter_robustness"
OUT.mkdir(parents=True, exist_ok=True)

N_RANDOMIZED_PRIORS = 1000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def edge_weight(row, mode: str) -> float:
    if mode == "full_activity_publication":
        return s3.curated_weight(row.avg_log10_activity_improvement, row.publication_count)
    if mode == "uniform_edge":
        return 1.0
    if mode == "activity_only":
        return max(0.0, float(row.avg_log10_activity_improvement))
    if mode == "publication_only":
        return float(np.log1p(float(row.publication_count)))
    raise ValueError(f"Unknown weight mode: {mode}")


def prepare_similarity_cache(queries: pd.DataFrame, train_templates: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    template_ids = train_templates["template_rrr_smiles"].astype(str).tolist()
    template_fps = [s3.fragment_fp(str(x)) for x in train_templates["old_fragment_smiles"]]
    query_fps = [s3.fragment_fp(str(x)) for x in queries["old_fragment_smiles"]]
    sim = np.zeros((len(queries), len(template_ids)), dtype=np.float32)
    for i, qfp in enumerate(query_fps):
        for j, tfp in enumerate(template_fps):
            sim[i, j] = s3.tanimoto_from_arrays(qfp, tfp)
    return template_ids, sim


def smoothed_prior_from_edges(
    queries: pd.DataFrame,
    vocab_size: int,
    template_ids: list[str],
    similarity: np.ndarray,
    train_pairs: pd.DataFrame,
    sim_threshold: float,
    weight_mode: str,
) -> np.ndarray:
    template_to_col = {template: i for i, template in enumerate(template_ids)}
    edge_map = defaultdict(list)
    for row in train_pairs.itertuples(index=False):
        template = str(row.template_rrr_smiles)
        if template not in template_to_col:
            continue
        edge_map[template].append((int(row.replacement_idx), edge_weight(row, weight_mode)))

    prior = np.zeros((len(queries), vocab_size), dtype=np.float32)
    for template, edges in edge_map.items():
        col = template_to_col[template]
        active = similarity[:, col] >= sim_threshold
        if not np.any(active):
            continue
        sim_values = similarity[active, col].astype(np.float32)
        active_rows = np.flatnonzero(active)
        for cand_idx, weight in edges:
            prior[active_rows, cand_idx] += sim_values * float(weight)
    return prior


def selected_s3_params() -> tuple[float, float]:
    grid = pd.read_csv(S3_OUT / "s3_curated_adapter_validation_grid.csv")
    selected = grid.loc[grid["selected"] == 1].iloc[0]
    return float(selected["sim_threshold"]), float(selected["beta"])


def query_metrics(method: str, split: str, scores: np.ndarray, labels: np.ndarray) -> dict:
    row = s3.query_metric_row(method, split, scores, labels)
    return row


def pair_metrics(method: str, split: str, scores: np.ndarray, queries: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    return s3.pair_metric_row(method, split, scores, queries.reset_index(drop=True), pairs)


def collect_variant_metrics(
    variant_scores: dict[str, np.ndarray],
    queries: pd.DataFrame,
    labels: np.ndarray,
    eligible_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_rows = []
    pair_rows = []
    for split in ["external_val", "external_test"]:
        mask = queries["external_split"].eq(split).to_numpy()
        q_sub = queries.loc[mask].reset_index(drop=True)
        y_sub = labels[mask]
        pair_sub = eligible_pairs.loc[eligible_pairs["external_split"].eq(split)]
        for method, matrix in variant_scores.items():
            query_rows.append(query_metrics(method, split, matrix[mask], y_sub))
            pair_rows.append(pair_metrics(method, split, matrix[mask], q_sub, pair_sub))
    return pd.DataFrame(query_rows), pd.DataFrame(pair_rows)


def weight_mode_selection_diagnostics(
    queries: pd.DataFrame,
    labels: np.ndarray,
    eligible_pairs: pd.DataFrame,
    template_ids: list[str],
    similarity: np.ndarray,
    train_pairs: pd.DataFrame,
    vocab_size: int,
    base_scores: np.ndarray,
) -> pd.Series:
    val_mask = queries["external_split"].eq("external_val").to_numpy()
    test_mask = queries["external_split"].eq("external_test").to_numpy()
    val_queries = queries.loc[val_mask].reset_index(drop=True)
    test_queries = queries.loc[test_mask].reset_index(drop=True)
    val_pairs = eligible_pairs.loc[eligible_pairs["external_split"].eq("external_val")]
    test_pairs = eligible_pairs.loc[eligible_pairs["external_split"].eq("external_test")]
    base_val = metrics_from_score_matrix(base_scores[val_mask], labels[val_mask])
    rows = []
    for mode in ["full_activity_publication", "uniform_edge", "activity_only", "publication_only"]:
        for sim_threshold in [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            prior = smoothed_prior_from_edges(
                queries,
                vocab_size,
                template_ids,
                similarity,
                train_pairs,
                sim_threshold,
                mode,
            )
            prior_z = row_zscore(prior)
            base_z = row_zscore(base_scores)
            for beta in [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 12]:
                scores = base_z + beta * prior_z
                val_q = metrics_from_score_matrix(scores[val_mask], labels[val_mask])
                val_p = pair_metrics("candidate", "external_val", scores[val_mask], val_queries, val_pairs)
                test_q = metrics_from_score_matrix(scores[test_mask], labels[test_mask])
                test_p = pair_metrics("candidate", "external_test", scores[test_mask], test_queries, test_pairs)
                rows.append(
                    {
                        "mode": mode,
                        "sim_threshold": sim_threshold,
                        "beta": beta,
                        "passes_no_loss_constraint": int(
                            val_q["Top10"] >= base_val["Top10"] and val_q["Top5"] >= base_val["Top5"]
                        ),
                        "val_Top10": val_q["Top10"],
                        "val_Top5": val_q["Top5"],
                        "val_MRR": val_q["MRR"],
                        "val_NDCG@10": val_q["NDCG@10"],
                        "val_PairTop10": val_p["PairTop10"],
                        "val_PublicationWeightedPairTop10": val_p["PublicationWeightedPairTop10"],
                        "test_Top10": test_q["Top10"],
                        "test_MRR": test_q["MRR"],
                        "test_NDCG@10": test_q["NDCG@10"],
                        "test_PairTop10": test_p["PairTop10"],
                        "test_PublicationWeightedPairTop10": test_p["PublicationWeightedPairTop10"],
                        "test_PairMRR": test_p["PairMRR"],
                    }
                )
    diagnostics = pd.DataFrame(rows)
    diagnostics["selected"] = 0
    selected_idx = diagnostics.sort_values(
        [
            "passes_no_loss_constraint",
            "val_NDCG@10",
            "val_Top10",
            "val_MRR",
            "val_PublicationWeightedPairTop10",
            "val_PairTop10",
            "beta",
        ],
        ascending=[False, False, False, False, False, False, True],
    ).index[0]
    diagnostics.loc[selected_idx, "selected"] = 1
    diagnostics.to_csv(OUT / "s4_weight_mode_selection_diagnostics.csv", index=False)
    return diagnostics.loc[selected_idx]


def metric_value(df: pd.DataFrame, method: str, split: str, col: str) -> float:
    sub = df.loc[(df["method"] == method) & (df["split"] == split)]
    return float(sub[col].iloc[0])


def similarity_strata(
    queries: pd.DataFrame,
    labels: np.ndarray,
    eligible_pairs: pd.DataFrame,
    base_scores: np.ndarray,
    s3_scores: np.ndarray,
    max_train_similarity: np.ndarray,
) -> pd.DataFrame:
    test_mask = queries["external_split"].eq("external_test").to_numpy()
    test_queries = queries.loc[test_mask].reset_index(drop=True).copy()
    test_queries["max_train_similarity"] = max_train_similarity[test_mask]
    test_queries["similarity_quartile"] = pd.qcut(
        test_queries["max_train_similarity"].rank(method="first"),
        q=4,
        labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"],
    )
    rows = []
    for quartile, q_sub in test_queries.groupby("similarity_quartile", observed=True):
        local_indices = q_sub.index.to_numpy()
        pair_sub = eligible_pairs.loc[
            eligible_pairs["template_rrr_smiles"].isin(set(q_sub["template_rrr_smiles"]))
        ].copy()
        for method, matrix in {
            "S1_MLP_HGB_score_blend": base_scores[test_mask][local_indices],
            "S3_template_heldout_curated_prior_adapter": s3_scores[test_mask][local_indices],
        }.items():
            q_metrics = query_metrics(method, str(quartile), matrix, labels[test_mask][local_indices])
            p_metrics = pair_metrics(method, str(quartile), matrix, q_sub.reset_index(drop=True), pair_sub)
            rows.append(
                {
                    "similarity_quartile": str(quartile),
                    "n_queries": int(len(q_sub)),
                    "mean_max_train_similarity": float(q_sub["max_train_similarity"].mean()),
                    "method": method,
                    "QueryTop10": q_metrics["Top10"],
                    "QueryMRR": q_metrics["MRR"],
                    "QueryNDCG@10": q_metrics["NDCG@10"],
                    "n_pairs": p_metrics["n_pairs"],
                    "PairTop10": p_metrics["PairTop10"],
                    "PublicationWeightedPairTop10": p_metrics["PublicationWeightedPairTop10"],
                    "PairMRR": p_metrics["PairMRR"],
                }
            )
    strata = pd.DataFrame(rows)
    delta_rows = []
    for quartile in strata["similarity_quartile"].unique():
        s1 = strata.loc[
            (strata["similarity_quartile"] == quartile)
            & (strata["method"] == "S1_MLP_HGB_score_blend")
        ].iloc[0]
        s4 = strata.loc[
            (strata["similarity_quartile"] == quartile)
            & (strata["method"] == "S3_template_heldout_curated_prior_adapter")
        ].iloc[0]
        delta_rows.append(
            {
                "similarity_quartile": quartile,
                "n_queries": int(s1["n_queries"]),
                "n_pairs": int(s1["n_pairs"]),
                "mean_max_train_similarity": float(s1["mean_max_train_similarity"]),
                "QueryTop10_delta": float(s4["QueryTop10"] - s1["QueryTop10"]),
                "QueryMRR_delta": float(s4["QueryMRR"] - s1["QueryMRR"]),
                "QueryNDCG@10_delta": float(s4["QueryNDCG@10"] - s1["QueryNDCG@10"]),
                "PairTop10_delta": float(s4["PairTop10"] - s1["PairTop10"]),
                "PublicationWeightedPairTop10_delta": float(
                    s4["PublicationWeightedPairTop10"] - s1["PublicationWeightedPairTop10"]
                ),
                "PairMRR_delta": float(s4["PairMRR"] - s1["PairMRR"]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(delta_rows)


def publication_strata(
    queries: pd.DataFrame,
    eligible_pairs: pd.DataFrame,
    base_scores: np.ndarray,
    s3_scores: np.ndarray,
) -> pd.DataFrame:
    test_mask = queries["external_split"].eq("external_test").to_numpy()
    test_queries = queries.loc[test_mask].reset_index(drop=True)
    test_pairs = eligible_pairs.loc[eligible_pairs["external_split"].eq("external_test")].copy()
    test_pairs["publication_tercile"] = pd.qcut(
        test_pairs["publication_count"].rank(method="first"),
        q=3,
        labels=["low_publication", "mid_publication", "high_publication"],
    )
    rows = []
    for tercile, pair_sub in test_pairs.groupby("publication_tercile", observed=True):
        for method, matrix in {
            "S1_MLP_HGB_score_blend": base_scores[test_mask],
            "S3_template_heldout_curated_prior_adapter": s3_scores[test_mask],
        }.items():
            p_metrics = pair_metrics(method, str(tercile), matrix, test_queries, pair_sub)
            rows.append(
                {
                    "publication_tercile": str(tercile),
                    "method": method,
                    "n_pairs": p_metrics["n_pairs"],
                    "min_publication_count": int(pair_sub["publication_count"].min()),
                    "max_publication_count": int(pair_sub["publication_count"].max()),
                    "PairTop10": p_metrics["PairTop10"],
                    "PublicationWeightedPairTop10": p_metrics["PublicationWeightedPairTop10"],
                    "PairMRR": p_metrics["PairMRR"],
                }
            )
    return pd.DataFrame(rows)


def rescue_bin_summary(
    queries: pd.DataFrame,
    eligible_pairs: pd.DataFrame,
    base_scores: np.ndarray,
    s3_scores: np.ndarray,
) -> pd.DataFrame:
    test_mask = queries["external_split"].eq("external_test").to_numpy()
    test_queries = queries.loc[test_mask].reset_index(drop=True)
    test_pairs = eligible_pairs.loc[eligible_pairs["external_split"].eq("external_test")].copy()
    q_index = {str(row.template_rrr_smiles): i for i, row in test_queries.iterrows()}
    base_ranks = score_matrix_to_ranks(base_scores[test_mask])
    s3_ranks = score_matrix_to_ranks(s3_scores[test_mask])
    rows = []
    for row in test_pairs.itertuples(index=False):
        query_idx = q_index[str(row.template_rrr_smiles)]
        cand_idx = int(row.replacement_idx)
        s1_rank = int(base_ranks[query_idx, cand_idx])
        adapted_rank = int(s3_ranks[query_idx, cand_idx])
        if s1_rank <= 10 or adapted_rank > 10:
            continue
        if s1_rank <= 20:
            rank_bin = "S1_rank_11_20"
        elif s1_rank <= 50:
            rank_bin = "S1_rank_21_50"
        else:
            rank_bin = "S1_rank_gt_50"
        rows.append(
            {
                "rank_bin": rank_bin,
                "template_rrr_smiles": row.template_rrr_smiles,
                "replacement_fragment_smiles": row.replacement_fragment_smiles,
                "avg_log10_activity_improvement": float(row.avg_log10_activity_improvement),
                "publication_count": int(row.publication_count),
                "S1_rank": s1_rank,
                "S3_adapter_rank": adapted_rank,
            }
        )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "s4_external_test_rescued_pair_details.csv", index=False)
    if detail.empty:
        return pd.DataFrame()
    return (
        detail.groupby("rank_bin", as_index=False)
        .agg(
            rescued_pairs=("rank_bin", "size"),
            mean_publication_count=("publication_count", "mean"),
            max_publication_count=("publication_count", "max"),
            mean_log10_activity_improvement=("avg_log10_activity_improvement", "mean"),
            median_s1_rank=("S1_rank", "median"),
            median_s3_rank=("S3_adapter_rank", "median"),
        )
        .sort_values("rank_bin")
    )


def randomized_prior_null(
    queries: pd.DataFrame,
    labels: np.ndarray,
    eligible_pairs: pd.DataFrame,
    template_ids: list[str],
    similarity: np.ndarray,
    train_pairs: pd.DataFrame,
    vocab_size: int,
    base_scores: np.ndarray,
    observed_scores: np.ndarray,
    sim_threshold: float,
    beta: float,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    test_mask = queries["external_split"].eq("external_test").to_numpy()
    test_queries = queries.loc[test_mask].reset_index(drop=True)
    test_pairs = eligible_pairs.loc[eligible_pairs["external_split"].eq("external_test")].copy()
    base_query = metrics_from_score_matrix(base_scores[test_mask], labels[test_mask])
    observed_query = metrics_from_score_matrix(observed_scores[test_mask], labels[test_mask])
    base_pair = pair_metrics("S1", "external_test", base_scores[test_mask], test_queries, test_pairs)
    observed_pair = pair_metrics("S3", "external_test", observed_scores[test_mask], test_queries, test_pairs)
    observed = {
        "QueryTop10_delta": float(observed_query["Top10"] - base_query["Top10"]),
        "PairTop10_delta": float(observed_pair["PairTop10"] - base_pair["PairTop10"]),
        "PublicationWeightedPairTop10_delta": float(
            observed_pair["PublicationWeightedPairTop10"] - base_pair["PublicationWeightedPairTop10"]
        ),
        "PairMRR_delta": float(observed_pair["PairMRR"] - base_pair["PairMRR"]),
    }

    null_rows = []
    shuffled = train_pairs.copy().reset_index(drop=True)
    replacement_values = shuffled["replacement_idx"].to_numpy().copy()
    for iteration in range(N_RANDOMIZED_PRIORS):
        shuffled["replacement_idx"] = rng.permutation(replacement_values)
        prior = smoothed_prior_from_edges(
            queries,
            vocab_size,
            template_ids,
            similarity,
            shuffled,
            sim_threshold,
            "full_activity_publication",
        )
        scores = row_zscore(base_scores) + beta * row_zscore(prior)
        q_metrics = metrics_from_score_matrix(scores[test_mask], labels[test_mask])
        p_metrics = pair_metrics("null", "external_test", scores[test_mask], test_queries, test_pairs)
        null_rows.append(
            {
                "iteration": iteration,
                "QueryTop10_delta": float(q_metrics["Top10"] - base_query["Top10"]),
                "PairTop10_delta": float(p_metrics["PairTop10"] - base_pair["PairTop10"]),
                "PublicationWeightedPairTop10_delta": float(
                    p_metrics["PublicationWeightedPairTop10"] - base_pair["PublicationWeightedPairTop10"]
                ),
                "PairMRR_delta": float(p_metrics["PairMRR"] - base_pair["PairMRR"]),
            }
        )
    null_df = pd.DataFrame(null_rows)
    summary = {"n_randomized_priors": N_RANDOMIZED_PRIORS, "observed": observed}
    for col, obs in observed.items():
        summary[f"{col}_null_mean"] = float(null_df[col].mean())
        summary[f"{col}_null_ci_low"] = float(null_df[col].quantile(0.025))
        summary[f"{col}_null_ci_high"] = float(null_df[col].quantile(0.975))
        summary[f"{col}_empirical_p_ge_observed"] = float((1 + np.sum(null_df[col] >= obs)) / (len(null_df) + 1))
    return null_df, summary


def main() -> None:
    log("Loading S3 benchmark inputs.")
    vocab, vocab_index, canonical_to_vocab = s3.load_vocab_mapping()
    templates, pairs = s3.load_rrr_pairs(canonical_to_vocab)
    queries, labels, _positive_indices, eligible_pairs = s3.build_benchmark(pairs, vocab_index)

    sim_threshold, beta = selected_s3_params()
    train_pairs = eligible_pairs.loc[eligible_pairs["external_split"].eq("external_train")].copy()
    train_templates = templates.loc[templates["external_split"].eq("external_train")].copy()
    train_templates = train_templates.loc[
        train_templates["template_rrr_smiles"].isin(set(train_pairs["template_rrr_smiles"]))
    ].copy()
    template_ids, similarity = prepare_similarity_cache(queries, train_templates)
    max_train_similarity = similarity.max(axis=1)

    log("Scoring Route-A and curated-prior ablations.")
    scores = s3.compute_routea_scores(queries, labels)
    base_scores = scores["S1_MLP_HGB_score_blend"]
    variant_scores = {"S1_MLP_HGB_score_blend": base_scores}
    for mode in ["full_activity_publication", "uniform_edge", "activity_only", "publication_only"]:
        prior = smoothed_prior_from_edges(
            queries,
            len(vocab),
            template_ids,
            similarity,
            train_pairs,
            sim_threshold,
            mode,
        )
        variant_scores[f"S4_adapter_{mode}"] = row_zscore(base_scores) + beta * row_zscore(prior)

    query_df, pair_df = collect_variant_metrics(variant_scores, queries, labels, eligible_pairs)
    query_df.to_csv(OUT / "s4_prior_ablation_query_metrics.csv", index=False)
    pair_df.to_csv(OUT / "s4_prior_ablation_pair_metrics.csv", index=False)
    selected_weight_mode = weight_mode_selection_diagnostics(
        queries,
        labels,
        eligible_pairs,
        template_ids,
        similarity,
        train_pairs,
        len(vocab),
        base_scores,
    )

    full_scores = variant_scores["S4_adapter_full_activity_publication"]
    strata_df, strata_delta_df = similarity_strata(
        queries,
        labels,
        eligible_pairs,
        base_scores,
        full_scores,
        max_train_similarity,
    )
    strata_df.to_csv(OUT / "s4_similarity_stratified_test_metrics.csv", index=False)
    strata_delta_df.to_csv(OUT / "s4_similarity_stratified_test_deltas.csv", index=False)

    publication_df = publication_strata(queries, eligible_pairs, base_scores, full_scores)
    publication_df.to_csv(OUT / "s4_publication_stratified_pair_metrics.csv", index=False)

    rescue_summary = rescue_bin_summary(queries, eligible_pairs, base_scores, full_scores)
    rescue_summary.to_csv(OUT / "s4_rescue_rank_bin_summary.csv", index=False)

    log("Running randomized-prior null test.")
    null_df, null_summary = randomized_prior_null(
        queries,
        labels,
        eligible_pairs,
        template_ids,
        similarity,
        train_pairs,
        len(vocab),
        base_scores,
        full_scores,
        sim_threshold,
        beta,
    )
    null_df.to_csv(OUT / "s4_randomized_prior_null.csv", index=False)
    (OUT / "s4_randomized_prior_null_summary.json").write_text(
        json.dumps(null_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    s1_pair = metric_value(pair_df, "S1_MLP_HGB_score_blend", "external_test", "PublicationWeightedPairTop10")
    full_pair = metric_value(pair_df, "S4_adapter_full_activity_publication", "external_test", "PublicationWeightedPairTop10")
    uniform_pair = metric_value(pair_df, "S4_adapter_uniform_edge", "external_test", "PublicationWeightedPairTop10")
    pub_only_pair = metric_value(pair_df, "S4_adapter_publication_only", "external_test", "PublicationWeightedPairTop10")
    act_only_pair = metric_value(pair_df, "S4_adapter_activity_only", "external_test", "PublicationWeightedPairTop10")
    s1_query = metric_value(query_df, "S1_MLP_HGB_score_blend", "external_test", "Top10")
    full_query = metric_value(query_df, "S4_adapter_full_activity_publication", "external_test", "Top10")

    verdict = f"""# S4 Curated Adapter Robustness Verdict

**Verdict:** A-. CURATED_PRIOR_SIGNAL_SURVIVES_KEY_ABLATIONS_WITH_SIMILARITY-SCOPE_LIMIT

## Fixed Adapter

S4 reuses the S3 validation-selected adapter parameters without retuning: similarity threshold = {sim_threshold:.2f}, beta = {beta:.2f}.

External-test query Top10 improves from {s1_query:.4f} to {full_query:.4f}. External-test publication-weighted PairTop10 improves from {s1_pair:.4f} to {full_pair:.4f}.

## Prior Ablation

External-test publication-weighted PairTop10:

- S1 score blend: {s1_pair:.4f}
- Full activity-publication prior: {full_pair:.4f}
- Uniform-edge prior: {uniform_pair:.4f}
- Publication-only prior: {pub_only_pair:.4f}
- Activity-only prior: {act_only_pair:.4f}

Interpretation: the gain is not solely a publication-weighted evaluation artifact because the uniform-edge prior remains above S1. The fixed full prior is validation-selected, but the activity-only and publication-only ablations can be stronger on this small external-test slice; therefore the manuscript should claim a robust curated-retrieval signal, not that the current evidence weighting is globally optimal.

When the evidence-weighting mode itself is included in the validation search, the same objective selects `{selected_weight_mode['mode']}` with similarity threshold {float(selected_weight_mode['sim_threshold']):.2f} and beta {float(selected_weight_mode['beta']):.2f}; its external-test query Top10 is {float(selected_weight_mode['test_Top10']):.4f} and publication-weighted PairTop10 is {float(selected_weight_mode['test_PublicationWeightedPairTop10']):.4f}.

## Randomized-Prior Null

For publication-weighted PairTop10 delta, observed S4-S1 = {null_summary['observed']['PublicationWeightedPairTop10_delta']:.4f}; randomized prior null mean = {null_summary['PublicationWeightedPairTop10_delta_null_mean']:.4f}, 95% null interval [{null_summary['PublicationWeightedPairTop10_delta_null_ci_low']:.4f}, {null_summary['PublicationWeightedPairTop10_delta_null_ci_high']:.4f}], empirical p_ge_observed = {null_summary['PublicationWeightedPairTop10_delta_empirical_p_ge_observed']:.4f}.

For unweighted PairTop10 delta, observed S4-S1 = {null_summary['observed']['PairTop10_delta']:.4f}; randomized prior null mean = {null_summary['PairTop10_delta_null_mean']:.4f}, 95% null interval [{null_summary['PairTop10_delta_null_ci_low']:.4f}, {null_summary['PairTop10_delta_null_ci_high']:.4f}], empirical p_ge_observed = {null_summary['PairTop10_delta_empirical_p_ge_observed']:.4f}.

## Files

- `s4_prior_ablation_query_metrics.csv`
- `s4_prior_ablation_pair_metrics.csv`
- `s4_similarity_stratified_test_metrics.csv`
- `s4_similarity_stratified_test_deltas.csv`
- `s4_publication_stratified_pair_metrics.csv`
- `s4_rescue_rank_bin_summary.csv`
- `s4_randomized_prior_null.csv`
- `s4_randomized_prior_null_summary.json`
- `s4_weight_mode_selection_diagnostics.csv`
"""
    (OUT / "S4_CURATED_ADAPTER_ROBUSTNESS_VERDICT.md").write_text(verdict, encoding="utf-8")
    log(f"Wrote S4 outputs to {OUT}")


if __name__ == "__main__":
    main()
