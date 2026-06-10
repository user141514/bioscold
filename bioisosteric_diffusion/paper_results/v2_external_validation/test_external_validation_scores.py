from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_external_validation import (
    add_scores,
    choose_fullblend_alpha,
    make_value_heldout_split,
    query_hit_values,
    ranking_metrics,
    summarize_scored,
)


def _row(qid, old_fragment, candidate, label, freq_like, content_like):
    return {
        "dataset": "diagnostic_v2_test",
        "query_id": qid,
        "old_fragment_smiles": old_fragment,
        "candidate_smiles": candidate,
        "label": label,
        "morgan": content_like,
        "bit_corr": content_like * 0.8,
        "dHeavy": 1.0 - content_like,
        "dRings": 0.0,
        "dMW": 1.0 - content_like,
        "dLogP": 1.0 - content_like,
        "dTPSA": 1.0 - content_like,
        "freq_like": freq_like,
    }


def test_diagnostic_v2_scores_include_fullblend_and_lbc_nofreq():
    train = pd.DataFrame(
        [
            _row("tr_a", "OF_A", "C_GLOBAL", 1, 1.0, 0.20),
            _row("tr_a", "OF_A", "C_NEAR_A", 0, 0.0, 0.95),
            _row("tr_b", "OF_B", "C_GLOBAL", 1, 1.0, 0.25),
            _row("tr_b", "OF_B", "C_NEAR_B", 0, 0.0, 0.90),
            _row("tr_c", "OF_C", "C_LOCAL", 1, 0.2, 0.90),
            _row("tr_c", "OF_C", "C_FAR", 0, 0.0, 0.10),
        ]
    )
    test = pd.DataFrame(
        [
            _row("te_a", "OF_D", "C_GLOBAL", 1, 1.0, 0.25),
            _row("te_a", "OF_D", "C_NEAR_D", 0, 0.0, 0.95),
            _row("te_b", "OF_E", "C_LOCAL", 1, 0.2, 0.90),
            _row("te_b", "OF_E", "C_FAR", 0, 0.0, 0.10),
        ]
    )

    scored = add_scores(train, test, k=1)
    summary, detail = summarize_scored(scored, k=1, seed=0)

    assert "score_fullblend" in scored.columns
    assert "score_fullcontent" in scored.columns
    assert "score_cf_c3" in scored.columns
    assert "score_ctcr" in scored.columns
    assert "score_lbc_nofreq" in scored.columns
    assert "fullcontent_hit@1" in detail.columns
    assert "fullcontent_mrr" in detail.columns
    assert "fullcontent_ndcg@10" in detail.columns
    assert "cf_c3_hit@1" in detail.columns
    assert "cf_c3_mrr" in detail.columns
    assert "cf_c3_ndcg@10" in detail.columns
    assert "fullblend_hit@1" in detail.columns
    assert "fullblend_hit@5" in detail.columns
    assert "fullblend_mrr" in detail.columns
    assert "fullblend_ndcg@10" in detail.columns
    assert "fullblend_hit@10_random_norm" in detail.columns
    assert "random_hit@10_expectation" in detail.columns
    assert "ctcr_hit@1" in detail.columns
    assert "lbc_nofreq_hit@1" in detail.columns
    assert "fullblend_query_hit@1" in summary
    assert "fullcontent_query_hit@1" in summary
    assert "fullcontent_query_mrr" in summary
    assert "fullcontent_of_macro_ndcg@10" in summary
    assert "cf_c3_query_hit@1" in summary
    assert "cf_c3_query_mrr" in summary
    assert "cf_c3_of_macro_ndcg@10" in summary
    assert "cf_c3_query_hit@10_random_norm" in summary
    assert "fullblend_query_hit@5" in summary
    assert "fullblend_query_mrr" in summary
    assert "fullblend_query_ndcg@10" in summary
    assert "fullblend_query_hit@10_random_norm" in summary
    assert "random_query_hit@10_expectation" in summary
    assert "ctcr_query_hit@1" in summary
    assert "lbc_nofreq_query_hit@1" in summary
    assert summary["fullblend_alpha"] in {0.0, 0.25, 0.5, 0.75, 1.0}
    assert summary["fullblend_tuning"] == "old_fragment_smiles"
    assert summary["cf_c3_C"] in {0.03, 0.1, 0.3, 1.0}
    assert summary["cf_c3_training_rows"] > 0
    assert summary["cf_c3_tuning"] == "old_fragment_smiles"
    assert scored["fullblend_tuning"].iloc[0] == "old_fragment_smiles"


def test_tied_scores_use_stable_candidate_tie_breaker():
    frame = pd.DataFrame(
        [
            _row("q_tie", "OF_T", "C_POS", 1, 0.0, 0.5),
            _row("q_tie", "OF_T", "C_NEG_A", 0, 0.0, 0.5),
            _row("q_tie", "OF_T", "C_NEG_B", 0, 0.0, 0.5),
        ]
    )
    scores = pd.Series([0.0, 0.0, 0.0], index=frame.index)

    reversed_frame = frame.iloc[::-1].reset_index(drop=True)
    reversed_scores = pd.Series([0.0, 0.0, 0.0], index=reversed_frame.index)

    assert query_hit_values(frame, scores, k=1) == query_hit_values(reversed_frame, reversed_scores, k=1)


def test_ranking_metrics_include_mrr_ndcg_and_random_normalization():
    labels = pd.Series([0, 1, 0, 1])
    scores = pd.Series([0.9, 0.8, 0.7, 0.6])
    metrics = ranking_metrics(
        labels.to_numpy(),
        scores.to_numpy(),
        k_values=[1, 5, 10],
        tie_values=pd.Series([0, 1, 2, 3]).to_numpy(),
    )

    assert metrics["hit@1"] == 0
    assert metrics["hit@5"] == 1
    assert metrics["hit@10"] == 1
    assert metrics["mrr"] == 0.5
    assert round(metrics["ndcg@10"], 6) == 0.650921
    assert metrics["random_hit@10_expectation"] == 1.0
    assert pd.isna(metrics["hit@10_random_norm"])
    assert metrics["hit@10_enrichment"] == 1.0


def test_value_heldout_split_removes_positive_support_overlap():
    rows = []
    for idx, (candidate, core_key) in enumerate(
        [
            ("C_A", "core_1"),
            ("C_B", "core_2"),
            ("C_C", "core_3"),
            ("C_D", "core_4"),
        ]
    ):
        qid = f"q_{idx}"
        rows.append(_row(qid, f"OF_{idx}", candidate, 1, 1.0, 0.8))
        rows[-1]["core_key"] = core_key
        rows.append(_row(qid, f"OF_{idx}", f"NEG_{idx}", 0, 0.0, 0.2))
        rows[-1]["core_key"] = core_key
    df = pd.DataFrame(rows)

    train_df, test_df, meta = make_value_heldout_split(
        df,
        holdout_column="candidate_smiles",
        seed=7,
        train_fraction=0.5,
        exclude_test_queries_from_train=True,
    )

    train_positive_candidates = set(train_df.loc[train_df["label"] == 1, "candidate_smiles"])
    test_positive_candidates = set(test_df.loc[test_df["label"] == 1, "candidate_smiles"])
    assert train_positive_candidates
    assert test_positive_candidates
    assert train_positive_candidates.isdisjoint(test_positive_candidates)
    assert set(train_df["query_id"]).isdisjoint(set(test_df["query_id"]))
    assert meta["test_positive_value_overlap_train_rows"] == 0
    assert meta["holdout_column"] == "candidate_smiles"


def test_core_heldout_split_removes_core_overlap():
    rows = []
    for idx, core_key in enumerate(["core_1", "core_2", "core_3", "core_4"]):
        qid = f"q_{idx}"
        rows.append(_row(qid, f"OF_{idx}", f"C_{idx}", 1, 1.0, 0.8))
        rows[-1]["core_key"] = core_key
        rows.append(_row(qid, f"OF_{idx}", f"NEG_{idx}", 0, 0.0, 0.2))
        rows[-1]["core_key"] = core_key
    df = pd.DataFrame(rows)

    train_df, test_df, meta = make_value_heldout_split(
        df,
        holdout_column="core_key",
        seed=11,
        train_fraction=0.5,
        exclude_test_queries_from_train=False,
    )

    assert set(train_df["core_key"]).isdisjoint(set(test_df["core_key"]))
    assert train_df["label"].nunique() == 2
    assert test_df["label"].sum() > 0
    assert meta["holdout_column"] == "core_key"


def test_candidate_cluster_heldout_removes_positive_cluster_overlap():
    rows = []
    for idx, (candidate, cluster) in enumerate(
        [
            ("C_A1", "cluster_A"),
            ("C_A2", "cluster_A"),
            ("C_B1", "cluster_B"),
            ("C_C1", "cluster_C"),
            ("C_D1", "cluster_D"),
        ]
    ):
        qid = f"q_{idx}"
        rows.append(_row(qid, f"OF_{idx}", candidate, 1, 1.0, 0.8))
        rows[-1]["core_key"] = f"core_{idx}"
        rows[-1]["candidate_cluster"] = cluster
        rows.append(_row(qid, f"OF_{idx}", f"NEG_{idx}", 0, 0.0, 0.2))
        rows[-1]["core_key"] = f"core_{idx}"
        rows[-1]["candidate_cluster"] = f"neg_cluster_{idx % 2}"
    df = pd.DataFrame(rows)

    train_df, test_df, meta = make_value_heldout_split(
        df,
        holdout_column="candidate_cluster",
        seed=19,
        train_fraction=0.5,
        exclude_test_queries_from_train=True,
    )

    train_positive_clusters = set(train_df.loc[train_df["label"] == 1, "candidate_cluster"])
    test_positive_clusters = set(test_df.loc[test_df["label"] == 1, "candidate_cluster"])
    assert train_positive_clusters
    assert test_positive_clusters
    assert train_positive_clusters.isdisjoint(test_positive_clusters)
    assert set(train_df["query_id"]).isdisjoint(set(test_df["query_id"]))
    assert meta["test_positive_cluster_overlap_train_rows"] == 0
    assert meta["holdout_column"] == "candidate_cluster"


def test_strict_candidate_heldout_removes_all_test_candidates_from_train():
    rows = []
    for idx, candidate in enumerate(["C_A", "C_B", "C_C", "C_D"]):
        qid = f"q_{idx}"
        rows.append(_row(qid, f"OF_{idx}", candidate, 1, 1.0, 0.8))
        rows[-1]["core_key"] = f"core_{idx}"
        shared_negative = "NEG_SHARED_A" if idx < 2 else "NEG_SHARED_B"
        rows.append(_row(qid, f"OF_{idx}", shared_negative, 0, 0.0, 0.2))
        rows[-1]["core_key"] = f"core_{idx}"
        rows.append(_row(qid, f"OF_{idx}", f"NEG_UNIQUE_{idx}", 0, 0.0, 0.1))
        rows[-1]["core_key"] = f"core_{idx}"
    df = pd.DataFrame(rows)

    train_df, test_df, meta = make_value_heldout_split(
        df,
        holdout_column="candidate_smiles",
        seed=7,
        train_fraction=0.5,
        exclude_test_queries_from_train=True,
        strict_all_test_values=True,
    )

    assert set(train_df["candidate_smiles"]).isdisjoint(set(test_df["candidate_smiles"]))
    assert meta["test_positive_value_overlap_train_rows"] == 0
    assert meta["test_any_value_overlap_train_rows"] == 0
    assert meta["strict_all_test_values"] is True


def test_strict_candidate_heldout_partitions_shared_negative_pool():
    rows = []
    positive_candidates = ["C_A", "C_B", "C_C", "C_D", "C_E", "C_F"]
    shared_negatives = ["NEG_SHARED_A", "NEG_SHARED_B", "NEG_SHARED_C", "NEG_SHARED_D"]
    for idx, candidate in enumerate(positive_candidates):
        qid = f"q_{idx}"
        rows.append(_row(qid, f"OF_{idx}", candidate, 1, 1.0, 0.8))
        rows[-1]["core_key"] = f"core_{idx}"
        for neg_idx, negative in enumerate(shared_negatives):
            rows.append(_row(qid, f"OF_{idx}", negative, 0, 0.0, 0.2 - 0.02 * neg_idx))
            rows[-1]["core_key"] = f"core_neg_{neg_idx}"
    df = pd.DataFrame(rows)

    train_df, test_df, meta = make_value_heldout_split(
        df,
        holdout_column="candidate_smiles",
        seed=13,
        train_fraction=0.5,
        exclude_test_queries_from_train=True,
        strict_all_test_values=True,
    )

    assert set(train_df["candidate_smiles"]).isdisjoint(set(test_df["candidate_smiles"]))
    assert train_df["label"].nunique() == 2
    assert test_df["label"].nunique() == 2
    assert meta["test_positive_value_overlap_train_rows"] == 0
    assert meta["test_any_value_overlap_train_rows"] == 0


def test_fullblend_inner_tuning_can_match_candidate_heldout():
    rows = []
    for idx, candidate in enumerate(["C_MEM_A", "C_MEM_B", "C_TRANSFER_A", "C_TRANSFER_B"]):
        qid = f"q_{idx}"
        content = 0.9
        rows.append(_row(qid, f"OF_{idx}", f"NEG_{idx}", 0, 0.0, 0.1))
        rows[-1]["core_key"] = f"core_{idx}"
        rows.append(_row(qid, f"OF_{idx}", candidate, 1, 1.0, content))
        rows[-1]["core_key"] = f"core_{idx}"
    df = pd.DataFrame(rows)

    of_alpha = choose_fullblend_alpha(df, k=1, tuning_holdout_column=None)
    candidate_alpha = choose_fullblend_alpha(df, k=1, tuning_holdout_column="candidate_smiles")

    assert of_alpha in {0.0, 0.25, 0.5, 0.75, 1.0}
    assert candidate_alpha == 0.0
