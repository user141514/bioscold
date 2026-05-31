#!/usr/bin/env python3
"""JCIM major-revision evidence patch.

This script does not tune models. It recomputes fixed 82-feature and
post-audit 77-feature candidate scorers on the same secondary blind query set,
then writes reviewer-facing dataset statistics and paired bootstrap summaries.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


PROJECT = Path("E:/zuhui/bioisosteric_diffusion")
D4S27 = PROJECT / "goal/A_improve/results"
TRAIN_DIR = PROJECT / "plan_results/routeA_chembl37k_d4s2_listwise_reranker/matrices/train"
OUT = PROJECT / "goal/PROJECT_5DAY_FULL_REVIEW/jcim_major_revision"
OUT.mkdir(parents=True, exist_ok=True)

CAT = ["old_fragment_smiles", "attachment_signature", "replacement_frequency_bin"]

GROUPS = {
    "model_scores": [
        "blend_score", "mlp_z", "hgb_z", "mlp_score", "hgb_refit_score",
        "score_DE", "score_HGB", "score_Borda", "score_attach",
    ],
    "model_ranks": ["mlp_rank", "blend_rank", "rank_DE", "rank_HGB", "rank_Borda", "rank_attach"],
    "topk_flags": [
        "candidate_in_attach_top10", "candidate_in_DE_top10",
        "candidate_in_HGB_top10", "candidate_in_Borda_top10",
    ],
    "prior_scores": [
        "cont_prior_score", "backoff_logit_score", "t_p4_logit_score",
        "p1_logit_score", "p3_logit_score",
    ],
    "prior_rates": [
        "t_p4_smoothed_rate", "p1_smoothed_rate", "p3_smoothed_rate",
        "backoff_smoothed_rate",
    ],
    "prior_supports": ["t_p4_support", "p1_support", "p3_support", "p0_support", "backoff_support"],
    "prior_positives": ["t_p4_positive", "p1_positive", "p3_positive"],
    "prior_ranks": [
        "backoff_logit_rank", "cont_prior_rank", "t_p4_logit_rank",
        "p1_logit_rank", "p3_logit_rank",
    ],
    "pmi": [
        "t_pmi_p4_p1", "pmi_p3_p1", "pmi_p1_p0",
        "t_pmi_p4_p1_rank", "pmi_p3_p1_rank", "pmi_p1_p0_rank",
    ],
    "mol_props": [
        "delta_heavy_atoms", "delta_ring_count", "delta_hetero_count",
        "delta_mw", "delta_logp", "delta_tpsa", "candidate_heavy_atoms",
        "candidate_mw", "candidate_logp", "candidate_tpsa",
    ],
    "sim_freq": ["_tanimoto_similarity", "replacement_frequency", "attachment_frequency"],
    "query_stats": ["query_prior_entropy", "query_prior_margin", "query_prior_max_support"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def all_features(drop_prior_ranks: bool) -> list[str]:
    feats: list[str] = []
    for name, cols in GROUPS.items():
        if drop_prior_ranks and name == "prior_ranks":
            continue
        feats.extend(cols)
    return feats


def build_x(df: pd.DataFrame, feats: list[str], templates: dict[str, set] | None = None):
    num = [c for c in feats if c not in CAT and c in df.columns]
    parts = [df[num].fillna(0).values.astype(np.float64)]
    tmpl: dict[str, set] = {}
    for cat in CAT:
        if cat not in df.columns:
            continue
        mapped = df[cat].copy()
        if templates and cat in templates:
            known = templates[cat]
            mapped = mapped.apply(lambda x: x if x in known else "OTHER")
        else:
            known = set(mapped.unique())
        dummies = pd.get_dummies(mapped, prefix=cat)
        tcols = [f"{cat}_{v}" for v in known] + [f"{cat}_OTHER"]
        dummies = dummies.reindex(columns=tcols, fill_value=0)
        parts.append(dummies.values.astype(np.float64))
        tmpl[cat] = known
    x = np.hstack(parts)
    return x if templates is not None else (x, tmpl)


def qmetrics(df: pd.DataFrame, score_col: str, label_col: str = "label") -> pd.DataFrame:
    rows = []
    for qid, g in df.groupby("query_id", sort=False):
        gs = g.sort_values(score_col, ascending=False)
        pos = np.where(gs[label_col].values == 1)[0] + 1
        if len(pos) == 0:
            continue
        best = int(pos.min())
        rows.append({
            "query_id": qid,
            "best_rank": best,
            "Top1": int(best <= 1),
            "Top5": int(best <= 5),
            "Top10": int(best <= 10),
            "Top20": int(best <= 20),
            "Top50": int(best <= 50),
            "MRR": 1.0 / best,
        })
    return pd.DataFrame(rows)


def ci(vals: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    vals = np.asarray(vals)
    boots = np.empty(n_boot, dtype=np.float64)
    n = len(vals)
    for i in range(n_boot):
        boots[i] = vals[rng.randint(0, n, n)].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_ci(a: pd.DataFrame, b: pd.DataFrame, metric: str, n_boot: int = 5000) -> tuple[float, float, float]:
    m = a[["query_id", metric]].merge(b[["query_id", metric]], on="query_id", suffixes=("_a", "_b"))
    diff = m[f"{metric}_a"].values - m[f"{metric}_b"].values
    lo, hi = ci(diff, n_boot=n_boot)
    return float(diff.mean()), lo, hi


def train_fixed_model(val: pd.DataFrame, blind: pd.DataFrame, feats: list[str]) -> np.ndarray:
    y = val["label"].values
    x_val, templates = build_x(val, feats)
    x_blind = build_x(blind, feats, templates=templates)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_neg = min(len(neg), len(pos) * 5)
    neg_sample = np.random.RandomState(42).choice(neg, n_neg, replace=False)
    train_idx = np.concatenate([pos, neg_sample])
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.1,
        early_stopping=False,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(x_val[train_idx], y[train_idx])
    return clf.predict_proba(x_blind)[:, 1]


def dataset_stats_for_frame(df: pd.DataFrame, split_name: str) -> dict[str, object]:
    label_col = "label" if "label" in df.columns else "is_positive"
    q = df.groupby("query_id", sort=False).agg(
        candidates=("candidate_smiles", "size"),
        positives=(label_col, "sum"),
        positive_set_size=("positive_set_size", "first"),
        old_fragment=("old_fragment_smiles", "first"),
        attachment_signature=("attachment_signature", "first"),
    )
    return {
        "split": split_name,
        "candidate_rows": int(len(df)),
        "queries": int(len(q)),
        "positive_rows": int(q["positives"].sum()),
        "unique_candidate_smiles": int(df["candidate_smiles"].nunique()),
        "unique_old_fragments": int(q["old_fragment"].nunique()),
        "unique_attachment_signatures": int(q["attachment_signature"].nunique()),
        "unique_transform_identities": int(q[["old_fragment", "attachment_signature"]].drop_duplicates().shape[0]),
        "candidate_per_query_min": int(q["candidates"].min()),
        "candidate_per_query_median": float(q["candidates"].median()),
        "candidate_per_query_max": int(q["candidates"].max()),
        "positive_set_size_min": int(q["positive_set_size"].min()),
        "positive_set_size_median": float(q["positive_set_size"].median()),
        "positive_set_size_max": int(q["positive_set_size"].max()),
        "single_positive_queries": int((q["positive_set_size"] == 1).sum()),
        "multi_positive_queries": int((q["positive_set_size"] > 1).sum()),
    }


def train_dataset_stats() -> dict[str, object]:
    usecols = [
        "query_id", "candidate_smiles", "is_positive", "positive_set_size",
        "old_fragment_smiles", "attachment_signature",
    ]
    query_rows: dict[str, tuple[int, int, int, str, str]] = {}
    candidates: set[str] = set()
    total_rows = 0
    positive_rows = 0
    for path in sorted(TRAIN_DIR.glob("*.csv.gz")):
        df = pd.read_csv(path, usecols=usecols)
        total_rows += len(df)
        positive_rows += int(df["is_positive"].sum())
        candidates.update(df["candidate_smiles"].dropna().astype(str).unique())
        g = df.groupby("query_id", sort=False).agg(
            candidates=("candidate_smiles", "size"),
            positives=("is_positive", "sum"),
            positive_set_size=("positive_set_size", "first"),
            old_fragment=("old_fragment_smiles", "first"),
            attachment_signature=("attachment_signature", "first"),
        )
        for qid, row in g.iterrows():
            query_rows[str(qid)] = (
                int(row["candidates"]),
                int(row["positive_set_size"]),
                int(row["positives"]),
                str(row["old_fragment"]),
                str(row["attachment_signature"]),
            )
    qdf = pd.DataFrame.from_dict(
        query_rows,
        orient="index",
        columns=["candidates", "positive_set_size", "positives", "old_fragment", "attachment_signature"],
    )
    return {
        "split": "train",
        "candidate_rows": int(total_rows),
        "queries": int(len(qdf)),
        "positive_rows": int(positive_rows),
        "unique_candidate_smiles": int(len(candidates)),
        "unique_old_fragments": int(qdf["old_fragment"].nunique()),
        "unique_attachment_signatures": int(qdf["attachment_signature"].nunique()),
        "unique_transform_identities": int(qdf[["old_fragment", "attachment_signature"]].drop_duplicates().shape[0]),
        "candidate_per_query_min": int(qdf["candidates"].min()),
        "candidate_per_query_median": float(qdf["candidates"].median()),
        "candidate_per_query_max": int(qdf["candidates"].max()),
        "positive_set_size_min": int(qdf["positive_set_size"].min()),
        "positive_set_size_median": float(qdf["positive_set_size"].median()),
        "positive_set_size_max": int(qdf["positive_set_size"].max()),
        "single_positive_queries": int((qdf["positive_set_size"] == 1).sum()),
        "multi_positive_queries": int((qdf["positive_set_size"] > 1).sum()),
    }


def write_metric_table(metrics: dict[str, pd.DataFrame]) -> None:
    rows = []
    for name, q in metrics.items():
        row = {"method": name, "n_queries": int(len(q))}
        for metric in ["Top1", "Top5", "Top10", "Top20", "Top50", "MRR"]:
            vals = q[metric].values
            lo, hi = ci(vals)
            row[metric] = float(vals.mean())
            row[f"{metric}_CI_lo"] = lo
            row[f"{metric}_CI_hi"] = hi
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "jcim_full_secondary_metrics_with_ci.csv", index=False)


def rescue_bootstrap(model: pd.DataFrame, ref: pd.DataFrame, ref_name: str) -> dict[str, object]:
    merged = model[["query_id", "Top10"]].merge(
        ref[["query_id", "Top10"]], on="query_id", suffixes=("_model", "_ref")
    )
    rescue = ((merged["Top10_ref"] == 0) & (merged["Top10_model"] == 1)).astype(int).values
    lost = ((merged["Top10_ref"] == 1) & (merged["Top10_model"] == 0)).astype(int).values
    net = rescue - lost
    ratio = (rescue.sum() / lost.sum()) if lost.sum() else np.inf
    net_lo, net_hi = ci(net)
    rng = np.random.RandomState(42)
    ratios = []
    n = len(merged)
    for _ in range(5000):
        idx = rng.randint(0, n, n)
        r = rescue[idx].sum()
        l = lost[idx].sum()
        ratios.append(r / l if l else np.inf)
    ratio_lo, ratio_hi = np.percentile(np.array(ratios), [2.5, 97.5])
    return {
        "reference": ref_name,
        "n_queries": int(n),
        "rescue": int(rescue.sum()),
        "lost": int(lost.sum()),
        "net": int(net.sum()),
        "net_per_query": float(net.mean()),
        "net_per_query_CI_lo": float(net_lo),
        "net_per_query_CI_hi": float(net_hi),
        "rescue_lost_ratio": float(ratio),
        "rescue_lost_ratio_CI_lo": float(ratio_lo),
        "rescue_lost_ratio_CI_hi": float(ratio_hi),
    }


def main() -> None:
    log("Loading candidate matrices")
    val = pd.read_parquet(D4S27 / "candidate_prior_scores_val.parquet")
    blind = pd.read_parquet(D4S27 / "candidate_prior_scores_blind.parquet")

    log("Writing dataset statistics")
    stats = [train_dataset_stats(), dataset_stats_for_frame(val, "development"), dataset_stats_for_frame(blind, "secondary_blind")]
    pd.DataFrame(stats).to_csv(OUT / "jcim_dataset_statistics.csv", index=False)

    log("Training fixed 82-feature and 77-feature models")
    blind = blind.copy()
    blind["score_candidate_82"] = train_fixed_model(val, blind, all_features(drop_prior_ranks=False))
    blind["score_candidate_77"] = train_fixed_model(val, blind, all_features(drop_prior_ranks=True))

    log("Computing query metrics")
    metrics = {
        "Attachment-Frequency": qmetrics(blind, "score_attach"),
        "DE": qmetrics(blind, "score_DE"),
        "HGB": qmetrics(blind, "score_HGB"),
        "Borda(DE,HGB)": qmetrics(blind, "score_Borda"),
        "Score Blend": qmetrics(blind, "blend_score"),
        "Initial 82-feature scorer": qmetrics(blind, "score_candidate_82"),
        "Post-audit 77-feature scorer": qmetrics(blind, "score_candidate_77"),
    }
    q_de = metrics["DE"]
    q_hgb = metrics["HGB"]
    q_best = q_de[["query_id"]].copy()
    q_best["Top10"] = ((q_de["Top10"].values == 1) | (q_hgb["Top10"].values == 1)).astype(int)
    for metric in ["Top1", "Top5", "Top20", "Top50"]:
        q_best[metric] = np.nan
    q_best["MRR"] = np.nan
    metrics["Best-of-DE+HGB diagnostic"] = q_best
    write_metric_table({k: v for k, v in metrics.items() if k != "Best-of-DE+HGB diagnostic"})

    log("Writing paired bootstrap summaries")
    ab_delta, ab_lo, ab_hi = paired_ci(
        metrics["Post-audit 77-feature scorer"],
        metrics["Initial 82-feature scorer"],
        "Top10",
    )
    sb_delta, sb_lo, sb_hi = paired_ci(metrics["Post-audit 77-feature scorer"], metrics["Score Blend"], "Top10")
    rows = [
        {
            "comparison": "Post-audit 77-feature scorer vs Initial 82-feature scorer",
            "metric": "Top10",
            "delta": ab_delta,
            "CI_lo": ab_lo,
            "CI_hi": ab_hi,
        },
        {
            "comparison": "Post-audit 77-feature scorer vs Score Blend",
            "metric": "Top10",
            "delta": sb_delta,
            "CI_lo": sb_lo,
            "CI_hi": sb_hi,
        },
    ]
    pd.DataFrame(rows).to_csv(OUT / "jcim_paired_bootstrap_key_deltas.csv", index=False)
    pd.DataFrame([
        rescue_bootstrap(metrics["Post-audit 77-feature scorer"], metrics["Score Blend"], "Score Blend"),
        rescue_bootstrap(metrics["Post-audit 77-feature scorer"], metrics["Initial 82-feature scorer"], "Initial 82-feature scorer"),
    ]).to_csv(OUT / "jcim_rescue_lost_bootstrap.csv", index=False)

    log("Writing query-level audit")
    q = metrics["Post-audit 77-feature scorer"][["query_id", "Top10", "best_rank"]].rename(
        columns={"Top10": "Top10_77", "best_rank": "best_rank_77"}
    )
    q = q.merge(
        metrics["Initial 82-feature scorer"][["query_id", "Top10", "best_rank"]].rename(
            columns={"Top10": "Top10_82", "best_rank": "best_rank_82"}
        ),
        on="query_id",
    )
    q = q.merge(
        metrics["Score Blend"][["query_id", "Top10", "best_rank"]].rename(
            columns={"Top10": "Top10_score_blend", "best_rank": "best_rank_score_blend"}
        ),
        on="query_id",
    )
    q.to_csv(OUT / "jcim_query_level_82_77_scoreblend_audit.csv", index=False)

    summary = {
        "outputs": {
            "dataset_statistics": str(OUT / "jcim_dataset_statistics.csv"),
            "full_secondary_metrics": str(OUT / "jcim_full_secondary_metrics_with_ci.csv"),
            "paired_bootstrap_key_deltas": str(OUT / "jcim_paired_bootstrap_key_deltas.csv"),
            "rescue_lost_bootstrap": str(OUT / "jcim_rescue_lost_bootstrap.csv"),
            "query_level_audit": str(OUT / "jcim_query_level_82_77_scoreblend_audit.csv"),
        },
        "notes": [
            "Fixed model definitions reused from D4S31 scripts; no hyperparameter tuning performed.",
            "The 77-feature scorer remains post-audit because prior_ranks removal was prompted by blind diagnostics.",
            "Bootstrap resamples query_id-level rows, not candidate rows.",
        ],
    }
    (OUT / "jcim_major_revision_evidence_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("Done")


if __name__ == "__main__":
    main()
