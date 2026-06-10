#!/usr/bin/env python3
"""Audit positive/negative feature and score distributions for external splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_external_validation import (
    FEATURE_COLUMNS,
    add_scores,
    ensure_features,
    make_value_heldout_split,
    read_table,
    split_ofs,
    validate_matrix,
    write_table,
)


AUDIT_COLUMNS = [
    "morgan",
    "bit_corr",
    "dHeavy",
    "dRings",
    "dMW",
    "dLogP",
    "dTPSA",
    "score_ca",
    "score_ctcr",
]


def quantile(values: pd.Series, q: float) -> float:
    if values.empty:
        return float("nan")
    return float(values.quantile(q))


def distribution_rows(scored: pd.DataFrame, seed: int) -> list[dict]:
    rows = []
    for metric in AUDIT_COLUMNS:
        for label, group in scored.groupby("label", sort=True):
            values = group[metric].astype(float)
            rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "label": int(label),
                    "count": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "min": float(values.min()),
                    "p10": quantile(values, 0.10),
                    "p25": quantile(values, 0.25),
                    "p50": quantile(values, 0.50),
                    "p75": quantile(values, 0.75),
                    "p90": quantile(values, 0.90),
                    "max": float(values.max()),
                }
            )
    return rows


def nofreq_lbc_coefficients(train_df: pd.DataFrame, seed: int) -> dict:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=20260609)
    clf.fit(x_train, np.asarray(train_df["label"]))
    row = {"seed": seed, "intercept": float(clf.intercept_[0])}
    for feature, coef in zip(FEATURE_COLUMNS, clf.coef_[0]):
        row[f"coef_{feature}"] = float(coef)
    return row


def split_and_score(df: pd.DataFrame, args: argparse.Namespace, split_seed: int) -> tuple[pd.DataFrame, dict, dict]:
    if args.protocol == "candidate_heldout":
        train_df, test_df, meta = make_value_heldout_split(
            df,
            holdout_column="candidate_smiles",
            seed=split_seed,
            train_fraction=args.train_fraction,
            exclude_test_queries_from_train=True,
        )
        scored = add_scores(train_df, test_df, args.k, tuning_holdout_column="candidate_smiles")
    elif args.protocol == "candidate_strict_heldout":
        train_df, test_df, meta = make_value_heldout_split(
            df,
            holdout_column="candidate_smiles",
            seed=split_seed,
            train_fraction=args.train_fraction,
            exclude_test_queries_from_train=True,
            strict_all_test_values=True,
        )
        scored = add_scores(train_df, test_df, args.k, tuning_holdout_column="candidate_smiles")
    elif args.protocol == "core_heldout":
        train_df, test_df, meta = make_value_heldout_split(
            df,
            holdout_column="core_key",
            seed=split_seed,
            train_fraction=args.train_fraction,
            exclude_test_queries_from_train=False,
        )
        scored = add_scores(train_df, test_df, args.k, tuning_holdout_column="core_key")
    elif args.protocol == "repeated_of_split":
        train_ofs, test_ofs = split_ofs(df, split_seed, args.train_fraction)
        train_df = df[df["old_fragment_smiles"].isin(train_ofs)].copy()
        test_df = df[df["old_fragment_smiles"].isin(test_ofs)].copy()
        meta = {
            "holdout_column": "old_fragment_smiles",
            "n_train_values": len(train_ofs),
            "n_test_values": len(test_ofs),
            "strict_all_test_values": False,
        }
        scored = add_scores(train_df, test_df, args.k)
    else:
        raise ValueError(f"Unsupported protocol: {args.protocol}")

    meta.update(
        {
            "split_seed": split_seed,
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "n_eval_queries": int(scored.loc[scored["label"] == 1, "query_id"].nunique()),
            "mean_candidates_per_query": float(scored.groupby("query_id").size().mean()),
            "median_candidates_per_query": float(scored.groupby("query_id").size().median()),
            "mean_positives_per_query": float(scored.groupby("query_id")["label"].sum().mean()),
            "score_freq_nonzero_rate": float((scored["score_freq"] > 0).mean()),
            "positive_score_freq_nonzero_rate": float((scored.loc[scored["label"] == 1, "score_freq"] > 0).mean()),
            "negative_score_freq_nonzero_rate": float((scored.loc[scored["label"] == 0, "score_freq"] > 0).mean()),
            "fullblend_alpha": float(scored["fullblend_alpha"].iloc[0]),
            "fullblend_tuning": str(scored["fullblend_tuning"].iloc[0]),
        }
    )
    return scored, meta, nofreq_lbc_coefficients(train_df, split_seed)


def mean_contrast_table(distributions: pd.DataFrame) -> pd.DataFrame:
    mean_by_label = (
        distributions.groupby(["metric", "label"], as_index=False)
        .agg(mean=("mean", "mean"), p50=("p50", "mean"), count=("count", "sum"))
        .pivot(index="metric", columns="label", values=["mean", "p50", "count"])
    )
    rows = []
    for metric in AUDIT_COLUMNS:
        pos_mean = float(mean_by_label.loc[metric, ("mean", 1)])
        neg_mean = float(mean_by_label.loc[metric, ("mean", 0)])
        pos_p50 = float(mean_by_label.loc[metric, ("p50", 1)])
        neg_p50 = float(mean_by_label.loc[metric, ("p50", 0)])
        rows.append(
            {
                "metric": metric,
                "positive_mean": pos_mean,
                "negative_mean": neg_mean,
                "positive_minus_negative_mean": pos_mean - neg_mean,
                "positive_p50": pos_p50,
                "negative_p50": neg_p50,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, float_cols: set[str]) -> list[str]:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if col in float_cols:
                cells.append(f"{float(value):.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    audit: pd.DataFrame,
    contrast: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audit_means = audit.mean(numeric_only=True).to_dict()
    lines = [
        f"# {args.dataset_name} Score Distribution Audit",
        "",
        f"Protocol: `{args.protocol}`",
        "",
        "## Split Audit",
        "",
        f"- seeds: {args.n_seeds}",
        f"- mean train rows: {audit_means.get('n_train_rows', float('nan')):.1f}",
        f"- mean test rows: {audit_means.get('n_test_rows', float('nan')):.1f}",
        f"- mean evaluated queries: {audit_means.get('n_eval_queries', float('nan')):.1f}",
        f"- mean candidates/query: {audit_means.get('mean_candidates_per_query', float('nan')):.3f}",
        f"- mean positives/query: {audit_means.get('mean_positives_per_query', float('nan')):.3f}",
        f"- score_freq nonzero rate: {audit_means.get('score_freq_nonzero_rate', float('nan')):.6f}",
        f"- positive score_freq nonzero rate: {audit_means.get('positive_score_freq_nonzero_rate', float('nan')):.6f}",
        f"- negative score_freq nonzero rate: {audit_means.get('negative_score_freq_nonzero_rate', float('nan')):.6f}",
        "",
        "## Positive vs Negative Mean Contrast",
        "",
    ]
    lines.extend(
        markdown_table(
            contrast,
            {
                "positive_mean",
                "negative_mean",
                "positive_minus_negative_mean",
                "positive_p50",
                "negative_p50",
            },
        )
    )
    coef_means = coefficients.mean(numeric_only=True)
    coef_table = pd.DataFrame(
        [{"feature": feature, "standardized_coef_mean": coef_means[f"coef_{feature}"]} for feature in FEATURE_COLUMNS]
    )
    lines.extend(["", "## No-Frequency LBC Coefficients", ""])
    lines.extend(markdown_table(coef_table, {"standardized_coef_mean"}))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--protocol",
        choices=["repeated_of_split", "candidate_heldout", "candidate_strict_heldout", "core_heldout"],
        default="candidate_heldout",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = ensure_features(validate_matrix(read_table(args.matrix), args.dataset_name))
    distribution_records = []
    audit_records = []
    coefficient_records = []
    for i in range(args.n_seeds):
        split_seed = args.seed + i
        scored, split_meta, coef_meta = split_and_score(df, args, split_seed)
        distribution_records.extend(distribution_rows(scored, i))
        split_meta["seed"] = i
        audit_records.append(split_meta)
        coef_meta["seed_index"] = i
        coefficient_records.append(coef_meta)

    distributions = pd.DataFrame(distribution_records)
    audit = pd.DataFrame(audit_records)
    coefficients = pd.DataFrame(coefficient_records)
    contrast = mean_contrast_table(distributions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dist_path = args.out_dir / f"{args.dataset_name}_score_distributions.csv"
    audit_path = args.out_dir / f"{args.dataset_name}_split_audit.csv"
    coef_path = args.out_dir / f"{args.dataset_name}_nofreq_lbc_coefficients.csv"
    contrast_path = args.out_dir / f"{args.dataset_name}_mean_contrast.csv"
    md_path = args.out_dir / f"{args.dataset_name}_score_distribution_audit.md"
    write_table(distributions, dist_path)
    write_table(audit, audit_path)
    write_table(coefficients, coef_path)
    write_table(contrast, contrast_path)
    write_markdown(md_path, args, audit, contrast, coefficients)
    print(f"Wrote {dist_path}")
    print(f"Wrote {audit_path}")
    print(f"Wrote {coef_path}")
    print(f"Wrote {contrast_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
