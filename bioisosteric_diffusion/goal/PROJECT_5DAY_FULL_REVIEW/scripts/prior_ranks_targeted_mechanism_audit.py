#!/usr/bin/env python3
"""Targeted prior-ranks mechanism audit for the JCIM revision.

This diagnostic script reuses the fixed candidate-level scorer protocol from
``jcim_major_revision_patch.py``. It does not tune hyperparameters, does not
select a new main model, and does not modify locked manuscript results.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jcim_major_revision_patch import (
    D4S27,
    GROUPS,
    PROJECT,
    all_features,
    ci,
    paired_ci,
    qmetrics,
    train_fixed_model,
)


OUT = PROJECT / "goal/PROJECT_5DAY_FULL_REVIEW/prior_ranks_mechanism_audit"
OUT.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_N = 5000

PALETTE = {
    "teal": "#087A74",
    "red": "#B31F24",
    "navy": "#0B3F99",
    "gray": "#666666",
    "soft_gray": "#7A7A7A",
    "dark": "#222222",
    "light_gray": "#D9D9D9",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def feature_list_without(drop_groups: set[str]) -> list[str]:
    feats: list[str] = []
    for group, cols in GROUPS.items():
        if group in drop_groups:
            continue
        feats.extend(cols)
    return feats


MODEL_SPECS = [
    {
        "model_id": "full_82_feature_scorer",
        "display_name": "Full 82-feature scorer",
        "dropped_groups": [],
        "audit_role": "prospective candidate-level blind result",
    },
    {
        "model_id": "drop_prior_ranks_77_feature_scorer",
        "display_name": "Drop prior ranks (77-feature scorer)",
        "dropped_groups": ["prior_ranks"],
        "audit_role": "post-audit locked prior-rank pruning",
    },
    {
        "model_id": "drop_F5_prior_scores",
        "display_name": "Drop F5 prior scores",
        "dropped_groups": ["prior_scores"],
        "audit_role": "targeted diagnostic ablation",
    },
    {
        "model_id": "drop_F6_prior_statistics",
        "display_name": "Drop F6 prior statistics",
        "dropped_groups": ["prior_rates", "prior_supports", "prior_positives", "query_stats"],
        "audit_role": "targeted diagnostic ablation",
        "definition_note": "F6 prior statistics = prior rates, supports, positive counts, and query prior statistics.",
    },
    {
        "model_id": "drop_F7_pmi_conditional_prior_contrast",
        "display_name": "Drop F7 PMI / conditional-prior contrast",
        "dropped_groups": ["pmi"],
        "audit_role": "targeted diagnostic ablation",
    },
    {
        "model_id": "drop_all_non_rank_prior_families",
        "display_name": "Drop all non-rank prior families",
        "dropped_groups": [
            "prior_scores",
            "prior_rates",
            "prior_supports",
            "prior_positives",
            "pmi",
            "query_stats",
        ],
        "audit_role": "feasibility diagnostic; keeps prior ranks while removing non-rank prior signals",
    },
]


def metric_summary(
    metrics: dict[str, pd.DataFrame],
    spec_by_id: dict[str, dict[str, object]],
) -> pd.DataFrame:
    base = metrics["full_82_feature_scorer"]
    rows = []
    for model_id, q in metrics.items():
        vals = q["Top10"].values
        top10 = float(vals.mean())
        lo, hi = ci(vals, n_boot=BOOTSTRAP_N)
        if model_id == "full_82_feature_scorer":
            delta = 0.0
            delta_lo = 0.0
            delta_hi = 0.0
        else:
            delta, delta_lo, delta_hi = paired_ci(q, base, "Top10", n_boot=BOOTSTRAP_N)
        spec = spec_by_id[model_id]
        rows.append(
            {
                "model_id": model_id,
                "display_name": spec["display_name"],
                "dropped_groups": ";".join(spec["dropped_groups"]),
                "n_feature_columns_requested": len(feature_list_without(set(spec["dropped_groups"]))),
                "n_queries": int(len(q)),
                "Top10": top10,
                "Top10_CI_lo": lo,
                "Top10_CI_hi": hi,
                "delta_vs_82_Top10": delta,
                "delta_vs_82_CI_lo": delta_lo,
                "delta_vs_82_CI_hi": delta_hi,
                "audit_role": spec["audit_role"],
                "definition_note": spec.get("definition_note", ""),
                "protocol_note": "Fixed HistGB protocol; validation training only; secondary blind diagnostic evaluation.",
            }
        )
    return pd.DataFrame(rows)


def query_context(blind: pd.DataFrame) -> pd.DataFrame:
    # Use the local train-derived prior support used by the prior scorer.
    # Do not include p0_support, which is a global constant in this matrix and
    # would erase the intended sparsity stratification.
    if "query_prior_max_support" in blind.columns:
        support = blind["query_prior_max_support"].fillna(0)
        support_definition = "Tertiles of query median query_prior_max_support; p0/global support excluded."
    elif "backoff_support" in blind.columns:
        support = blind["backoff_support"].fillna(0)
        support_definition = "Tertiles of query median backoff_support; p0/global support excluded."
    else:
        support_cols = [c for c in ["t_p4_support", "p1_support", "p3_support"] if c in blind.columns]
        support = blind[support_cols].fillna(0).max(axis=1) if support_cols else pd.Series(0, index=blind.index)
        support_definition = "Tertiles of query median max(t_p4_support, p1_support, p3_support); p0/global support excluded."
    tmp = blind[["query_id", "old_fragment_smiles", "attachment_signature", "positive_set_size"]].copy()
    tmp["candidate_prior_support_for_bin"] = support.values
    tmp["support_definition"] = support_definition
    agg = tmp.groupby("query_id", sort=False).agg(
        old_fragment_smiles=("old_fragment_smiles", "first"),
        attachment_signature=("attachment_signature", "first"),
        positive_set_size=("positive_set_size", "first"),
        support_definition=("support_definition", "first"),
        query_prior_support_median=("candidate_prior_support_for_bin", "median"),
        query_prior_support_mean=("candidate_prior_support_for_bin", "mean"),
        query_prior_support_p90=("candidate_prior_support_for_bin", lambda x: float(np.percentile(x, 90))),
        query_prior_support_max=("candidate_prior_support_for_bin", "max"),
    )
    agg = agg.reset_index()
    try:
        agg["support_bin"] = pd.qcut(
            agg["query_prior_support_median"].rank(method="first"),
            3,
            labels=["low_support", "medium_support", "high_support"],
        ).astype(str)
    except ValueError:
        values = agg["query_prior_support_median"]
        lo, hi = values.quantile([1 / 3, 2 / 3])
        agg["support_bin"] = np.select(
            [values <= lo, values <= hi],
            ["low_support", "medium_support"],
            default="high_support",
        )
    return agg


def support_bin_audit(
    q82: pd.DataFrame,
    q77: pd.DataFrame,
    qctx: pd.DataFrame,
) -> pd.DataFrame:
    merged = (
        qctx.merge(q82[["query_id", "Top10", "best_rank"]].rename(columns={"Top10": "Top10_82", "best_rank": "best_rank_82"}), on="query_id")
        .merge(q77[["query_id", "Top10", "best_rank"]].rename(columns={"Top10": "Top10_77", "best_rank": "best_rank_77"}), on="query_id")
    )
    rows = []
    order = ["low_support", "medium_support", "high_support"]
    for i, support_bin in enumerate(order, start=1):
        g = merged[merged["support_bin"] == support_bin].copy()
        if g.empty:
            continue
        q77_g = g[["query_id", "Top10_77"]].rename(columns={"Top10_77": "Top10"})
        q82_g = g[["query_id", "Top10_82"]].rename(columns={"Top10_82": "Top10"})
        delta, delta_lo, delta_hi = paired_ci(q77_g, q82_g, "Top10", n_boot=BOOTSTRAP_N)
        rescue = int(((g["Top10_82"] == 0) & (g["Top10_77"] == 1)).sum())
        lost = int(((g["Top10_82"] == 1) & (g["Top10_77"] == 0)).sum())
        rows.append(
            {
                "support_bin": support_bin,
                "display_order": i,
                "n_queries": int(len(g)),
                "query_prior_support_median_min": float(g["query_prior_support_median"].min()),
                "query_prior_support_median_median": float(g["query_prior_support_median"].median()),
                "query_prior_support_median_max": float(g["query_prior_support_median"].max()),
                "Top10_82": float(g["Top10_82"].mean()),
                "Top10_77": float(g["Top10_77"].mean()),
                "delta_77_vs_82_Top10": delta,
                "delta_77_vs_82_CI_lo": delta_lo,
                "delta_77_vs_82_CI_hi": delta_hi,
                "rescue_77_vs_82": rescue,
                "lost_77_vs_82": lost,
                "net_77_vs_82": rescue - lost,
                "support_definition": str(g["support_definition"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def old_fragment_audit(
    q82: pd.DataFrame,
    q77: pd.DataFrame,
    qsb: pd.DataFrame,
    qctx: pd.DataFrame,
) -> pd.DataFrame:
    merged = (
        qctx.merge(q82[["query_id", "Top10", "best_rank"]].rename(columns={"Top10": "Top10_82", "best_rank": "best_rank_82"}), on="query_id")
        .merge(q77[["query_id", "Top10", "best_rank"]].rename(columns={"Top10": "Top10_77", "best_rank": "best_rank_77"}), on="query_id")
        .merge(qsb[["query_id", "Top10", "best_rank"]].rename(columns={"Top10": "Top10_score_blend", "best_rank": "best_rank_score_blend"}), on="query_id")
    )
    rows = []
    for frag, g in merged.groupby("old_fragment_smiles", sort=False):
        rescue = int(((g["Top10_82"] == 0) & (g["Top10_77"] == 1)).sum())
        lost = int(((g["Top10_82"] == 1) & (g["Top10_77"] == 0)).sum())
        rows.append(
            {
                "old_fragment_smiles": frag,
                "n_queries": int(len(g)),
                "Top10_score_blend": float(g["Top10_score_blend"].mean()),
                "Top10_82": float(g["Top10_82"].mean()),
                "Top10_77": float(g["Top10_77"].mean()),
                "delta_82_vs_score_blend_Top10": float(g["Top10_82"].mean() - g["Top10_score_blend"].mean()),
                "delta_77_vs_score_blend_Top10": float(g["Top10_77"].mean() - g["Top10_score_blend"].mean()),
                "delta_77_vs_82_Top10": float(g["Top10_77"].mean() - g["Top10_82"].mean()),
                "rescue_77_vs_82": rescue,
                "lost_77_vs_82": lost,
                "net_77_vs_82": rescue - lost,
                "initial_82_negative_vs_score_blend": bool((g["Top10_82"].mean() - g["Top10_score_blend"].mean()) < 0),
                "post_audit_77_negative_vs_score_blend": bool((g["Top10_77"].mean() - g["Top10_score_blend"].mean()) < 0),
                "mean_best_rank_82": float(g["best_rank_82"].mean()),
                "mean_best_rank_77": float(g["best_rank_77"].mean()),
                "mean_best_rank_score_blend": float(g["best_rank_score_blend"].mean()),
                "median_query_prior_support": float(g["query_prior_support_median"].median()),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(["delta_77_vs_82_Top10", "n_queries"], ascending=[True, False]).reset_index(drop=True)
    out.insert(0, "fragment_id", [f"F{i:02d}" for i in range(1, len(out) + 1)])
    return out


def write_mechanism_figure(ablation: pd.DataFrame, support: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.75), gridspec_kw={"width_ratios": [1.12, 1.0]})
    fig.patch.set_facecolor("white")

    panel_a = ablation[ablation["model_id"] != "full_82_feature_scorer"].copy()
    panel_a["plot_label"] = panel_a["display_name"].replace(
        {
            "Drop prior ranks (77-feature scorer)": "Drop prior ranks",
            "Drop F5 prior scores": "Drop prior scores",
            "Drop F6 prior statistics": "Drop prior statistics",
            "Drop F7 PMI / conditional-prior contrast": "Drop PMI/contrast",
            "Drop all non-rank prior families": "Drop non-rank priors",
        }
    )
    panel_a = panel_a.iloc[::-1]
    y = np.arange(len(panel_a))
    ax = axes[0]
    colors = [PALETTE["teal"] if mid == "drop_prior_ranks_77_feature_scorer" else PALETTE["soft_gray"] for mid in panel_a["model_id"]]
    ax.axvline(0, color=PALETTE["dark"], lw=0.8)
    for yi, (_, row), color in zip(y, panel_a.iterrows(), colors):
        x = row["delta_vs_82_Top10"]
        if row["model_id"] == "drop_prior_ranks_77_feature_scorer":
            ax.hlines(yi, row["delta_vs_82_CI_lo"], row["delta_vs_82_CI_hi"], color=color, lw=1.4)
        alpha = 1.0 if row["model_id"] == "drop_prior_ranks_77_feature_scorer" else 0.78
        ax.plot(x, yi, "o", color=color, ms=4.0, alpha=alpha)
        ha = "left" if x >= 0 else "right"
        label_x = x + (0.006 if x >= 0 else -0.006)
        ax.text(label_x, yi, f"{x:+.4f}", ha=ha, va="center", fontsize=6.6, color=color, alpha=alpha)
    ax.set_yticks(y)
    ax.set_yticklabels(panel_a["plot_label"])
    ax.set_xlim(-0.085, 0.055)
    ax.set_xlabel("ΔTop-10 after dropping feature family")
    ax.set_title("A. Targeted prior-family ablation", loc="left", fontsize=8.8, fontweight="bold")
    ax.grid(axis="x", color=PALETTE["light_gray"], lw=0.45)

    ax = axes[1]
    panel_b = support.sort_values("display_order").copy()
    x = panel_b["delta_77_vs_82_Top10"].values
    y = np.arange(len(panel_b))
    xerr = np.vstack(
        [
            x - panel_b["delta_77_vs_82_CI_lo"].values,
            panel_b["delta_77_vs_82_CI_hi"].values - x,
        ]
    )
    ax.axvline(0, color=PALETTE["dark"], lw=0.8)
    ax.errorbar(x, y, xerr=xerr, fmt="o", color=PALETTE["teal"], ecolor=PALETTE["teal"], lw=1.2, ms=4.0, capsize=2)
    for (_, row), xi, yi in zip(panel_b.iterrows(), x, y):
        label_x = row["delta_77_vs_82_CI_hi"] + 0.002
        ax.text(label_x, yi, f"{xi:+.4f}", ha="left", va="center", fontsize=6.6, color=PALETTE["teal"])
    ax.set_yticks(y)
    ax.set_yticklabels(["Low support", "Medium support", "High support"])
    ax.invert_yaxis()
    ax.set_xlabel("77-feature minus 82-feature Top-10")
    ax.set_title("B. Gain by train-derived support bin", loc="left", fontsize=8.8, fontweight="bold")
    ax.grid(axis="x", color=PALETTE["light_gray"], lw=0.45)
    ax.set_xlim(-0.003, 0.064)

    fig.tight_layout(w_pad=2.55)
    stem = OUT / "figure_s_prior_ranks_mechanism"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    log("Loading development/calibration and secondary blind candidate matrices")
    val = pd.read_parquet(D4S27 / "candidate_prior_scores_val.parquet")
    blind = pd.read_parquet(D4S27 / "candidate_prior_scores_blind.parquet").copy()
    spec_by_id = {str(spec["model_id"]): spec for spec in MODEL_SPECS}

    metrics: dict[str, pd.DataFrame] = {}
    score_cols: dict[str, str] = {}

    for spec in MODEL_SPECS:
        model_id = str(spec["model_id"])
        dropped = set(spec["dropped_groups"])
        feats = all_features(drop_prior_ranks=False) if not dropped else feature_list_without(dropped)
        log(f"Training fixed diagnostic model: {model_id}")
        score_col = f"score_{model_id}"
        blind[score_col] = train_fixed_model(val, blind, feats)
        score_cols[model_id] = score_col
        metrics[model_id] = qmetrics(blind, score_col)
        log(f"Computed query metrics for {model_id}: Top10={metrics[model_id]['Top10'].mean():.6f}")
        gc.collect()

    log("Writing targeted ablation table")
    ablation = metric_summary(metrics, spec_by_id)
    ablation.to_csv(OUT / "prior_ranks_targeted_ablation.csv", index=False)

    log("Writing support-bin audit")
    qctx = query_context(blind)
    support = support_bin_audit(
        metrics["full_82_feature_scorer"],
        metrics["drop_prior_ranks_77_feature_scorer"],
        qctx,
    )
    support.to_csv(OUT / "prior_ranks_support_bin_audit.csv", index=False)

    log("Writing old-fragment block audit")
    oldfrag = old_fragment_audit(
        metrics["full_82_feature_scorer"],
        metrics["drop_prior_ranks_77_feature_scorer"],
        qmetrics(blind, "blend_score"),
        qctx,
    )
    oldfrag.to_csv(OUT / "prior_ranks_old_fragment_delta.csv", index=False)

    log("Writing mechanism figure")
    write_mechanism_figure(ablation, support)

    manifest = {
        "outputs": {
            "targeted_ablation": str(OUT / "prior_ranks_targeted_ablation.csv"),
            "support_bin_audit": str(OUT / "prior_ranks_support_bin_audit.csv"),
            "old_fragment_delta": str(OUT / "prior_ranks_old_fragment_delta.csv"),
            "mechanism_figure_svg": str(OUT / "figure_s_prior_ranks_mechanism.svg"),
            "mechanism_figure_pdf": str(OUT / "figure_s_prior_ranks_mechanism.pdf"),
            "mechanism_figure_png": str(OUT / "figure_s_prior_ranks_mechanism.png"),
        },
        "protocol": [
            "Fixed HistGB protocol copied from jcim_major_revision_patch.py.",
            "Training uses the development/calibration matrix only.",
            "No hyperparameter search or model selection is performed on secondary blind labels.",
            "All bootstraps resample query-level rows.",
            "This is a post-audit mechanism diagnostic, not a new main scorer.",
        ],
        "score_columns": score_cols,
    }
    (OUT / "prior_ranks_mechanism_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("Done")


if __name__ == "__main__":
    main()
