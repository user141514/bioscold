#!/usr/bin/env python
"""Main Figure 3: secondary blind Top-10 performance.

All values are locked manuscript values supplied in the figure specification.
This script intentionally does not read experimental result files.
"""
from __future__ import print_function

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT_STEM = HERE / "Figure_3_secondary_blind_performance"
DATA_CSV = HERE / "Figure_3_secondary_blind_performance_data.csv"

BASE = "#7a7a7a"
FUSION = "#4f79a8"
INITIAL = "#c9874b"
POST = "#3a8782"
DIAG = "#b7b7b7"
GRID = "#e7e7e7"


LOCKED_ROWS = [
    {
        "method": "Attachment-Frequency",
        "top10": 0.6019,
        "ci_low": 0.5933,
        "ci_high": 0.6104,
        "method_group": "base_ranker",
        "evidence_role": "baseline",
        "display_order": 1,
        "annotation": "",
    },
    {
        "method": "HGB",
        "top10": 0.7437,
        "ci_low": 0.7356,
        "ci_high": 0.7516,
        "method_group": "base_ranker",
        "evidence_role": "baseline",
        "display_order": 2,
        "annotation": "",
    },
    {
        "method": "Dual Encoder (DE)",
        "top10": 0.8055,
        "ci_low": 0.7986,
        "ci_high": 0.8122,
        "method_group": "base_ranker",
        "evidence_role": "baseline",
        "display_order": 3,
        "annotation": "",
    },
    {
        "method": "Borda(DE,HGB)",
        "top10": 0.8384,
        "ci_low": 0.8321,
        "ci_high": 0.8447,
        "method_group": "fusion_baseline",
        "evidence_role": "baseline",
        "display_order": 4,
        "annotation": "",
    },
    {
        "method": "MLP (rank-only)",
        "top10": 0.8402,
        "ci_low": 0.8339,
        "ci_high": 0.8466,
        "method_group": "fusion_baseline",
        "evidence_role": "baseline",
        "display_order": 5,
        "annotation": "",
    },
    {
        "method": "Score Blend (MLP + HGB)",
        "top10": 0.8558,
        "ci_low": 0.8495,
        "ci_high": 0.8621,
        "method_group": "fusion_baseline",
        "evidence_role": "strongest_pre_D4S_baseline",
        "display_order": 6,
        "annotation": "",
    },
    {
        "method": "Initial 82-feature scorer",
        "top10": 0.8851,
        "ci_low": 0.8796,
        "ci_high": 0.8903,
        "method_group": "candidate_level",
        "evidence_role": "prospective_candidate_level_result",
        "display_order": 7,
        "annotation": "prospective candidate-level result",
    },
    {
        "method": "Post-audit 77-feature scorer",
        "top10": 0.9243,
        "ci_low": 0.9199,
        "ci_high": 0.9287,
        "method_group": "candidate_level",
        "evidence_role": "post_audit_locked_result",
        "display_order": 8,
        "annotation": "post-audit feature-pruned result",
    },
    {
        "method": "Best-of-DE+HGB diagnostic",
        "top10": 0.8686,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "method_group": "diagnostic",
        "evidence_role": "diagnostic_only",
        "display_order": 9,
        "annotation": "",
    },
]

CSV_COLUMNS = [
    "method",
    "top10",
    "ci_low",
    "ci_high",
    "method_group",
    "evidence_role",
    "display_order",
    "annotation",
]


def locked_dataframe():
    df = pd.DataFrame(LOCKED_ROWS)[CSV_COLUMNS].sort_values("display_order")
    return df


def color_for_group(group, method):
    if group == "base_ranker":
        return BASE
    if group == "fusion_baseline":
        return FUSION
    if method == "Initial 82-feature scorer":
        return INITIAL
    if method == "Post-audit 77-feature scorer":
        return POST
    return DIAG


def draw(df):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.0,
        }
    )

    fig, ax = plt.subplots(figsize=(4.70, 3.78))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(df))
    ax.set_xlim(0.58, 0.955)
    ax.set_ylim(-0.65, len(df) - 0.35)
    ax.invert_yaxis()

    ax.set_yticks(y)
    ax.set_yticklabels(df["method"].tolist(), fontsize=7.0)
    ax.tick_params(axis="y", length=0)

    ax.set_xticks([0.60, 0.70, 0.80, 0.90])
    ax.set_xticklabels(["0.60", "0.70", "0.80", "0.90"], fontsize=7.0)
    ax.grid(axis="x", color=GRID, lw=0.50, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Secondary blind Top-10", fontsize=7.2, labelpad=5)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#555555")
    ax.spines["bottom"].set_linewidth(0.6)

    for i, row in df.iterrows():
        yi = int(row["display_order"]) - 1
        color = color_for_group(row["method_group"], row["method"])
        if row["method_group"] == "diagnostic":
            ax.scatter(
                row["top10"],
                yi,
                s=26,
                facecolors="white",
                edgecolors=DIAG,
                linewidths=0.95,
                zorder=4,
            )
            continue

        xerr = np.array([[row["top10"] - row["ci_low"]], [row["ci_high"] - row["top10"]]])
        ax.errorbar(
            row["top10"],
            yi,
            xerr=xerr,
            fmt="o",
            markersize=4.4,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            ecolor=color,
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.75,
            zorder=3,
        )

    # Minimal evidence-role annotations for the two candidate-level rows.
    ax.annotate(
        "prospective\ncandidate-level result",
        xy=(0.8851, 6),
        xytext=(0.895, 5.63),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=7.0,
        color="#7b562b",
        arrowprops=dict(arrowstyle="-", lw=0.45, color="#b68a5b", shrinkA=2, shrinkB=2),
    )
    ax.annotate(
        "post-audit\nfeature-pruned result",
        xy=(0.9243, 7),
        xytext=(0.913, 7.46),
        textcoords="data",
        ha="right",
        va="center",
        fontsize=7.0,
        color=POST,
        arrowprops=dict(arrowstyle="-", lw=0.45, color=POST, shrinkA=2, shrinkB=2),
    )

    fig.text(0.34, 0.955, "Secondary blind Top-10 performance", ha="left", va="top", fontsize=8.7, weight="bold")
    fig.text(
        0.34,
        0.913,
        "Points show Top-10 accuracy; error bars show 95% bootstrap confidence intervals.",
        ha="left",
        va="top",
        fontsize=7.0,
        color="#666666",
    )

    plt.subplots_adjust(left=0.34, right=0.975, top=0.845, bottom=0.145)
    return fig


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    df = locked_dataframe()
    df.to_csv(DATA_CSV, index=False, float_format="%.4f", na_rep="")

    fig = draw(df)
    fig.savefig(str(OUT_STEM) + ".pdf", bbox_inches="tight")
    fig.savefig(str(OUT_STEM) + ".svg", bbox_inches="tight")
    fig.savefig(str(OUT_STEM) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:")
    for suffix in [".pdf", ".svg", ".png"]:
        print("  %s%s" % (OUT_STEM, suffix))
    print("  %s" % DATA_CSV)


if __name__ == "__main__":
    main()
