#!/usr/bin/env python
"""Main Figure 3 final: secondary blind Top-10 performance.

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
OUT_STEM = HERE / "Figure_3_secondary_blind_performance_locked"
DATA_CSV = HERE / "Figure_3_secondary_blind_performance_data.csv"

BASE = "#7a7a7a"
FUSION = "#536f8a"
INITIAL = "#c9874b"
POST = "#3a8782"
DIAG = "#b8b8b8"
GRID = "#e7e7e7"
TAG_ROLE = "#b9b9b9"

ROWS = [
    ["Attachment-Frequency", "Attachment-Frequency", 0.6019, 0.5933, 0.6104, "base_ranker", "baseline", np.nan, 1],
    ["HGB", "HGB", 0.7437, 0.7356, 0.7516, "base_ranker", "baseline", np.nan, 2],
    ["Dual Encoder (DE)", "Dual Encoder", 0.8055, 0.7986, 0.8122, "base_ranker", "baseline", np.nan, 3],
    ["Borda(DE,HGB)", "Borda(DE,HGB)", 0.8384, 0.8321, 0.8447, "fusion_baseline", "baseline", np.nan, 4],
    ["MLP (rank-only)", "MLP", 0.8402, 0.8339, 0.8466, "fusion_baseline", "baseline", np.nan, 5],
    ["Score Blend (MLP + HGB)", "Score Blend", 0.8558, 0.8495, 0.8621, "fusion_baseline", "strongest_pre_D4S_baseline", 0.0000, 6],
    ["Initial 82-feature scorer", "82-feature scorer / (prospective)", 0.8851, 0.8796, 0.8903, "candidate_level", "prospective_candidate_level_result", 0.0293, 7],
    ["Post-audit 77-feature scorer", "77-feature scorer / (post-audit)", 0.9243, 0.9199, 0.9287, "candidate_level", "post_audit_locked_result", 0.0686, 8],
    ["Best-of-DE+HGB diagnostic", "Best-of-DE+HGB / (diagnostic)", 0.8686, np.nan, np.nan, "diagnostic", "diagnostic_only", 0.0128, 9],
]

COLUMNS = [
    "method",
    "display_label",
    "top10",
    "ci_low",
    "ci_high",
    "method_group",
    "evidence_role",
    "delta_vs_scoreblend",
    "display_order",
]


def locked_dataframe():
    return pd.DataFrame(ROWS, columns=COLUMNS).sort_values("display_order")


def color_for(row):
    group = row["method_group"]
    if group == "base_ranker":
        return BASE
    if group == "fusion_baseline":
        return FUSION
    if row["method"] == "Initial 82-feature scorer":
        return INITIAL
    if row["method"] == "Post-audit 77-feature scorer":
        return POST
    return DIAG


def tag_color_for(row):
    return TAG_ROLE


def draw_y_labels(ax, df):
    trans = ax.get_yaxis_transform()
    for yi, (_, row) in enumerate(df.iterrows()):
        label = row["display_label"]
        if " / " in label:
            main, tag = label.split(" / ", 1)
            ax.text(
                -0.018,
                yi - 0.10,
                main,
                transform=trans,
                ha="right",
                va="center",
                fontsize=7.0,
                color="#222222",
                clip_on=False,
            )
            ax.text(
                -0.018,
                yi + 0.16,
                tag,
                transform=trans,
                ha="right",
                va="center",
                fontsize=5.6,
                color=tag_color_for(row),
                clip_on=False,
            )
        else:
            ax.text(
                -0.018,
                yi,
                label,
                transform=trans,
                ha="right",
                va="center",
                fontsize=7.0,
                color="#222222",
                clip_on=False,
            )


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

    fig, ax = plt.subplots(figsize=(6.75, 3.92))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(df))
    ax.set_xlim(0.58, 0.94)
    ax.set_ylim(-0.65, len(df) - 0.35)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    draw_y_labels(ax, df)

    ax.set_xticks([0.60, 0.70, 0.80, 0.90])
    ax.set_xticklabels(["0.60", "0.70", "0.80", "0.90"], fontsize=7.0)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Secondary blind Top-10", fontsize=7.3, labelpad=5)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#555555")
    ax.spines["bottom"].set_linewidth(0.6)

    for sep in [2.5, 5.5, 7.5]:
        ax.axhline(sep, color="#eeeeee", lw=0.30, alpha=0.72, zorder=0)

    for yi, (_, row) in enumerate(df.iterrows()):
        color = color_for(row)
        if row["method_group"] == "diagnostic":
            ax.scatter(
                row["top10"],
                yi,
                s=24,
                facecolors="white",
                edgecolors=DIAG,
                linewidths=0.85,
                zorder=4,
            )
            continue

        xerr = np.array([[row["top10"] - row["ci_low"]], [row["ci_high"] - row["top10"]]])
        ax.errorbar(
            row["top10"],
            yi,
            xerr=xerr,
            fmt="o",
            markersize=4.0,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            ecolor=color,
            elinewidth=0.78,
            capsize=1.95,
            capthick=0.66,
            zorder=3,
        )

    fig.text(0.295, 0.940, "Secondary blind Top-10 performance", ha="left", va="top", fontsize=5.9, weight="bold")
    plt.subplots_adjust(left=0.295, right=0.975, top=0.880, bottom=0.165)
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
