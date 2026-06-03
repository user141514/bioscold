#!/usr/bin/env python
"""Main Figure 3 refined: secondary blind Top-10 performance.

All values are locked manuscript values supplied in the figure specification.
This script intentionally does not read experimental result files.
"""
from __future__ import print_function

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT_STEM = HERE / "Figure_3_secondary_blind_performance_refined"
DATA_CSV = HERE / "Figure_3_secondary_blind_performance_data.csv"

BASE = "#7a7a7a"
FUSION = "#516f8e"
INITIAL = "#c9874b"
POST = "#3a8782"
DIAG = "#b8b8b8"
GRID = "#e8e8e8"
TEXT = "#222222"

ROWS = [
    ["Attachment-Frequency", 0.6019, 0.5933, 0.6104, "base_ranker", "baseline", np.nan, 1],
    ["HGB", 0.7437, 0.7356, 0.7516, "base_ranker", "baseline", np.nan, 2],
    ["Dual Encoder (DE)", 0.8055, 0.7986, 0.8122, "base_ranker", "baseline", np.nan, 3],
    ["Borda(DE,HGB)", 0.8384, 0.8321, 0.8447, "fusion_baseline", "baseline", np.nan, 4],
    ["MLP (rank-only)", 0.8402, 0.8339, 0.8466, "fusion_baseline", "baseline", np.nan, 5],
    ["Score Blend (MLP + HGB)", 0.8558, 0.8495, 0.8621, "fusion_baseline", "strongest_pre_D4S_baseline", 0.0000, 6],
    ["Initial 82-feature scorer", 0.8851, 0.8796, 0.8903, "candidate_level", "prospective_candidate_level_result", 0.0293, 7],
    ["Post-audit 77-feature scorer", 0.9243, 0.9199, 0.9287, "candidate_level", "post_audit_locked_result", 0.0686, 8],
    ["Best-of-DE+HGB diagnostic", 0.8686, np.nan, np.nan, "diagnostic", "diagnostic_only", 0.0128, 9],
]

COLUMNS = [
    "method",
    "top10",
    "ci_low",
    "ci_high",
    "method_group",
    "evidence_role",
    "delta_vs_scoreblend",
    "display_order",
]

ROLE_TEXT = {
    "strongest_pre_D4S_baseline": "strongest pre-D4S baseline",
    "prospective_candidate_level_result": "prospective candidate-level result",
    "post_audit_locked_result": "post-audit locked result",
    "diagnostic_only": "diagnostic only",
}


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


def write_csv(df):
    df.to_csv(DATA_CSV, index=False, float_format="%.4f", na_rep="")


def setup_table_axis(ax, n_rows):
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.75, n_rows - 0.35)
    ax.invert_yaxis()
    ax.axis("off")


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

    fig = plt.figure(figsize=(7.15, 3.58))
    fig.patch.set_facecolor("white")
    gs = GridSpec(
        1,
        3,
        figure=fig,
        left=0.245,
        right=0.985,
        top=0.840,
        bottom=0.155,
        width_ratios=[1.00, 0.72, 0.28],
        wspace=0.055,
    )
    ax = fig.add_subplot(gs[0, 0])
    role_ax = fig.add_subplot(gs[0, 1], sharey=ax)
    delta_ax = fig.add_subplot(gs[0, 2], sharey=ax)

    n_rows = len(df)
    y = np.arange(n_rows)

    ax.set_xlim(0.58, 0.94)
    ax.set_ylim(-0.75, n_rows - 0.35)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(df["method"].tolist(), fontsize=7.0)
    ax.tick_params(axis="y", length=0)

    ax.set_xticks([0.60, 0.70, 0.80, 0.90])
    ax.set_xticklabels(["0.60", "0.70", "0.80", "0.90"], fontsize=7.0)
    ax.grid(axis="x", color=GRID, lw=0.48, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Secondary blind Top-10", fontsize=7.2, labelpad=5)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#555555")
    ax.spines["bottom"].set_linewidth(0.6)

    setup_table_axis(role_ax, n_rows)
    setup_table_axis(delta_ax, n_rows)

    separators = [2.5, 5.5, 7.5]
    for axis in [ax, role_ax, delta_ax]:
        for sep in separators:
            axis.axhline(sep, color="#dddddd", lw=0.55, zorder=0)

    for yi, (_, row) in enumerate(df.iterrows()):
        color = color_for(row)
        if row["method_group"] == "diagnostic":
            ax.scatter(
                row["top10"],
                yi,
                s=28,
                facecolors="white",
                edgecolors=DIAG,
                linewidths=0.95,
                zorder=4,
            )
        else:
            xerr = np.array([[row["top10"] - row["ci_low"]], [row["ci_high"] - row["top10"]]])
            ax.errorbar(
                row["top10"],
                yi,
                xerr=xerr,
                fmt="o",
                markersize=4.2,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.35,
                ecolor=color,
                elinewidth=0.85,
                capsize=2.1,
                capthick=0.72,
                zorder=3,
            )

        role = ROLE_TEXT.get(row["evidence_role"], "")
        if role:
            role_color = "#555555" if row["method_group"] in ["fusion_baseline", "diagnostic"] else color
            role_ax.text(0.02, yi, role, ha="left", va="center", fontsize=7.0, color=role_color)

        if not pd.isna(row["delta_vs_scoreblend"]):
            if row["method"] == "Score Blend (MLP + HGB)":
                delta_text = "ref."
                delta_color = "#666666"
            else:
                delta_text = "%+.4f" % row["delta_vs_scoreblend"]
                delta_color = DIAG if row["method_group"] == "diagnostic" else color
            delta_ax.text(0.06, yi, delta_text, ha="left", va="center", fontsize=7.0, color=delta_color)

    # Compact table headers.
    header_y = -0.58
    ax.text(0.58, header_y, "Method", ha="left", va="center", fontsize=7.0, weight="bold", color=TEXT)
    role_ax.text(0.02, header_y, "Evidence role", ha="left", va="center", fontsize=7.0, weight="bold", color=TEXT)
    delta_ax.text(0.06, header_y, "Delta vs\nScore Blend", ha="left", va="center", fontsize=7.0, weight="bold", color=TEXT, linespacing=1.0)

    fig.text(0.245, 0.945, "Secondary blind Top-10 performance", ha="left", va="top", fontsize=8.3, weight="bold")
    return fig


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    df = locked_dataframe()
    write_csv(df)
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
