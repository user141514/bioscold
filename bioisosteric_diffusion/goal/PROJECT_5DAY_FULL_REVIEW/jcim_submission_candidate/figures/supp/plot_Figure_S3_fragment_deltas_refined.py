#!/usr/bin/env python
"""Supplementary Figure S3: per-fragment blind Top-10 deltas."""
from __future__ import print_function

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = Path("E:/zuhui/bioisosteric_diffusion")

SRC_82 = (
    PROJECT
    / "goal/A_improve/D4S28_candidate_transform_scorer/results/d4s28_blind_fixed_strata.csv"
)
SRC_77 = PROJECT / "goal/A_improve/D4S31_feature_pruning/results/d4s31_blind_strata.csv"

OUT_STEM = HERE / "Figure_S3_fragment_deltas_final_v2"
SOURCE_DATA = HERE / "Figure_S3_fragment_deltas_final_v2_source_data.csv"

RED = "#b65a52"
RED_DARK = "#8a4039"
TEAL = "#3a8782"
TEAL_DARK = "#2a6662"
GRAY = "#666666"
GRID = "#e7e7e7"


def load_source_data():
    d82 = pd.read_csv(SRC_82)
    d77 = pd.read_csv(SRC_77)

    d82 = d82.rename(
        columns={
            "fragment": "old_fragment",
            "n": "n_queries",
            "blend_Top10": "score_blend_top10",
            "scorer_Top10": "top10_82_feature",
        }
    )
    d82["delta_82_feature"] = d82["top10_82_feature"] - d82["score_blend_top10"]

    d77 = d77.rename(
        columns={
            "fragment": "old_fragment",
            "sc_Top10": "top10_77_feature",
            "delta": "delta_77_feature",
        }
    )

    merged = d82[
        [
            "old_fragment",
            "n_queries",
            "score_blend_top10",
            "top10_82_feature",
            "delta_82_feature",
        ]
    ].merge(
        d77[["old_fragment", "top10_77_feature", "delta_77_feature"]],
        on="old_fragment",
        how="inner",
    )

    if len(merged) != 19:
        raise RuntimeError("Expected 19 old-fragment strata, got %d" % len(merged))

    merged["is_82_negative"] = merged["delta_82_feature"] < 0
    merged["is_77_negative"] = merged["delta_77_feature"] < 0

    n82 = int(merged["is_82_negative"].sum())
    n77 = int(merged["is_77_negative"].sum())
    if n82 != 5:
        raise RuntimeError("Expected five 82-feature negative strata, got %d" % n82)
    if n77 != 0:
        raise RuntimeError("Expected zero 77-feature negative strata, got %d" % n77)

    merged = merged.sort_values(["delta_82_feature", "old_fragment"], ascending=[True, True]).reset_index(drop=True)
    merged["fragment_id"] = ["F%02d" % (i + 1) for i in range(len(merged))]
    merged["axis_label"] = merged.apply(
        lambda r: "%s (n=%s)" % (r["fragment_id"], "{:,}".format(int(r["n_queries"]))),
        axis=1,
    )
    return merged


def fmt_signed(x):
    return "%+.3f" % x


def setup_axis(ax, xlim, xlabel):
    ax.set_xlim(*xlim)
    ax.axvline(0, color="#7a7a7a", lw=0.65, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.45, zorder=0)
    ax.set_xlabel(xlabel, fontsize=7.0, labelpad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#555555")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0, labelsize=7.0)
    ax.tick_params(axis="x", labelsize=7.0, width=0.6, length=2.5)


def draw_panel_a(ax, df):
    neg = df[df["is_82_negative"]].copy()
    y = np.arange(len(neg))

    ax.axvspan(-1.05, 0, color="#fbf1ef", alpha=0.22, lw=0, zorder=0)
    ax.barh(
        y,
        neg["delta_82_feature"].values,
        height=0.52,
        color=RED,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(neg["axis_label"].tolist())
    ax.invert_yaxis()
    ax.set_ylim(len(neg) - 0.35, -0.65)
    setup_axis(ax, (-1.05, 0.05), "Top-10 delta vs Score Blend")
    ax.set_xticks([-1.0, -0.75, -0.50, -0.25, 0.0])
    ax.set_xticklabels(["-1.00", "-0.75", "-0.50", "-0.25", "0"])
    ax.set_title(r"$\bf{A}$   Initial 82-feature scorer: negative strata only", loc="left", fontsize=7.7, weight="bold", pad=6)
    ax.text(
        0.02,
        0.97,
        "5/19 old-fragment strata negative",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=RED_DARK,
    )

    for yi, val in enumerate(neg["delta_82_feature"].values):
        if val <= -0.20:
            x = max(val + 0.035, -1.01)
            ha = "left"
            color = "white"
        else:
            x = min(val - 0.040, -0.035)
            ha = "right"
            color = RED_DARK
        ax.text(x, yi, fmt_signed(val), ha=ha, va="center", fontsize=7.0, color=color, zorder=4)


def draw_panel_b(ax, df):
    y = np.arange(len(df))
    vals = df["delta_77_feature"].values

    for yi, val in zip(y, vals):
        ax.plot([0, val], [yi, yi], color=TEAL, lw=0.85, alpha=0.85, zorder=2)
    ax.scatter(vals, y, s=16, color=TEAL, edgecolor="white", linewidth=0.35, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(df["axis_label"].tolist())
    ax.invert_yaxis()
    ax.set_ylim(len(df) - 0.35, -0.65)
    setup_axis(ax, (-0.02, 0.10), "Top-10 delta vs Score Blend")
    ax.set_xticks([-0.02, 0.00, 0.05, 0.10])
    ax.set_xticklabels(["-0.02", "0", "+0.05", "+0.10"])
    ax.set_title(r"$\bf{B}$   Post-audit 77-feature scorer", loc="left", fontsize=7.7, weight="bold", pad=6)
    ax.text(
        0.985,
        1.000,
        "0/19 old-fragment strata negative",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color=TEAL_DARK,
    )

    label_ids = set(df.sort_values("delta_77_feature", ascending=False).head(4)["fragment_id"].tolist())
    for yi, row in df.iterrows():
        if row["fragment_id"] not in label_ids:
            continue
        ax.text(
            row["delta_77_feature"] + 0.004,
            yi,
            fmt_signed(row["delta_77_feature"]),
            ha="left",
            va="center",
            fontsize=7.0,
            color=TEAL_DARK,
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
        }
    )

    fig = plt.figure(figsize=(7.25, 4.75))
    fig.patch.set_facecolor("white")
    gs = GridSpec(
        1,
        2,
        figure=fig,
        left=0.105,
        right=0.985,
        top=0.810,
        bottom=0.145,
        width_ratios=[0.92, 1.18],
        wspace=0.42,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    fig.text(
        0.105,
        0.935,
        "Fragment-level Top-10 deltas before and after prior-rank pruning",
        ha="left",
        va="top",
        fontsize=8.5,
        weight="bold",
    )
    fig.text(
        0.105,
        0.890,
        "Point-estimate deltas are shown relative to Score Blend; full old-fragment identities are reported in Supplementary Table S5.",
        ha="left",
        va="top",
        fontsize=7.0,
        color="#777777",
    )

    draw_panel_a(ax_a, df)
    draw_panel_b(ax_b, df)

    return fig


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    df = load_source_data()

    export_df = df[
        [
            "fragment_id",
            "old_fragment",
            "n_queries",
            "score_blend_top10",
            "top10_82_feature",
            "delta_82_feature",
            "top10_77_feature",
            "delta_77_feature",
            "is_82_negative",
            "is_77_negative",
        ]
    ].copy()
    numeric_cols = [
        "score_blend_top10",
        "top10_82_feature",
        "delta_82_feature",
        "top10_77_feature",
        "delta_77_feature",
    ]
    export_df[numeric_cols] = export_df[numeric_cols].round(6)
    export_df.to_csv(SOURCE_DATA, index=False)

    fig = draw(df)
    fig.savefig(str(OUT_STEM) + ".pdf", bbox_inches="tight")
    fig.savefig(str(OUT_STEM) + ".svg", bbox_inches="tight")
    fig.savefig(str(OUT_STEM) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:")
    for suffix in [".pdf", ".svg", ".png"]:
        print("  %s%s" % (OUT_STEM, suffix))
    print("  %s" % SOURCE_DATA)


if __name__ == "__main__":
    main()
