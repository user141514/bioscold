#!/usr/bin/env python
"""Generate Figure S4: categorical schema alignment audit.

Panel layout:
  top row    : bug path
  middle row : repair path
  bottom row : repair-effect bar chart

The goal is to retain the audit-flow reading path while moving the numerical
comparison out of the crowded right side of the schematic.
"""
from __future__ import print_function

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT_STEM = HERE / "Figure_S4_categorical_schema_alignment_audit_v4_flow_bottombar"
SOURCE_DATA = HERE / "Figure_S4_categorical_schema_alignment_audit_v4_flow_bottombar_source_data.csv"

TOP10_NAIVE = 0.7120
TOP10_FROZEN = 0.8851
DELTA = TOP10_FROZEN - TOP10_NAIVE

RED = "#c65a4a"
RED_DARK = "#8f3c32"
RED_LIGHT = "#fbebe8"
GREEN = "#2f8f8a"
GREEN_DARK = "#1f6966"
GREEN_LIGHT = "#e9f5f3"
GREY = "#4c4c4c"
LIGHT_GREY = "#f2f2f2"


def add_arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.9,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def add_box(ax, x, y, w, h, text, edge, fill, fontsize=6.8):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=0.95))
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold",
        color="#222222",
        linespacing=1.15,
    )


def draw_one_row_table(ax, x, y, w, h, title, row_label, col_labels, values, active_cols, color):
    """Draw a compact one-row categorical schema table."""
    n_cols = len(col_labels)
    label_w = w * 0.21
    header_h = h * 0.43
    row_h = h - header_h
    col_w = (w - label_w) / float(n_cols)

    ax.text(x, y + h + 0.013, title, ha="left", va="bottom", fontsize=6.2, weight="bold")
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=GREY, linewidth=0.65))
    ax.add_patch(Rectangle((x, y + row_h), w, header_h, facecolor=LIGHT_GREY, edgecolor="none"))
    ax.plot([x, x + w], [y + row_h, y + row_h], color=GREY, lw=0.55)
    ax.plot([x + label_w, x + label_w], [y, y + h], color=GREY, lw=0.55)

    for j in range(1, n_cols):
        xx = x + label_w + j * col_w
        ax.plot([xx, xx], [y, y + h], color="#d6d6d6", lw=0.42)

    ax.text(x + label_w * 0.50, y + row_h * 0.50, row_label, ha="center", va="center", fontsize=5.7)

    for j, col in enumerate(col_labels):
        ax.text(
            x + label_w + (j + 0.5) * col_w,
            y + row_h + header_h * 0.50,
            col,
            ha="center",
            va="center",
            fontsize=5.35,
            color="#222222",
        )
        if j in active_cols:
            ax.add_patch(
                Rectangle(
                    (x + label_w + j * col_w + 0.006, y + 0.007),
                    col_w - 0.012,
                    row_h - 0.014,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.95,
                )
            )
            value_color = "white"
        else:
            ax.add_patch(
                Rectangle(
                    (x + label_w + j * col_w + 0.006, y + 0.007),
                    col_w - 0.012,
                    row_h - 0.014,
                    facecolor="white",
                    edgecolor="#d0d0d0",
                    linewidth=0.35,
                )
            )
            value_color = "#333333"
        ax.text(
            x + label_w + (j + 0.5) * col_w,
            y + row_h * 0.50,
            str(values[j]),
            ha="center",
            va="center",
            fontsize=5.8,
            color=value_color,
        )


def draw_main_schematic(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.040, 0.965, "Categorical schema alignment audit", ha="left", va="top", fontsize=8.7, weight="bold")
    ax.text(
        0.040,
        0.930,
        "Independent one-hot columns can shift semantic meaning across splits; frozen templates preserve column identity.",
        ha="left",
        va="top",
        fontsize=6.4,
        color="#555555",
    )

    # Bug path.
    ax.text(0.045, 0.790, "Bug path", ha="left", va="center", fontsize=7.2, weight="bold", color=RED_DARK)
    add_box(ax, 0.060, 0.665, 0.165, 0.098, "Naive one-hot\nencoding", RED, RED_LIGHT, fontsize=6.9)
    add_arrow(ax, 0.225, 0.715, 0.282, 0.760, RED_DARK)

    draw_one_row_table(
        ax,
        0.285,
        0.740,
        0.255,
        0.090,
        "Validation columns",
        "val",
        ["old=A", "old=B", "att=x"],
        [1, 0, 1],
        active_cols={0, 2},
        color="#d39a63",
    )
    draw_one_row_table(
        ax,
        0.285,
        0.595,
        0.255,
        0.090,
        "Blind columns",
        "blind",
        ["old=D", "old=A", "att=x"],
        [1, 0, 1],
        active_cols={0, 2},
        color=RED,
    )
    ax.text(0.410, 0.560, "same column position, different category", ha="center", va="center", fontsize=5.8, color=RED_DARK)

    add_arrow(ax, 0.540, 0.785, 0.610, 0.720, RED_DARK)
    add_arrow(ax, 0.540, 0.640, 0.610, 0.705, RED_DARK)
    add_box(ax, 0.620, 0.665, 0.175, 0.098, "Mismatched blind\ncolumn semantics", RED, RED_LIGHT, fontsize=6.6)
    add_arrow(ax, 0.795, 0.715, 0.835, 0.715, RED_DARK)
    add_box(ax, 0.845, 0.665, 0.105, 0.098, "Invalid\nprediction", RED, RED_LIGHT, fontsize=6.3)

    # Repair path.
    ax.text(0.045, 0.445, "Repair path", ha="left", va="center", fontsize=7.2, weight="bold", color=GREEN_DARK)
    add_box(ax, 0.060, 0.318, 0.165, 0.098, "Frozen categorical\nschema", GREEN, GREEN_LIGHT, fontsize=6.8)
    add_arrow(ax, 0.225, 0.368, 0.282, 0.415, GREEN_DARK)

    draw_one_row_table(
        ax,
        0.285,
        0.395,
        0.305,
        0.090,
        "Frozen template",
        "val",
        ["old=A", "old=B", "att=x", "OTHER"],
        [1, 0, 1, 0],
        active_cols={0, 2},
        color=GREEN,
    )
    draw_one_row_table(
        ax,
        0.285,
        0.250,
        0.305,
        0.090,
        "Blind encoded with template",
        "blind",
        ["old=A", "old=B", "att=x", "OTHER"],
        [0, 0, 1, 1],
        active_cols={2, 3},
        color=GREEN,
    )
    ax.text(0.438, 0.215, "fixed column order before blind prediction", ha="center", va="center", fontsize=5.8, color=GREEN_DARK)

    add_arrow(ax, 0.590, 0.440, 0.635, 0.370, GREEN_DARK)
    add_arrow(ax, 0.590, 0.295, 0.635, 0.355, GREEN_DARK)
    add_box(ax, 0.645, 0.318, 0.180, 0.098, "Unseen categories\nmapped to OTHER", GREEN, GREEN_LIGHT, fontsize=6.5)
    add_arrow(ax, 0.825, 0.368, 0.850, 0.368, GREEN_DARK)
    add_box(ax, 0.860, 0.318, 0.090, 0.098, "Valid\nprediction", GREEN, GREEN_LIGHT, fontsize=6.3)


def draw_bottom_bar(ax):
    vals = [TOP10_NAIVE, TOP10_FROZEN]
    labels = ["Naive schema", "Frozen schema"]
    colors = [RED, GREEN]
    ypos = [1, 0]

    bars = ax.barh(ypos, vals, color=colors, height=0.36, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0.60, 0.92)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=6.4)
    ax.set_xlabel("Blind Top-10", fontsize=6.8)
    ax.set_title("Repair effect", loc="left", fontsize=7.3, weight="bold")
    ax.grid(axis="x", color="#e7e7e7", lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(axis="x", labelsize=5.9, width=0.5, length=2.5)
    ax.tick_params(axis="y", width=0, length=0, pad=2)

    for bar, val in zip(bars, vals):
        ax.text(
            val + 0.005,
            bar.get_y() + bar.get_height() / 2.0,
            "%.4f" % val,
            ha="left",
            va="center",
            fontsize=6.2,
            weight="bold",
        )

    ax.text(
        0.785,
        -0.37,
        "Delta = +%.4f" % DELTA,
        ha="center",
        va="center",
        fontsize=6.4,
        weight="bold",
        color=GREEN_DARK,
    )


def draw_figure():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
        }
    )

    fig = plt.figure(figsize=(7.25, 5.15))
    fig.patch.set_facecolor("white")

    ax_main = fig.add_axes([0.00, 0.24, 1.00, 0.76])
    draw_main_schematic(ax_main)

    ax_bar = fig.add_axes([0.175, 0.075, 0.680, 0.135])
    draw_bottom_bar(ax_bar)

    return fig


def write_source_data():
    df = pd.DataFrame(
        [
            {
                "condition": "naive_independent_one_hot",
                "schema_control": "none",
                "blind_column_status": "mismatched",
                "prediction_status": "invalid",
                "blind_top10": TOP10_NAIVE,
            },
            {
                "condition": "frozen_categorical_schema",
                "schema_control": "OTHER_plus_fixed_column_order",
                "blind_column_status": "aligned",
                "prediction_status": "valid",
                "blind_top10": TOP10_FROZEN,
            },
        ]
    )
    df["delta_vs_naive"] = df["blind_top10"] - TOP10_NAIVE
    df.to_csv(SOURCE_DATA, index=False, float_format="%.4f")


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    write_source_data()
    fig = draw_figure()
    fig.savefig(str(OUT_STEM) + ".svg", bbox_inches="tight")
    fig.savefig(str(OUT_STEM) + ".pdf", bbox_inches="tight")
    fig.savefig(str(OUT_STEM) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(OUT_STEM) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:")
    for suffix in [".svg", ".pdf", ".png", ".tiff"]:
        print("  %s%s" % (OUT_STEM, suffix))
    print("  %s" % SOURCE_DATA)


if __name__ == "__main__":
    main()
