#!/usr/bin/env python
"""Supplementary Figure S4: categorical schema alignment audit."""
from __future__ import print_function

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT_STEM = HERE / "Figure_S4_categorical_schema_alignment_audit_final"
SOURCE_DATA = HERE / "Figure_S4_categorical_schema_alignment_audit_final_source_data.csv"

TOP10_NAIVE = 0.7120
TOP10_FROZEN = 0.8851
DELTA_TOP10 = TOP10_FROZEN - TOP10_NAIVE

BUG = "#ad5f55"
BUG_DARK = "#7f463f"
REPAIR = "#3a8782"
REPAIR_DARK = "#2a6662"
GRAY = "#4b4b4b"
LIGHT_GRAY = "#f1f1f1"


def add_arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def add_box(ax, x, y, w, h, text, edge, fontsize=7.2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=edge, linewidth=0.8))
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold",
        color="#222222",
        linespacing=1.12,
    )


def column_center(x, w, n_cols, col_idx):
    label_w = w * 0.25
    col_w = (w - label_w) / float(n_cols)
    return x + label_w + (col_idx + 0.5) * col_w


def add_column_outline(ax, x, y, w, h, n_cols, col_idx, color):
    label_w = w * 0.25
    col_w = (w - label_w) / float(n_cols)
    rx = x + label_w + col_idx * col_w + 0.003
    ax.add_patch(
        Rectangle(
            (rx, y + 0.004),
            col_w - 0.006,
            h - 0.008,
            facecolor="none",
            edgecolor=color,
            linewidth=0.65,
            linestyle=(0, (2.0, 2.0)),
        )
    )


def add_table(ax, x, y, w, h, title, columns, row_label, values, active_cols, accent, ghost_cols=None):
    ghost_cols = ghost_cols or set()
    n_cols = len(columns)
    label_w = w * 0.25
    header_h = h * 0.45
    row_h = h - header_h
    col_w = (w - label_w) / float(n_cols)

    ax.text(x, y + h + 0.009, title, ha="left", va="bottom", fontsize=7.0, weight="bold", color="#222222")
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=GRAY, linewidth=0.65))
    ax.add_patch(Rectangle((x, y + row_h), w, header_h, facecolor=LIGHT_GRAY, edgecolor="none"))
    ax.plot([x, x + w], [y + row_h, y + row_h], color=GRAY, lw=0.55)
    ax.plot([x + label_w, x + label_w], [y, y + h], color=GRAY, lw=0.55)
    for j in range(1, n_cols):
        xx = x + label_w + j * col_w
        ax.plot([xx, xx], [y, y + h], color="#d4d4d4", lw=0.42)

    for j, col in enumerate(columns):
        header_color = "#b5b5b5" if j in ghost_cols else "#222222"
        ax.text(
            x + label_w + (j + 0.5) * col_w,
            y + row_h + header_h / 2.0,
            col,
            ha="center",
            va="center",
            fontsize=7.0,
            color=header_color,
        )
        cell_x = x + label_w + j * col_w + 0.006
        cell_y = y + 0.007
        cell_w = col_w - 0.012
        cell_h = row_h - 0.014
        if j in ghost_cols:
            fc = "#fdfdfd"
            tc = "#b5b5b5"
            ec = "#eeeeee"
        elif j in active_cols:
            fc = accent
            tc = "white"
            ec = accent
        else:
            fc = "white"
            tc = "#333333"
            ec = "#d2d2d2"
        ax.add_patch(Rectangle((cell_x, cell_y), cell_w, cell_h, facecolor=fc, edgecolor=ec, linewidth=0.38))
        if values[j] != "":
            ax.text(
                x + label_w + (j + 0.5) * col_w,
                y + row_h / 2.0,
                str(values[j]),
                ha="center",
                va="center",
                fontsize=7.0,
                color=tc,
            )

    ax.text(x + label_w / 2.0, y + row_h / 2.0, row_label, ha="center", va="center", fontsize=7.0, color="#222222")


def add_top10_inset(ax):
    ax.text(
        0.880,
        0.176,
        "Audit-only Top-10 check",
        ha="left",
        va="top",
        fontsize=7.0,
        color="#333333",
        weight="bold",
    )
    x0, y0 = 0.880, 0.046
    vals = [TOP10_NAIVE, TOP10_FROZEN]
    labels = ["Naive", "Frozen"]
    colors = [BUG, REPAIR]
    scale_min, scale_max = 0.60, 0.90

    for i, (val, label, color) in enumerate(zip(vals, labels, colors)):
        yy = y0 + 0.055 - i * 0.037
        ax.text(x0, yy, label, ha="left", va="center", fontsize=7.0, color="#333333")
        bar_x = x0 + 0.050
        bar_w = 0.045 * (val - scale_min) / (scale_max - scale_min)
        ax.add_patch(Rectangle((bar_x, yy - 0.0045), bar_w, 0.009, facecolor=color, edgecolor="none", alpha=0.70))
        ax.text(bar_x + bar_w + 0.005, yy, "%.4f" % val, ha="left", va="center", fontsize=7.0, weight="bold")

    ax.text(
        x0 + 0.050,
        y0 - 0.002,
        u"\u0394Top-10 = +%.4f" % DELTA_TOP10,
        ha="left",
        va="top",
        fontsize=7.0,
        color=REPAIR_DARK,
    )
    ax.text(
        x0,
        y0 - 0.046,
        "implementation repair effect;\nnot a model contribution",
        ha="left",
        va="top",
        fontsize=7.0,
        color="#555555",
        linespacing=1.05,
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

    fig = plt.figure(figsize=(7.70, 4.55))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.045, 0.955, "Categorical schema alignment audit", ha="left", va="top", fontsize=9.0, weight="bold")
    ax.text(
        0.045,
        0.918,
        "Independent one-hot encoding can shift column meaning; frozen schemas preserve feature semantics.",
        ha="left",
        va="top",
        fontsize=7.2,
        color="#555555",
    )

    schema_x = 0.250
    schema_w = 0.300
    table_h = 0.086
    start_x = 0.055
    start_w = 0.126
    start_h = 0.070
    mid_x = 0.595
    mid_w = 0.120
    end_x = 0.745
    end_w = 0.125
    box_h = 0.088

    top_dev_y = 0.735
    top_blind_y = 0.610
    bottom_dev_y = 0.390
    bottom_blind_y = 0.265
    top_center_y = (top_blind_y + top_dev_y + table_h) / 2.0
    bottom_center_y = (bottom_blind_y + bottom_dev_y + table_h) / 2.0

    # Top row: bug path.
    ax.text(start_x, top_center_y + 0.078, "bug path", ha="left", va="center", fontsize=7.2, weight="bold", color=BUG_DARK)
    add_box(ax, start_x, top_center_y - start_h / 2.0, start_w, start_h, "Naive independent\nencoding", BUG, fontsize=7.0)
    add_arrow(ax, start_x + start_w, top_center_y, schema_x - 0.008, top_center_y, BUG_DARK)

    add_table(
        ax,
        schema_x,
        top_dev_y,
        schema_w,
        table_h,
        "Development/calibration encoder",
        ["old=A", "old=B", "att=x", "OTHER"],
        "dev.",
        [1, 0, 1, ""],
        {0, 2},
        BUG,
        ghost_cols={3},
    )
    add_table(
        ax,
        schema_x,
        top_blind_y,
        schema_w,
        table_h,
        "Secondary blind encoder",
        ["old=D", "old=A", "att=x", "OTHER"],
        "blind",
        [1, 0, 1, ""],
        {0, 2},
        BUG,
        ghost_cols={3},
    )
    add_column_outline(ax, schema_x, top_dev_y, schema_w, table_h, 4, 0, BUG_DARK)
    add_column_outline(ax, schema_x, top_blind_y, schema_w, table_h, 4, 0, BUG_DARK)
    ax.text(schema_x + schema_w / 2.0, top_blind_y - 0.032, "column 1 changes meaning", ha="center", va="center", fontsize=7.0, color=BUG_DARK)

    add_arrow(ax, schema_x + schema_w, top_dev_y + table_h / 2.0, mid_x - 0.008, top_center_y, BUG_DARK)
    add_arrow(ax, schema_x + schema_w, top_blind_y + table_h / 2.0, mid_x - 0.008, top_center_y, BUG_DARK)
    add_box(ax, mid_x, top_center_y - box_h / 2.0, mid_w, box_h, "column-semantic\nmismatch", BUG, fontsize=7.0)
    add_arrow(ax, mid_x + mid_w, top_center_y, end_x - 0.008, top_center_y, BUG_DARK)
    add_box(ax, end_x, top_center_y - box_h / 2.0, end_w, box_h, "misaligned\nblind prediction", BUG, fontsize=7.0)

    # Bottom row: repair path.
    ax.text(start_x, bottom_center_y + 0.078, "repair path", ha="left", va="center", fontsize=7.2, weight="bold", color=REPAIR_DARK)
    add_box(ax, start_x, bottom_center_y - start_h / 2.0, start_w, start_h, "Frozen schema\nalignment", REPAIR, fontsize=7.0)
    add_arrow(ax, start_x + start_w, bottom_center_y, schema_x - 0.008, bottom_center_y, REPAIR_DARK)

    add_table(
        ax,
        schema_x,
        bottom_dev_y,
        schema_w,
        table_h,
        "Frozen development/calibration schema",
        ["old=A", "old=B", "att=x", "OTHER"],
        "dev.",
        [1, 0, 1, 0],
        {0, 2},
        REPAIR,
    )
    add_table(
        ax,
        schema_x,
        bottom_blind_y,
        schema_w,
        table_h,
        "Secondary blind encoded using frozen schema",
        ["old=A", "old=B", "att=x", "OTHER"],
        "blind",
        [0, 0, 1, 1],
        {2, 3},
        REPAIR,
    )
    add_column_outline(ax, schema_x, bottom_dev_y, schema_w, table_h, 4, 3, REPAIR_DARK)
    add_column_outline(ax, schema_x, bottom_blind_y, schema_w, table_h, 4, 3, REPAIR_DARK)
    ax.text(schema_x + schema_w / 2.0, bottom_blind_y - 0.032, "unseen old=D mapped to OTHER", ha="center", va="center", fontsize=7.0, color=REPAIR_DARK)

    add_arrow(ax, schema_x + schema_w, bottom_dev_y + table_h / 2.0, mid_x - 0.008, bottom_center_y, REPAIR_DARK)
    add_arrow(ax, schema_x + schema_w, bottom_blind_y + table_h / 2.0, mid_x - 0.008, bottom_center_y, REPAIR_DARK)
    add_box(ax, mid_x, bottom_center_y - box_h / 2.0, mid_w, box_h, "fixed\nschema", REPAIR, fontsize=7.0)
    add_arrow(ax, mid_x + mid_w, bottom_center_y, end_x - 0.008, bottom_center_y, REPAIR_DARK)
    add_box(ax, end_x, bottom_center_y - box_h / 2.0, end_w, box_h, "schema-aligned\nblind prediction", REPAIR, fontsize=7.0)

    add_top10_inset(ax)
    return fig


def write_source_data():
    df = pd.DataFrame(
        [
            {
                "condition": "naive_independent_one_hot",
                "schema_control": "none",
                "blind_column_status": "mismatched",
                "prediction_status": "misaligned",
                "blind_top10": TOP10_NAIVE,
            },
            {
                "condition": "frozen_categorical_schema",
                "schema_control": "OTHER_plus_fixed_column_order",
                "blind_column_status": "aligned",
                "prediction_status": "schema_aligned",
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
