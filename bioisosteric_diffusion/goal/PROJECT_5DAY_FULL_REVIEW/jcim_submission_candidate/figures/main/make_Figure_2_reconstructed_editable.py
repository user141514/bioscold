#!/usr/bin/env python
"""Reconstruct the provided Figure 2 screenshot as an editable SVG.

The output is intentionally vector-native: boxes, arrows, rules, and text are
separate SVG elements. No raster screenshot is embedded.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


HERE = Path(__file__).resolve().parent
OUT = HERE / "Figure_2_reconstructed_editable_bus_aligned_v2"

DARK = "#1f1f1f"
ARROW = "#333333"
GRAY_EDGE = "#4a4a4a"
GRAY_TEXT = "#333333"
BLUE = "#0b3f99"
TEAL = "#0b6f6b"
RED = "#b30000"
FILL_BLUE = "#f4f8ff"
FILL_TEAL = "#f2fbfa"
FILL_RED = "#fff6f4"
WHITE = "#ffffff"


def setup():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def add_text(ax, x, y, text, size=11, color=DARK, weight="normal",
             ha="left", va="center", linespacing=1.15):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        weight=weight,
        ha=ha,
        va=va,
        linespacing=linespacing,
    )


def add_rect(ax, x, y, w, h, edge=GRAY_EDGE, fill=WHITE, lw=1.15):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            joinstyle="miter",
        )
    )


def add_arrow(ax, x1, y1, x2, y2, color=ARROW, lw=1.0, scale=13):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0",
        )
    )


def add_centered_box(ax, x, y, w, h, lines, edge=GRAY_EDGE, fill=WHITE,
                     color=DARK, lw=1.15, sizes=None, weights=None):
    add_rect(ax, x, y, w, h, edge=edge, fill=fill, lw=lw)
    if sizes is None:
        sizes = [10.5] * len(lines)
    if weights is None:
        weights = ["normal"] * len(lines)
    spacing = 0.31
    start = y + h / 2 + spacing * (len(lines) - 1) / 2
    for i, line in enumerate(lines):
        add_text(
            ax,
            x + w / 2,
            start - spacing * i,
            line,
            size=sizes[i],
            color=color,
            weight=weights[i],
            ha="center",
            va="center",
        )


def draw_panel_a(ax):
    add_text(ax, 0.20, 9.72, "A. Candidate-level scoring pipeline",
             size=13.5, weight="bold")
    pipeline_y = 7.01

    # Query-candidate pair.
    q_y = pipeline_y - 1.45 / 2
    add_rect(ax, 0.20, q_y, 2.45, 1.45, edge=GRAY_EDGE, fill=WHITE, lw=1.05)
    add_text(ax, 0.42, q_y + 1.10, "Query-candidate pair", size=8.8, weight="bold")
    add_text(ax, 0.42, q_y + 0.68, "old fragment + attachment", size=7.5)
    add_text(ax, 0.42, q_y + 0.38, "candidate", size=7.5)
    add_arrow(ax, 2.78, pipeline_y, 3.18, pipeline_y, lw=1.05, scale=14)

    # Base-ranker evidence block.
    add_text(ax, 4.55, 8.98, "Base-ranker evidence", size=10.0,
             color=BLUE, weight="bold", ha="center")
    x0, w = 3.35, 2.75
    ys = [8.12, 7.42, 6.72, 6.02, 5.32]
    labels = [
        "Attachment-Frequency",
        "HGB",
        "Dual Encoder",
        "Borda(DE,HGB)",
        "MLP / Score Blend",
    ]
    for y, label in zip(ys, labels):
        add_centered_box(
            ax,
            x0,
            y,
            w,
            0.58,
            [label],
            edge=BLUE,
            fill=FILL_BLUE,
            color=DARK,
            lw=1.05,
            sizes=[8.1],
            weights=["bold" if label == "Attachment-Frequency" else "normal"],
        )

    # Evidence bus: ranker outputs are stacked before entering the feature matrix.
    # This avoids a dense fan-in arrowhead cluster and reads more like an ML
    # architecture diagram than a debug flowchart.
    bus_x = 6.62
    lane_centers = [8.41, 7.71, 7.01, 6.31, 5.61]
    for y in lane_centers:
        ax.plot([x0 + w + 0.02, bus_x], [y, y], color=ARROW, lw=0.72)
    ax.plot([bus_x, bus_x], [min(lane_centers), max(lane_centers)], color=ARROW, lw=1.25)
    add_text(ax, bus_x + 0.07, 5.25, "stack", size=7.0, color=GRAY_TEXT, ha="left")

    # Candidate feature matrix.
    mx, my, mw, mh = 7.18, 5.02, 3.42, 3.95
    add_rect(ax, mx, my, mw, mh, edge=TEAL, fill=FILL_TEAL, lw=1.35)
    add_text(ax, mx + mw / 2, my + mh - 0.55,
             "Candidate-level feature", size=9.5, color=TEAL,
             weight="bold", ha="center")
    add_text(ax, mx + mw / 2, my + mh - 0.92,
             "matrix", size=9.5, color=TEAL, weight="bold", ha="center")
    feature_lines = [
        ("model outputs", 7.47),
        ("train-derived priors", 6.82),
        ("molecular descriptors", 6.17),
        ("frozen categorical schema", 5.52),
    ]
    for label, y in feature_lines:
        add_text(ax, mx + mw / 2, y, label, size=8.0, ha="center")
    for y in [7.18, 6.53, 5.88]:
        ax.plot([mx + 0.28, mx + mw - 0.28], [y, y], color=TEAL, lw=0.65)

    # Scorer and final ranking.
    bus_out_y = lane_centers[2]
    add_arrow(ax, bus_x, bus_out_y, mx - 0.12, bus_out_y, lw=1.05, scale=14)
    add_arrow(ax, mx + mw + 0.10, pipeline_y, 10.95, pipeline_y, lw=1.05, scale=14)
    add_centered_box(
        ax,
        11.10,
        pipeline_y - 1.65 / 2,
        1.60,
        1.65,
        ["HistGB scorer", "candidate-level", "integration"],
        edge=TEAL,
        fill=FILL_TEAL,
        color=TEAL,
        lw=1.25,
        sizes=[8.5, 7.2, 7.2],
        weights=["bold", "normal", "normal"],
    )
    add_arrow(ax, 12.75, pipeline_y, 13.10, pipeline_y, lw=1.05, scale=14)
    add_centered_box(
        ax,
        13.28,
        pipeline_y - 1.65 / 2,
        1.50,
        1.65,
        ["Final ranking", "per-query", "candidate order"],
        edge=GRAY_EDGE,
        fill=WHITE,
        color=DARK,
        lw=1.05,
        sizes=[8.4, 7.2, 7.2],
        weights=["bold", "normal", "normal"],
    )


def draw_panel_b(ax):
    add_text(ax, 0.20, 4.17, "B. Post-audit pruning strip",
             size=13.5, weight="bold")

    y, h = 2.55, 1.32
    initial = (0.72, y, 3.22, h)
    diagnostic = (4.62, y, 2.35, h)
    remove = (7.62, y, 2.72, h)
    post = (11.06, y, 3.18, h)

    add_centered_box(
        ax,
        *initial,
        ["Initial 82-feature scorer", "base 77-feature set", "+ prior_ranks"],
        edge=TEAL,
        fill=FILL_TEAL,
        color=TEAL,
        lw=1.25,
        sizes=[8.7, 7.4, 7.4],
        weights=["bold", "normal", "normal"],
    )
    add_centered_box(
        ax,
        1.28,
        2.07,
        2.10,
        0.36,
        ["prospective result"],
        edge=TEAL,
        fill=WHITE,
        color=TEAL,
        lw=1.15,
        sizes=[8.5],
        weights=["bold"],
    )

    add_centered_box(
        ax,
        *diagnostic,
        ["Secondary blind", "diagnostics"],
        edge=GRAY_EDGE,
        fill=WHITE,
        color=DARK,
        lw=1.05,
        sizes=[8.8, 8.8],
        weights=["bold", "bold"],
    )

    add_centered_box(
        ax,
        *remove,
        ["Remove prior_ranks", "non-transferable", "shortcut"],
        edge=RED,
        fill=FILL_RED,
        color=RED,
        lw=1.10,
        sizes=[8.4, 7.4, 7.4],
        weights=["bold", "normal", "normal"],
    )
    add_centered_box(
        ax,
        7.90,
        2.02,
        2.08,
        0.38,
        ["removed add-on"],
        edge=RED,
        fill=WHITE,
        color=RED,
        lw=1.05,
        sizes=[8.2],
        weights=["bold"],
    )

    add_centered_box(
        ax,
        *post,
        ["Post-audit 77-feature scorer", "base 77-feature set"],
        edge=TEAL,
        fill=FILL_TEAL,
        color=TEAL,
        lw=1.25,
        sizes=[8.5, 7.4],
        weights=["bold", "normal"],
    )
    add_centered_box(
        ax,
        11.48,
        2.02,
        2.20,
        0.38,
        ["post-audit locked"],
        edge=TEAL,
        fill=WHITE,
        color=TEAL,
        lw=1.05,
        sizes=[8.2],
        weights=["bold"],
    )

    # Strip arrows.
    cy = y + h / 2
    add_arrow(ax, initial[0] + initial[2] + 0.15, cy, diagnostic[0] - 0.15, cy, lw=1.05, scale=14)
    add_arrow(ax, diagnostic[0] + diagnostic[2] + 0.15, cy, remove[0] - 0.15, cy, lw=1.05, scale=14)
    add_arrow(ax, remove[0] + remove[2] + 0.15, cy, post[0] - 0.15, cy, lw=1.05, scale=14)

    # Bottom-right notes, kept as editable text rather than cards.
    add_text(ax, 10.60, 1.00, "Score Blend - strongest pre-D4S baseline",
             size=8.8, weight="bold")
    add_text(ax, 10.60, 0.63, "A4C strata - exploratory triage only",
             size=8.8, weight="bold")


def write_audit():
    audit = HERE / "Figure_2_reconstructed_editable_audit.md"
    audit.write_text(
        "# Figure 2 reconstructed editable SVG audit\n\n"
        "- The SVG is reconstructed from the provided screenshot using vector rectangles, lines, arrows, and editable text.\n"
        "- No raster screenshot is embedded in the SVG.\n"
        "- Text remains SVG text because `svg.fonttype` is set to `none`.\n"
        "- The A/B layout, base-ranker evidence stack, candidate-level feature matrix, HistGB scorer, final ranking, post-audit pruning strip, and bottom-right Score Blend/A4C notes are preserved.\n",
        encoding="utf-8",
    )


def main():
    setup()
    fig = plt.figure(figsize=(10, 7.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 11.25)
    ax.axis("off")

    add_text(
        ax,
        7.5,
        10.72,
        "Candidate-level scoring pipeline and evidence hierarchy",
        size=17,
        weight="bold",
        ha="center",
    )
    draw_panel_a(ax)
    draw_panel_b(ax)

    fig.savefig(f"{OUT}.svg", facecolor="white")
    fig.savefig(f"{OUT}.pdf", facecolor="white")
    fig.savefig(f"{OUT}.png", dpi=600, facecolor="white")
    plt.close(fig)
    write_audit()

    print(f"{OUT}.svg")
    print(f"{OUT}.pdf")
    print(f"{OUT}.png")
    print(HERE / "Figure_2_reconstructed_editable_audit.md")


if __name__ == "__main__":
    main()
