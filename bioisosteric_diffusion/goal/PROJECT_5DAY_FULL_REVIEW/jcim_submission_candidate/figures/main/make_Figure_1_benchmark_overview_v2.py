#!/usr/bin/env python
"""Main Figure 1 v2: benchmark overview, not modeling pipeline."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


HERE = Path(__file__).resolve().parent
OUT = HERE / "Figure_1_benchmark_overview_v2"
CAPTION = HERE / "Figure_1_caption_v2.md"
AUDIT = HERE / "Figure_1_audit_v2.md"

DARK = "#222222"
MID = "#5b5b5b"
LIGHT = "#d6d6d6"
BLUE = "#1b4f9c"
TEAL = "#0b6f6b"
RED = "#b23a3a"
FILL_BLUE = "#f5f8ff"
FILL_TEAL = "#f2fbfa"
FILL_RED = "#fff5f3"
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


def text(ax, x, y, s, size=8, color=DARK, weight="normal",
         ha="left", va="center", linespacing=1.12):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color,
        weight=weight,
        ha=ha,
        va=va,
        linespacing=linespacing,
    )


def rect(ax, x, y, w, h, edge=LIGHT, fill=WHITE, lw=0.8, ls="-"):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            linestyle=ls,
            joinstyle="miter",
        )
    )


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=0.85, scale=10):
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


def panel_title(ax, x, y, letter, title, color=BLUE):
    text(ax, x, y, letter, size=12, color=color, weight="bold")
    text(ax, x + 0.26, y, title, size=9.2, color=DARK, weight="bold")


def draw_fragment_icon(ax, x, y, scale=1.0):
    """Abstract scaffold-fragment sketch; intentionally not a real molecule."""
    # scaffold line
    ax.plot([x, x + 0.78 * scale], [y, y], color=DARK, lw=1.0)
    # old fragment block
    rect(ax, x + 0.80 * scale, y - 0.17 * scale, 0.42 * scale, 0.34 * scale,
         edge=BLUE, fill=FILL_BLUE, lw=0.9)
    # attachment dot
    ax.add_patch(Circle((x + 0.78 * scale, y), 0.055 * scale,
                        facecolor=TEAL, edgecolor=TEAL, lw=0.5))
    text(ax, x + 0.21 * scale, y - 0.28 * scale, "scaffold", size=5.4, color=MID, ha="center")
    text(ax, x + 1.01 * scale, y - 0.28 * scale, "old", size=5.4, color=BLUE, ha="center")
    text(ax, x + 0.78 * scale, y + 0.25 * scale, "sigma", size=5.3, color=TEAL, ha="center")


def draw_candidate_list(ax, x, y):
    rect(ax, x, y, 2.05, 2.18, edge=BLUE, fill=WHITE, lw=0.8)
    text(ax, x + 0.12, y + 1.98, "closed vocabulary", size=7.5, color=BLUE, weight="bold")
    candidates = [
        ("c1", False),
        ("c2", True),
        ("c3", False),
        ("...", False),
        ("c149", True),
        ("c150", False),
    ]
    for i, (label, pos) in enumerate(candidates):
        yy = y + 1.65 - i * 0.27
        text(ax, x + 0.30, yy, label, size=7.0, color=DARK)
        if pos:
            text(ax, x + 0.76, yy + 0.01, "*", size=10.0, color=TEAL, weight="bold")
            text(ax, x + 0.95, yy, "true", size=6.2, color=TEAL)
    text(ax, x + 0.12, y + 0.15, "candidate fragments", size=6.1, color=MID)


def draw_ranked_list(ax, x, y):
    rect(ax, x, y, 2.30, 2.18, edge=TEAL, fill=FILL_TEAL, lw=0.9)
    text(ax, x + 0.13, y + 1.98, "ranked list", size=7.6, color=TEAL, weight="bold")
    # Top-10 bracket.
    ax.plot([x + 0.18, x + 0.18], [y + 0.56, y + 1.70], color=TEAL, lw=1.0)
    ax.plot([x + 0.18, x + 0.30], [y + 1.70, y + 1.70], color=TEAL, lw=1.0)
    ax.plot([x + 0.18, x + 0.30], [y + 0.56, y + 0.56], color=TEAL, lw=1.0)
    text(ax, x + 0.32, y + 1.60, "1", size=6.5)
    text(ax, x + 0.32, y + 1.34, "2  *", size=6.5, color=TEAL, weight="bold")
    text(ax, x + 0.32, y + 1.08, "3", size=6.5)
    text(ax, x + 0.32, y + 0.82, "...", size=6.5)
    text(ax, x + 0.32, y + 0.58, "10", size=6.5)
    text(ax, x + 0.82, y + 1.13, "Top-10\nwindow", size=6.2, color=TEAL, ha="center")
    text(ax, x + 0.13, y + 0.22, "hit if any true replacement\nappears in top 10",
         size=5.9, color=DARK, linespacing=1.05)


def draw_panel_a(ax):
    x, y, w, h = 0.45, 4.90, 13.10, 3.35
    rect(ax, x, y, w, h, edge=LIGHT, fill=WHITE, lw=0.65)
    panel_title(ax, x + 0.22, y + h - 0.34, "A", "One query, many candidates", color=BLUE)

    draw_fragment_icon(ax, x + 0.75, y + 1.78, scale=1.18)
    text(ax, x + 0.67, y + 1.10, "q = (old fragment, attachment signature)",
         size=7.2, color=DARK, weight="bold")
    text(ax, x + 0.78, y + 0.75, "multi-positive ranking", size=6.5, color=MID)

    arrow(ax, x + 3.58, y + 1.78, x + 4.10, y + 1.78, lw=0.8, scale=10)
    draw_candidate_list(ax, x + 4.28, y + 0.74)
    arrow(ax, x + 6.52, y + 1.78, x + 7.04, y + 1.78, lw=0.8, scale=10)
    draw_ranked_list(ax, x + 7.22, y + 0.74)

    # Minimal vocabulary source note.
    rect(ax, x + 10.12, y + 1.02, 2.40, 1.62, edge=LIGHT, fill=FILL_BLUE, lw=0.65)
    text(ax, x + 10.32, y + 2.31, "closed vocabulary", size=7.2, color=BLUE, weight="bold")
    text(ax, x + 10.32, y + 1.91, "train-derived\nreplacement set", size=6.5, color=DARK, linespacing=1.08)
    text(ax, x + 10.32, y + 1.26, "no open-vocabulary\ngeneration", size=6.2, color=MID, linespacing=1.08)


def draw_transform_card(ax, x, y, title, edge, fill, risk=False):
    h = 1.18
    rect(ax, x, y, 3.12, h, edge=edge, fill=fill, lw=0.85)
    text(ax, x + 0.18, y + h - 0.22, title, size=7.0, color=edge, weight="bold")
    if risk:
        text(ax, x + 0.22, y + 0.62, "train: (old A, sigma1)", size=5.95, color=DARK)
        text(ax, x + 0.22, y + 0.37, "test:  (old A, sigma1)", size=5.95, color=DARK)
        text(ax, x + 1.12, y + 0.12, "same transform", size=5.3, color=RED, weight="bold")
    else:
        text(ax, x + 0.22, y + 0.62, "train transforms", size=5.95, color=DARK)
        text(ax, x + 0.22, y + 0.37, "blind transforms", size=5.95, color=DARK)
        # Small non-overlap sets.
        for i, label in enumerate(["T1", "T2", "T3"]):
            rect(ax, x + 1.56 + i * 0.36, y + 0.52, 0.25, 0.20, edge=LIGHT, fill=WHITE, lw=0.45)
            text(ax, x + 1.685 + i * 0.36, y + 0.62, label, size=4.8, ha="center")
        for i, label in enumerate(["U1", "U2", "U3"]):
            rect(ax, x + 1.56 + i * 0.36, y + 0.27, 0.25, 0.20, edge=LIGHT, fill=WHITE, lw=0.45)
            text(ax, x + 1.685 + i * 0.36, y + 0.37, label, size=4.8, ha="center")
        text(ax, x + 0.22, y + 0.10, "zero (old fragment, sigma) overlap",
             size=5.15, color=TEAL, weight="bold")


def draw_panel_bc(ax):
    y = 2.78
    rect(ax, 0.45, y, 6.40, 1.98, edge=LIGHT, fill=WHITE, lw=0.65)
    rect(ax, 7.15, y, 6.40, 1.98, edge=LIGHT, fill=WHITE, lw=0.65)
    panel_title(ax, 0.67, y + 1.63, "B", "Random split risk", color=RED)
    panel_title(ax, 7.37, y + 1.63, "C", "Transform-heldout protocol", color=TEAL)
    draw_transform_card(ax, 1.00, y + 0.22, "Random split", RED, FILL_RED, risk=True)
    text(ax, 4.70, y + 0.80, "leakage\nrisk", size=6.15, color=RED, weight="bold", ha="center")
    draw_transform_card(ax, 7.70, y + 0.22, "Transform-heldout", TEAL, FILL_TEAL, risk=False)
    text(ax, 11.45, y + 0.80, "no shared\ntransform identity", size=5.95, color=TEAL, weight="bold", ha="center")


def draw_panel_d(ax):
    x, y, w, h = 0.45, 0.55, 13.10, 1.70
    rect(ax, x, y, w, h, edge=LIGHT, fill=WHITE, lw=0.65)
    panel_title(ax, x + 0.22, y + h - 0.34, "D", "Evaluation tiers", color=BLUE)

    labels = [
        ("Train", "build vocabulary"),
        ("Development/\ncalibration", "tune protocol only"),
        ("Secondary blind", "primary claim"),
        ("Canonical analysis", "robustness/mechanism only"),
    ]
    centers = [x + 1.85, x + 4.82, x + 7.78, x + 10.75]
    colors = [MID, BLUE, TEAL, MID]
    for i, ((title, role), cx, color) in enumerate(zip(labels, centers, colors)):
        text(ax, cx, y + 0.87, title, size=7.4, color=color, weight="bold", ha="center", linespacing=1.0)
        text(ax, cx, y + 0.43, role, size=5.9, color=MID, ha="center", linespacing=1.02)
        if i < len(centers) - 1:
            arrow(ax, cx + 0.90, y + 0.86, centers[i + 1] - 0.92, y + 0.86, color=MID, lw=0.72, scale=9)
    ax.plot([centers[1], centers[2]], [y + 0.20, y + 0.20], color=TEAL, lw=0.75)
    text(ax, (centers[1] + centers[2]) / 2, y + 0.08, "no blind tuning", size=5.7, color=TEAL, ha="center")


def write_docs():
    CAPTION.write_text(
        "Figure 1. Leakage-controlled closed-vocabulary fragment replacement ranking. "
        "A query is defined by an old fragment and attachment signature, and the task is to rank attachment-compatible replacements from a closed train-derived vocabulary. "
        "A Top-10 hit is counted if any structure-derived true replacement appears within the top 10. "
        "Random splits can place the same fragment-attachment transform in train and evaluation sets, creating transform leakage. "
        "The transform-heldout protocol separates transform identities across splits; secondary blind queries support primary performance claims, while canonical analyses are used only for robustness and mechanism checks.\n",
        encoding="utf-8",
    )
    AUDIT.write_text(
        "# Figure 1 v2 audit\n\n"
        "1. Does Panel A use a query/candidate visual anchor rather than a generic PPT flow? Yes.\n"
        "2. Are base rankers, Borda, MLP, Score Blend, HistGB scorer, 82/77-feature scorers, prior-rank removal, A4C, performance values, and rescue/lost absent? Yes.\n"
        "3. Does Panel B show the random split risk with repeated transform identity? Yes.\n"
        "4. Does Panel C show transform-heldout separation and zero overlap? Yes.\n"
        "5. Does Panel D show Train -> Development/calibration -> Secondary blind -> Canonical analysis with role labels? Yes.\n"
        "6. Is secondary blind bounded as the primary claim tier and canonical analysis as robustness/mechanism only? Yes.\n"
        "7. Does the SVG keep editable text and avoid embedded raster images? Yes; svg.fonttype is none and no image is embedded.\n",
        encoding="utf-8",
    )


def main():
    setup()
    fig = plt.figure(figsize=(7.35, 4.55))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.65)
    ax.axis("off")

    text(ax, 7.0, 8.35, "Leakage-controlled closed-vocabulary fragment replacement ranking",
         size=11.4, color=DARK, weight="bold", ha="center")

    draw_panel_a(ax)
    draw_panel_bc(ax)
    draw_panel_d(ax)

    fig.savefig(f"{OUT}.svg", facecolor="white")
    fig.savefig(f"{OUT}.pdf", facecolor="white")
    fig.savefig(f"{OUT}.png", dpi=600, facecolor="white")
    plt.close(fig)
    write_docs()

    for path in [f"{OUT}.svg", f"{OUT}.pdf", f"{OUT}.png", str(CAPTION), str(AUDIT)]:
        print(path)


if __name__ == "__main__":
    main()
