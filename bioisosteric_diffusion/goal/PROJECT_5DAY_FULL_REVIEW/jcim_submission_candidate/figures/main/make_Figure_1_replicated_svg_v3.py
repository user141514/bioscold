#!/usr/bin/env python
"""Replicate the provided Figure 1 sketch as editable vector SVG."""
from pathlib import Path
from math import cos, sin, pi

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon, Rectangle


HERE = Path(__file__).resolve().parent
OUT = HERE / "Figure_1_replicated_svg_v3"
CAPTION = HERE / "Figure_1_caption_replicated_v3.md"
AUDIT = HERE / "Figure_1_audit_replicated_v3.md"

NAVY = "#08204a"
BLUE = "#0b2e66"
TEAL = "#087a74"
RED = "#b31f24"
DARK = "#111111"
MID = "#555555"
LIGHT = "#c9c9c9"
VERY_LIGHT = "#f7f7f7"
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
         ha="left", va="center", linespacing=1.08, style="normal"):
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
        style=style,
    )


def rect(ax, x, y, w, h, edge=NAVY, fill=WHITE, lw=1.0, ls="-"):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            linestyle=ls,
            joinstyle="round",
        )
    )


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.15, scale=13, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0",
        )
    )


def panel(ax, x, y, w, h, color=NAVY, lw=1.0):
    rect(ax, x, y, w, h, edge=color, fill=WHITE, lw=lw)


def draw_star(ax, x, y, color=NAVY, size=0.10):
    pts = []
    for i in range(10):
        r = size if i % 2 == 0 else size * 0.42
        a = pi / 2 + i * pi / 5
        pts.append((x + r * cos(a), y + r * sin(a)))
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=color, lw=0.4))


def draw_database(ax, x, y, w=0.38, h=0.60, color=NAVY):
    ax.add_patch(Arc((x + w / 2, y + h), w, 0.18, theta1=0, theta2=360, color=color, lw=1.0))
    ax.plot([x, x], [y + 0.08, y + h], color=color, lw=1.0)
    ax.plot([x + w, x + w], [y + 0.08, y + h], color=color, lw=1.0)
    ax.add_patch(Arc((x + w / 2, y + 0.08), w, 0.18, theta1=180, theta2=360, color=color, lw=1.0))
    for yy in [y + 0.22, y + 0.38]:
        ax.add_patch(Arc((x + w / 2, yy), w, 0.18, theta1=180, theta2=360, color=color, lw=0.9))


def draw_chemical_query(ax, x, y):
    # enclosing molecule box
    rect(ax, x, y, 2.35, 1.15, edge=MID, fill=WHITE, lw=0.75)
    # abstract benzene ring
    cx, cy, r = x + 0.72, y + 0.58, 0.35
    pts = [(cx + r * cos(pi / 6 + i * pi / 3), cy + r * sin(pi / 6 + i * pi / 3)) for i in range(6)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=WHITE, edgecolor=DARK, lw=0.9))
    # internal alternating bonds
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        ax.plot([pts[i][0] * 0.88 + cx * 0.12, pts[j][0] * 0.88 + cx * 0.12],
                [pts[i][1] * 0.88 + cy * 0.12, pts[j][1] * 0.88 + cy * 0.12],
                color=DARK, lw=0.65)
    # attachment squiggle and chlorine
    ax.plot([x + 0.18, pts[3][0]], [cy, pts[3][1]], color=DARK, lw=0.9)
    text(ax, x + 0.13, cy + 0.02, "}", size=16, color=DARK, ha="center", va="center")
    ax.plot([pts[0][0], x + 1.29], [pts[0][1], y + 0.80], color=DARK, lw=0.9)
    text(ax, x + 1.33, y + 0.82, "Cl", size=9, color=DARK)
    # plus and attachment signature sketch
    text(ax, x + 1.50, y + 0.60, "+", size=12, color=DARK, ha="center")
    c2x, c2y = x + 2.00, y + 0.62
    ax.add_patch(Circle((c2x, c2y), 0.055, facecolor="#cccccc", edgecolor=DARK, lw=0.7))
    for dx, dy in [(0, 0.33), (-0.25, -0.20), (0.25, -0.20)]:
        ax.plot([c2x, c2x + dx], [c2y, c2y + dy], color=DARK, lw=0.7)
        ax.add_patch(Circle((c2x + dx, c2y + dy), 0.045, facecolor=WHITE, edgecolor=NAVY, lw=0.8))


def draw_panel_a(ax):
    panel(ax, 0.20, 4.95, 10.70, 3.38, LIGHT, lw=0.75)
    text(ax, 0.38, 7.94, "A", size=21, color=NAVY, weight="bold")
    text(ax, 1.00, 7.95, "One query, many candidates", size=13.5, color=NAVY, weight="bold")

    text(ax, 1.00, 7.38, r"$q$ = (old fragment,", size=10.3, color=NAVY)
    text(ax, 1.00, 7.10, "attachment signature)", size=10.3, color=NAVY)
    draw_chemical_query(ax, 0.52, 5.68)

    arrow(ax, 3.24, 6.58, 3.75, 6.58, NAVY, lw=1.35, scale=16)

    # closed vocabulary list
    rect(ax, 3.95, 5.36, 1.52, 2.39, edge=NAVY, fill=WHITE, lw=1.0)
    text(ax, 4.10, 7.55, "Candidate set", size=10.1, color=NAVY, weight="bold")
    text(ax, 4.10, 7.31, "closed train-derived", size=6.9, color=NAVY)
    text(ax, 4.10, 7.12, "vocabulary", size=6.9, color=NAVY)
    text(ax, 4.10, 6.93, "~150 replacements", size=6.4, color=MID)
    y0 = 6.66
    entries = [(r"$c_1$", False), (r"$c_2$", True), (r"$c_3$", False), ("...", False), (r"$c_{150}$", True)]
    for i, (lab, pos) in enumerate(entries):
        yy = y0 - i * 0.30
        text(ax, 4.25, yy, lab, size=10.0, color=DARK)
        if pos:
            draw_star(ax, 4.67, yy + 0.03, NAVY, 0.085)
            text(ax, 4.98, yy, "true", size=9.2, color=NAVY)

    arrow(ax, 5.78, 6.58, 6.28, 6.58, NAVY, lw=1.35, scale=16)

    # ranked list
    rect(ax, 6.42, 5.50, 1.56, 2.25, edge=NAVY, fill=WHITE, lw=1.0)
    text(ax, 6.80, 7.53, "Ranked list", size=10.4, color=NAVY, weight="bold")
    ranks = [("1", r"$c_9$", False), ("2", r"$c_2$", True), ("3", r"$c_{42}$", False), ("...", "...", False), ("10", r"$c_{150}$", True)]
    for i, (rk, cand, pos) in enumerate(ranks):
        yy = 7.12 - i * 0.34
        text(ax, 6.72, yy, rk, size=9.6, color=DARK, ha="center")
        text(ax, 7.18, yy, cand, size=9.6, color=DARK, ha="center")
        if pos:
            draw_star(ax, 7.52, yy + 0.03, NAVY, 0.078)
    # Top-10 bracket
    bx = 7.74
    ax.plot([bx, bx + 0.14], [7.14, 7.14], color=NAVY, lw=1.0)
    ax.plot([bx + 0.14, bx + 0.14], [5.78, 7.14], color=NAVY, lw=1.0)
    ax.plot([bx, bx + 0.14], [5.78, 5.78], color=NAVY, lw=1.0)
    arrow(ax, bx + 0.18, 6.46, bx + 0.29, 6.46, NAVY, lw=0.85, scale=9)
    text(ax, bx + 0.34, 6.45, "Top-10", size=10.0, color=NAVY)

    # divider and closed vocabulary note
    ax.plot([8.80, 8.80], [5.34, 7.84], color=LIGHT, lw=0.9, linestyle=(0, (1, 3)))
    text(ax, 9.12, 6.58, "side note", size=5.7, color=MID, weight="bold")
    text(ax, 9.12, 6.34, "closed train-derived\nvocabulary", size=6.2, color=NAVY, linespacing=1.05)
    text(ax, 9.12, 5.92, "no open-vocabulary\ngeneration", size=5.9, color=MID, linespacing=1.05)

    text(ax, 5.92, 5.12, "Top-10 hit if any true replacement appears in window",
         size=10.5, color=NAVY)


def draw_panel_b(ax):
    panel(ax, 0.20, 0.40, 4.86, 4.42, LIGHT, lw=0.75)
    text(ax, 0.38, 4.46, "B", size=21, color=RED, weight="bold")
    text(ax, 1.00, 4.50, "Random split leakage", size=12.2, color=RED, weight="bold")
    text(ax, 1.52, 3.80, "Train examples", size=8.5, color=RED, weight="bold")
    text(ax, 3.25, 3.80, "Evaluation examples", size=8.5, color=RED, weight="bold")
    text(ax, 0.43, 3.20, "Transform", size=8.5, color=RED, weight="bold")
    text(ax, 0.52, 2.93, "identity", size=8.5, color=RED, weight="bold")

    tx, ty, tw, th = 1.18, 2.57, 3.42, 1.17
    rect(ax, tx, ty, tw, th, edge=MID, fill=WHITE, lw=0.55)
    ax.plot([tx + tw / 2, tx + tw / 2], [ty, ty + th], color=LIGHT, lw=0.6)
    for yy in [ty + th / 3, ty + 2 * th / 3]:
        ax.plot([tx, tx + tw], [yy, yy], color=LIGHT, lw=0.55)
    rows = [r"$T_1$", r"$T_2$", r"$T_3$"]
    rows2 = [r"$T_2$", r"$T_4$", r"$T_5$"]
    for i, lab in enumerate(rows):
        text(ax, tx + tw * 0.25, ty + th - 0.22 - i * th / 3, lab, size=10.0, ha="center")
        text(ax, tx + tw * 0.75, ty + th - 0.22 - i * th / 3, rows2[i], size=10.0, ha="center")
    # Highlight shared T2 in train and test. Train row 2 is centered at th/2;
    # test row 1 is centered at th - 0.22.
    train_t2_y = ty + th / 2
    test_t2_y = ty + th - 0.22
    rect(ax, tx, ty + th / 3, tw / 2, th / 3, edge=RED, fill="none", lw=0.75)
    rect(ax, tx + tw / 2, ty + 2 * th / 3, tw / 2, th / 3, edge=RED, fill="none", lw=0.75)
    ax.plot([tx + tw * 0.25, tx + tw * 0.75],
            [train_t2_y, test_t2_y], color=RED, lw=0.8, linestyle=(0, (3, 3)))

    arrow(ax, 2.89, 2.47, 2.89, 2.13, RED, lw=1.0, scale=13)
    text(ax, 2.89, 1.94, "Shared transform identity across train/test",
         size=8.0, color=RED, ha="center")
    text(ax, 2.89, 1.68, "→ leakage risk", size=9.2, color=RED, weight="bold", ha="center")


def draw_panel_c(ax):
    panel(ax, 5.18, 0.40, 5.72, 4.42, LIGHT, lw=0.75)
    text(ax, 5.38, 4.46, "C", size=21, color=TEAL, weight="bold")
    text(ax, 6.05, 4.50, "Transform-heldout protocol", size=12.2, color=TEAL, weight="bold")
    text(ax, 6.52, 4.02, "Train transforms", size=8.2, color=TEAL, weight="bold")
    text(ax, 8.78, 4.02, "Blind transforms", size=8.2, color=TEAL, weight="bold")
    text(ax, 5.40, 3.40, "Transform", size=8.0, color=TEAL, weight="bold")
    text(ax, 5.48, 3.12, "identity", size=8.0, color=TEAL, weight="bold")

    tx, ty, tw, th = 6.16, 2.68, 3.92, 1.18
    rect(ax, tx, ty, tw, th, edge=MID, fill=WHITE, lw=0.55)
    ax.plot([tx + tw / 2, tx + tw / 2], [ty, ty + th], color=TEAL, lw=0.9, linestyle=(0, (3, 3)))
    for yy in [ty + th / 3, ty + 2 * th / 3]:
        ax.plot([tx, tx + tw], [yy, yy], color=LIGHT, lw=0.55)
    rows = [r"$T_1$", r"$T_2$", r"$T_3$"]
    rows2 = [r"$U_1$", r"$U_2$", r"$U_3$"]
    for i in range(3):
        text(ax, tx + tw * 0.22, ty + th - 0.22 - i * th / 3, rows[i], size=10.0, ha="center")
        text(ax, tx + tw * 0.78, ty + th - 0.22 - i * th / 3, rows2[i], size=10.0, ha="center")
    # no-overlap symbol
    ax.add_patch(Circle((tx + tw / 2, ty + th / 2), 0.13, facecolor=WHITE, edgecolor=TEAL, lw=1.1))
    ax.plot([tx + tw / 2 - 0.08, tx + tw / 2 + 0.08],
            [ty + th / 2 - 0.08, ty + th / 2 + 0.08], color=TEAL, lw=1.1)
    arrow(ax, tx + tw / 2, ty - 0.06, tx + tw / 2, ty - 0.34, TEAL, lw=1.1, scale=13)
    text(ax, 6.26, 2.20, "Zero (old fragment, attachment signature) overlap",
         size=8.0, color=TEAL, style="italic")
    text(ax, 6.55, 1.90, "Secondary blind = primary evaluation",
         size=9.2, color=TEAL, weight="bold")

    # Evaluation rail.
    y = 1.12
    boxes = [
        (5.70, "Train"),
        (7.18, "Development/\ncalibration"),
        (9.04, "Secondary blind"),
    ]
    widths = [1.02, 1.36, 1.26]
    for (x, label), w in zip(boxes, widths):
        rect(ax, x, y, w, 0.40, edge=LIGHT, fill=WHITE, lw=0.55)
        text(ax, x + w / 2, y + 0.21, label, size=6.7, color=TEAL, weight="bold",
             ha="center", linespacing=1.0)
    arrow(ax, 6.82, y + 0.20, 7.10, y + 0.20, TEAL, lw=0.8, scale=9)
    arrow(ax, 8.64, y + 0.20, 8.94, y + 0.20, TEAL, lw=0.8, scale=9)
    # dashed no blind tuning bracket
    ax.plot([6.20, 6.20, 9.70, 9.70], [1.12, 0.82, 0.82, 1.12],
            color=TEAL, lw=0.8, linestyle=(0, (3, 3)))
    text(ax, 7.32, 0.58, "No blind tuning", size=8.0, color=TEAL, style="italic")


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
        "# Figure 1 replicated v3 audit\n\n"
        "- Reconstructed as vector primitives: boxes, text, arrows, abstract structure, stars, tables, and database icon.\n"
        "- No raster screenshot is embedded.\n"
        "- The figure contains no base rankers, Borda, MLP, Score Blend, HistGB scorer, 82/77-feature scorers, prior-rank removal, A4C, performance values, or rescue/lost analysis.\n"
        "- Panel A is the dominant query/candidate visual anchor; Panels B/C explain random-split leakage and transform-heldout separation.\n"
        "- Text remains editable in SVG because `svg.fonttype` is set to `none`.\n",
        encoding="utf-8",
    )


def main():
    setup()
    fig = plt.figure(figsize=(10.3, 7.55))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11.15)
    ax.set_ylim(0, 8.70)
    ax.axis("off")

    text(ax, 5.58, 8.48, "Leakage-controlled closed-vocabulary fragment replacement ranking",
         size=18.0, color=NAVY, weight="bold", ha="center")
    draw_panel_a(ax)
    draw_panel_b(ax)
    draw_panel_c(ax)

    fig.savefig(f"{OUT}.svg", facecolor="white")
    fig.savefig(f"{OUT}.pdf", facecolor="white")
    fig.savefig(f"{OUT}.png", dpi=600, facecolor="white")
    plt.close(fig)
    write_docs()

    for path in [f"{OUT}.svg", f"{OUT}.pdf", f"{OUT}.png", str(CAPTION), str(AUDIT)]:
        print(path)


if __name__ == "__main__":
    main()
