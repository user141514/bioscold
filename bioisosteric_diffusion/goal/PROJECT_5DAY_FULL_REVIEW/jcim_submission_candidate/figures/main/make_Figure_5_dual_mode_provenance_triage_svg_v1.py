#!/usr/bin/env python
"""Figure 5: dual-mode provenance triage, replicated as editable SVG."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


HERE = Path(__file__).resolve().parent
OUT = HERE / "Figure_5_dual_mode_provenance_triage_math_refined"
CAPTION = HERE / "Figure_5_caption_math_refined.md"
AUDIT = HERE / "Figure_5_audit_math_refined.md"

NAVY = "#173f72"
TEAL = "#087a74"
RED = "#cf3c1f"
G4 = "#365f66"
DARK = "#111111"
MID = "#666666"
LIGHT = "#c9c9c9"
PANEL = "#555555"
FILL_BLUE = "#f8fbff"
FILL_TEAL = "#f3fbfa"
FILL_RED = "#fff7f4"
WHITE = "#ffffff"


def setup():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "mathtext.default": "it",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def text(ax, x, y, s, size=8, color=DARK, weight="normal",
         ha="left", va="center", style="normal", linespacing=1.12):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color,
        weight=weight,
        ha=ha,
        va=va,
        style=style,
        linespacing=linespacing,
    )


def rect(ax, x, y, w, h, edge=PANEL, fill=WHITE, lw=0.8, ls="-", hatch=None):
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
            hatch=hatch,
        )
    )


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=0.9, scale=12, ls="-"):
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


def panel(ax, x, y, w, h):
    rect(ax, x, y, w, h, edge=PANEL, fill=WHITE, lw=0.75)


def flow_box(ax, x, y, w, h, label, edge, fill, size=12):
    rect(ax, x, y, w, h, edge=edge, fill=fill, lw=1.0)
    text(ax, x + w / 2, y + h / 2, label, size=size, color=edge if edge != PANEL else DARK,
         weight="bold" if "\n" not in label else "normal", ha="center", linespacing=1.05)


def draw_panel_a(ax):
    x, y, w, h = 0.18, 0.68, 3.72, 6.45
    panel(ax, x, y, w, h)
    text(ax, x + 0.16, y + h - 0.44, "A. Dual-mode proposal routes",
         size=15.0, color=DARK, weight="bold")
    # Divider.
    ax.plot([x + w / 2, x + w / 2], [y + 0.34, y + h - 1.02],
            color=LIGHT, lw=0.8, linestyle=(0, (4, 4)))

    # Headers.
    text(ax, x + 0.82, y + h - 1.20, "Conservative Mode", size=11.0,
         color=NAVY, weight="bold", ha="center")
    text(ax, x + 0.82, y + h - 1.55, "HGB route", size=11.0,
         color=NAVY, style="italic", ha="center")
    text(ax, x + 2.75, y + h - 1.20, "Exploration Mode", size=11.0,
         color=TEAL, weight="bold", ha="center")
    text(ax, x + 2.75, y + h - 1.55, "Borda(DE,HGB) route", size=10.2,
         color=TEAL, style="italic", ha="center")

    # Conservative route.
    hgb_x, hgb_w = x + 0.33, 1.45
    hgb_c = hgb_x + hgb_w / 2
    hgb_y, hgb_h = y + h - 3.10, 0.92
    freq_y, freq_h = y + h - 4.85, 1.22
    flow_box(ax, hgb_x, hgb_y, hgb_w, hgb_h, "HGB", NAVY, FILL_BLUE, size=13.0)
    flow_box(ax, hgb_x, freq_y, hgb_w, freq_h,
             "frequency-\naligned\nproposals", NAVY, FILL_BLUE, size=11.4)
    arrow(ax, hgb_c, hgb_y, hgb_c, freq_y + freq_h, NAVY, lw=1.05, scale=13)

    # Exploration route. Keep boxes separated so the route reads as a flow,
    # not as a stacked merged card.
    ex_x, ex_w = x + 2.18, 1.50
    ex_c = ex_x + ex_w / 2
    borda_y, borda_h = y + h - 3.06, 0.82
    prop_y, prop_h = y + h - 4.36, 0.96
    strata_y, strata_h = y + h - 5.78, 0.96
    flow_box(ax, ex_x, borda_y, ex_w, borda_h,
             "Borda(DE,HGB)", TEAL, FILL_TEAL, size=10.8)
    flow_box(ax, ex_x, prop_y, ex_w, prop_h,
             "exploratory\nproposals", TEAL, FILL_TEAL, size=10.9)
    flow_box(ax, ex_x, strata_y, ex_w, strata_h,
             "provenance\nstrata", TEAL, FILL_TEAL, size=10.9)
    arrow(ax, ex_c, borda_y, ex_c, prop_y + prop_h, TEAL, lw=1.05, scale=13)
    arrow(ax, ex_c, prop_y, ex_c, strata_y + strata_h, TEAL, lw=1.05, scale=13)

    text(ax, x + 0.84, y + 0.35, "Driven by HGB", size=9.2, color=NAVY,
         style="italic", ha="center")
    text(ax, x + 2.75, y + 0.35, "Driven by Borda(DE,HGB)", size=8.5, color=TEAL,
         style="italic", ha="center")


def strata_box(ax, x, y, w, h, title, formula, subtitle, edge, fill, color):
    rect(ax, x, y, w, h, edge=edge, fill=fill, lw=0.9)
    text(ax, x + w / 2, y + h - 0.36, title, size=15.0,
         color=color, weight="bold", ha="center")
    text(ax, x + w / 2, y + h - 0.82, formula, size=12.2,
         color=DARK, ha="center")
    text(ax, x + w / 2, y + 0.33, subtitle, size=10.8,
         color=DARK, ha="center", linespacing=1.10)


def draw_panel_b(ax):
    x, y, w, h = 4.02, 0.68, 2.85, 6.45
    panel(ax, x, y, w, h)
    text(ax, x + 0.16, y + h - 0.44, "B. Provenance strata",
         size=15.0, color=DARK, weight="bold")

    bw = 2.35
    strata_box(
        ax,
        x + 0.38,
        y + h - 2.48,
        bw,
        1.52,
        "G4 shared",
        r"$\mathit{K}_{\mathrm{HGB}}\cap\mathit{K}_{\mathrm{Borda}}$",
        "consensus",
        G4,
        FILL_TEAL,
        G4,
    )
    strata_box(
        ax,
        x + 0.38,
        y + h - 4.42,
        bw,
        1.66,
        "G3 DE-elevated",
        r"$\mathit{K}_{\mathrm{Borda}}\thinspace\backslash\thinspace\mathit{K}_{\mathrm{HGB}}$",
        "similarity-supported\nexpansion",
        TEAL,
        FILL_TEAL,
        TEAL,
    )
    strata_box(
        ax,
        x + 0.38,
        y + h - 6.36,
        bw,
        1.50,
        "G2 Borda-emergent",
        r"$\mathit{K}_{\mathrm{Borda}}\thinspace\backslash\thinspace(\mathit{K}_{\mathrm{HGB}}\cup\mathit{K}_{\mathrm{DE}})$",
        "highest novelty",
        RED,
        FILL_RED,
        RED,
    )


def draw_bar(ax, x, y, w, h, frac, color=TEAL, unknown=False):
    rect(ax, x, y, w, h, edge="#888888", fill=WHITE, lw=0.55)
    if unknown:
        rect(ax, x, y, w * frac, h, edge=color, fill=color, lw=0.0)
        rect(ax, x + w * frac, y, w * (1 - frac), h, edge=LIGHT, fill=WHITE,
             lw=0.0, hatch="////")
    else:
        rect(ax, x, y, w * frac, h, edge=color, fill=color, lw=0.0)


def draw_panel_c(ax):
    x, y, w, h = 7.00, 0.68, 3.90, 6.45
    panel(ax, x, y, w, h)
    text(ax, x + 0.16, y + h - 0.44, "C. Coverage-limited A4C triage",
         size=15.0, color=DARK, weight="bold")

    tx, ty, tw, th = x + 0.20, y + 1.20, 3.70, 4.65
    rect(ax, tx, ty, tw, th, edge="#777777", fill=WHITE, lw=0.65)
    # Column boundaries.
    c1, c2 = tx + 0.74, tx + 2.15
    ax.plot([c1, c1], [ty, ty + th], color="#999999", lw=0.55)
    ax.plot([c2, c2], [ty, ty + th], color="#999999", lw=0.55)
    header_h = 0.56
    ax.plot([tx, tx + tw], [ty + th - header_h, ty + th - header_h],
            color="#999999", lw=0.55)
    row_h = (th - header_h) / 3
    for i in range(1, 3):
        yy = ty + th - header_h - i * row_h
        ax.plot([tx, tx + tw], [yy, yy], color="#999999", lw=0.55)
    text(ax, tx + 0.37, ty + th - 0.28, "Group", size=11.2, ha="center")
    text(ax, (c1 + c2) / 2, ty + th - 0.28, "Coverage", size=11.2, ha="center")
    text(ax, (c2 + tx + tw) / 2, ty + th - 0.28, "Alert signal", size=11.2, ha="center")

    row_centers = [
        ty + th - header_h - row_h * 0.5,
        ty + th - header_h - row_h * 1.5,
        ty + th - header_h - row_h * 2.5,
    ]
    groups = [("G2", RED), ("G3", TEAL), ("G4", G4)]
    for (lab, col), yy in zip(groups, row_centers):
        text(ax, tx + 0.37, yy, lab, size=15.0, color=col, weight="bold", ha="center")

    # Coverage bars and alert signals.
    text(ax, (c1 + c2) / 2, row_centers[0] + 0.24, "100%", size=10.8, ha="center")
    draw_bar(ax, c1 + 0.08, row_centers[0] - 0.18, 1.20, 0.23, 1.0, TEAL)
    text(ax, (c2 + tx + tw) / 2, row_centers[0], "46.85%", size=12.6, color=RED,
         weight="bold", ha="center")

    text(ax, (c1 + c2) / 2, row_centers[1] + 0.24, "100%", size=10.8, ha="center")
    draw_bar(ax, c1 + 0.08, row_centers[1] - 0.18, 1.20, 0.23, 1.0, TEAL)
    text(ax, (c2 + tx + tw) / 2, row_centers[1], "9.67%", size=12.6, color=TEAL,
         weight="bold", ha="center")

    text(ax, (c1 + c2) / 2 - 0.30, row_centers[2] + 0.28, "5.63%", size=9.9, ha="center")
    draw_bar(ax, c1 + 0.08, row_centers[2] - 0.15, 1.20, 0.23, 0.0563, TEAL, unknown=True)
    text(ax, c1 + 0.08, row_centers[2] - 0.42, "covered", size=7.2, ha="left")
    text(ax, c1 + 1.02, row_centers[2] - 0.42, "unknown", size=7.2, ha="left")
    text(ax, (c2 + tx + tw) / 2, row_centers[2] + 0.25, "17.60%", size=11.0,
         color=NAVY, weight="bold", ha="center")
    text(ax, (c2 + tx + tw) / 2, row_centers[2] - 0.02, "among covered;", size=7.4, ha="center")
    text(ax, (c2 + tx + tw) / 2, row_centers[2] - 0.30, "94.37%", size=9.0,
         color=MID, ha="center")
    text(ax, (c2 + tx + tw) / 2, row_centers[2] - 0.52, "unknown", size=7.4, ha="center")

    text(ax, x + w / 2, y + 0.50,
         "A4C = computational triage only;\nnot experimental validation.",
         size=10.0, color=MID, ha="center", linespacing=1.20)


def write_docs():
    CAPTION.write_text(
        "Figure 5. Dual-mode provenance triage for exploratory replacement proposals. "
        "Conservative mode routes through HGB-derived, frequency-aligned proposals, whereas exploration mode uses Borda(DE,HGB) proposals and assigns provenance strata according to overlap with HGB, Borda, and DE candidate sets. "
        "The A4C panel is a coverage-limited computational triage summary and is not experimental validation.\n",
        encoding="utf-8",
    )
    AUDIT.write_text(
        "# Figure 5 SVG replication audit\n\n"
        "- The figure is reconstructed from vector primitives: panel boxes, route boxes, arrows, provenance strata cards, table grid, coverage bars, and text.\n"
        "- No raster screenshot is embedded.\n"
        "- The provenance-set formulas use STIX-style math typography with italic set symbols and roman subscripts.\n"
        "- The A4C language is bounded as computational triage only, not experimental validation.\n"
        "- The protected displayed values are G2 alert 46.85%, G3 alert 9.67%, G4 coverage 5.63%, G4 covered alert 17.60%, and G4 unknown 94.37%.\n",
        encoding="utf-8",
    )


def main():
    setup()
    fig = plt.figure(figsize=(10.6, 7.25))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11.15)
    ax.set_ylim(0, 8.1)
    ax.axis("off")

    text(ax, 5.58, 7.78, "Dual-mode provenance triage for exploratory replacement proposals",
         size=18.0, color=DARK, weight="bold", ha="center")
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
