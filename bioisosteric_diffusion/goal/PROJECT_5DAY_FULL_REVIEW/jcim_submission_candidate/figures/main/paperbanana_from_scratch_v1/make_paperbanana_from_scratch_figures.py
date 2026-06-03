#!/usr/bin/env python
"""From-scratch PaperBanana-style figure suite for JCIM submission.

This script does not edit previous SVGs. It redraws Figures 1-5 using a
single restrained JCIM visual system and preserves locked numeric evidence.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#0B3F99"
TEAL = "#087A74"
RED = "#B31F24"
SOFT_RED = "#A43A3A"
DARK = "#222222"
MID = "#666666"
LIGHT = "#D9D9D9"
GRID = "#E8E8E8"
FILL_BLUE = "#F4F8FF"
FILL_TEAL = "#F2FBFA"
FILL_RED = "#FFF6F4"
WHITE = "#FFFFFF"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def setup_canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    return fig, ax


def panel(ax, x, y, w, h, label=None, title=None):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.006",
            linewidth=0.55,
            edgecolor=LIGHT,
            facecolor=WHITE,
            zorder=0,
        )
    )
    if label:
        ax.text(x + 0.018, y + h - 0.04, label, color=NAVY, fontsize=14, fontweight="bold")
    if title:
        ax.text(
            x + 0.055,
            y + h - 0.038,
            title,
            color=DARK,
            fontsize=11,
            fontweight="bold",
            va="center",
        )


def box(ax, x, y, w, h, title="", lines=(), edge=DARK, fill=WHITE, text_color=DARK, lw=1.0, fs=8.2, bold=True):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            linewidth=lw,
            edgecolor=edge,
            facecolor=fill,
        )
    )
    cy = y + h / 2
    if title and lines:
        ax.text(x + w / 2, cy + 0.018 * len(lines), title, ha="center", va="center", color=text_color, fontsize=fs, fontweight="bold" if bold else "normal")
        for i, line in enumerate(lines):
            ax.text(x + w / 2, cy - 0.023 * (i + 1), line, ha="center", va="center", color=DARK if text_color != MID else MID, fontsize=fs - 1.3)
    elif title:
        ax.text(x + w / 2, cy, title, ha="center", va="center", color=text_color, fontsize=fs, fontweight="bold" if bold else "normal")
    return (x, y, w, h)


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.05, ms=10):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=ms),
    )


def save_fig(fig, stem: str):
    svg = OUT / f"{stem}.svg"
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return png


def draw_molecule_icon(ax, x, y, s=0.08):
    pts = []
    for i in range(6):
        import math

        a = math.pi / 6 + i * math.pi / 3
        pts.append((x + s * math.cos(a), y + s * math.sin(a)))
    ax.add_patch(patches.Polygon(pts, closed=True, fill=False, edgecolor=DARK, linewidth=1.0))
    ax.plot([x - s * 1.1, x - s * 1.55], [y, y], color=DARK, lw=1.0)
    ax.plot([x + s * 0.86, x + s * 1.35], [y + s * 0.5, y + s * 0.7], color=DARK, lw=1.0)
    ax.text(x + s * 1.43, y + s * 0.72, "Cl", fontsize=7, va="center", ha="left", color=DARK)
    ax.text(x - s * 1.7, y, "*", fontsize=12, va="center", ha="center", color=DARK)


def figure1():
    fig, ax = setup_canvas(8.2, 5.2)
    ax.text(0.5, 0.975, "Leakage-controlled closed-vocabulary fragment replacement ranking", ha="center", va="top", color=NAVY, fontsize=13.2, fontweight="bold")

    panel(ax, 0.03, 0.56, 0.94, 0.36, "A", "One query, many candidates")
    ax.text(0.11, 0.82, r"$q$ = (old fragment,", color=NAVY, fontsize=9, ha="left")
    ax.text(0.13, 0.78, "attachment signature)", color=NAVY, fontsize=8.5, ha="left")
    box(ax, 0.08, 0.65, 0.16, 0.10, edge=LIGHT, fill=WHITE, lw=0.8)
    draw_molecule_icon(ax, 0.135, 0.70, 0.035)
    ax.text(0.205, 0.70, "+", fontsize=13, ha="center", va="center", color=DARK)
    ax.scatter([0.225, 0.242, 0.242], [0.715, 0.702, 0.728], s=12, facecolors=WHITE, edgecolors=NAVY, linewidths=0.8)
    ax.plot([0.233, 0.242], [0.708, 0.702], color=DARK, lw=0.8)
    ax.plot([0.233, 0.242], [0.722, 0.728], color=DARK, lw=0.8)
    arrow(ax, 0.27, 0.70, 0.31, 0.70, color=NAVY)

    ax.add_patch(patches.FancyBboxPatch((0.33, 0.61), 0.16, 0.23, boxstyle="round,pad=0.006,rounding_size=0.008", linewidth=1.0, edgecolor=NAVY, facecolor=FILL_BLUE))
    ax.text(0.41, 0.815, "Candidate set", color=NAVY, fontsize=8.0, fontweight="bold", ha="center", va="center")
    ax.text(0.41, 0.790, "closed train-derived", color=MID, fontsize=5.8, ha="center")
    ax.text(0.41, 0.770, "~150 replacements", color=MID, fontsize=5.8, ha="center")
    for yy, label in [(0.735, "$c_1$"), (0.700, "$c_2$"), (0.665, "$c_3$"), (0.632, "..."), (0.605, "$c_{150}$")]:
        ax.text(0.355, yy, label, fontsize=7.8, ha="left", va="center", color=DARK)
    ax.text(0.405, 0.700, "*", color=NAVY, fontsize=12, ha="center", va="center")
    ax.text(0.430, 0.700, "true", color=NAVY, fontsize=6.8, ha="left", va="center")
    ax.text(0.410, 0.605, "*", color=NAVY, fontsize=12, ha="center", va="center")
    ax.text(0.430, 0.605, "true", color=NAVY, fontsize=6.8, ha="left", va="center")
    arrow(ax, 0.50, 0.70, 0.54, 0.70, color=NAVY)

    ax.add_patch(patches.FancyBboxPatch((0.56, 0.61), 0.14, 0.23, boxstyle="round,pad=0.006,rounding_size=0.008", linewidth=1.0, edgecolor=NAVY, facecolor=FILL_BLUE))
    ax.text(0.63, 0.815, "Ranked list", color=NAVY, fontsize=8.0, fontweight="bold", ha="center", va="center")
    ranked = [("1", "$c_9$"), ("2", "$c_2$  *"), ("3", "$c_{42}$"), ("...", "..."), ("10", "$c_{150}$  *")]
    for yy, (r, c) in zip([0.735, 0.700, 0.665, 0.632, 0.605], ranked):
        ax.text(0.585, yy, r, fontsize=8, ha="left", va="center", color=DARK)
        ax.text(0.62, yy, c, fontsize=8, ha="left", va="center", color=NAVY if "*" in c else DARK)
    ax.plot([0.685, 0.70], [0.735, 0.735], color=NAVY, lw=1.0)
    ax.plot([0.70, 0.70], [0.735, 0.605], color=NAVY, lw=1.0)
    ax.plot([0.685, 0.70], [0.605, 0.605], color=NAVY, lw=1.0)
    ax.text(0.712, 0.668, "Top-10", color=NAVY, fontsize=8.8, va="center")
    ax.text(0.49, 0.585, "Top-10 hit if any true replacement appears in window", color=NAVY, fontsize=7.4, ha="center")
    ax.plot([0.755, 0.755], [0.61, 0.84], color=LIGHT, lw=0.9, ls=(0, (1, 3)))
    ax.text(0.79, 0.72, "closed vocabulary\nno open-vocabulary\ngeneration", color=MID, fontsize=6.2, ha="left", va="center")

    panel(ax, 0.03, 0.09, 0.45, 0.42, "B", "Random split leakage")
    ax.text(0.18, 0.40, "Train examples", color=RED, fontsize=8.2, fontweight="bold", ha="center")
    ax.text(0.34, 0.40, "Evaluation examples", color=RED, fontsize=8.2, fontweight="bold", ha="center")
    ax.text(0.08, 0.31, "Transform\nidentity", color=RED, fontsize=7.5, fontweight="bold", ha="center")
    x0, y0, w, h = 0.13, 0.245, 0.28, 0.13
    ax.add_patch(patches.Rectangle((x0, y0), w, h, fill=False, edgecolor=LIGHT, lw=0.7))
    ax.plot([x0 + w / 2, x0 + w / 2], [y0, y0 + h], color=LIGHT, lw=0.7)
    for i, yy in enumerate([y0 + h * 2 / 3, y0 + h / 3]):
        ax.plot([x0, x0 + w], [yy, yy], color=LIGHT, lw=0.7)
    rows = [("T1", "T2"), ("T2", "T2"), ("T3", "T5")]
    for i, (a, b) in enumerate(rows):
        yy = y0 + h - (i + 0.5) * h / 3
        color = RED if i == 1 else DARK
        ax.text(x0 + w * 0.25, yy, a, ha="center", va="center", fontsize=8.5, color=color)
        ax.text(x0 + w * 0.75, yy, b, ha="center", va="center", fontsize=8.5, color=color)
    ax.add_patch(patches.Rectangle((x0, y0 + h / 3), w / 2, h / 3, fill=False, edgecolor=RED, lw=0.9))
    ax.add_patch(patches.Rectangle((x0 + w / 2, y0 + h / 3), w / 2, h / 3, fill=False, edgecolor=RED, lw=0.9))
    ax.plot([x0 + w * 0.25, x0 + w * 0.75], [y0 + h / 2, y0 + h / 2], color=RED, lw=0.8, ls=(0, (3, 2)))
    arrow(ax, 0.27, 0.235, 0.27, 0.205, color=RED, ms=9)
    ax.text(0.27, 0.178, "Shared transform identity across train/evaluation", color=RED, fontsize=7.5, ha="center")
    ax.text(0.27, 0.145, "-> leakage risk", color=RED, fontsize=10, fontweight="bold", ha="center")

    panel(ax, 0.50, 0.09, 0.47, 0.42, "C", "Transform-heldout protocol")
    ax.text(0.63, 0.39, "Train transforms", color=TEAL, fontsize=8.2, fontweight="bold", ha="center")
    ax.text(0.80, 0.39, "Blind transforms", color=TEAL, fontsize=8.2, fontweight="bold", ha="center")
    ax.text(0.535, 0.31, "Transform\nidentity", color=TEAL, fontsize=7.5, fontweight="bold", ha="center")
    x0, y0, w, h = 0.58, 0.245, 0.30, 0.13
    ax.add_patch(patches.Rectangle((x0, y0), w, h, fill=False, edgecolor=LIGHT, lw=0.7))
    ax.plot([x0 + w / 2, x0 + w / 2], [y0, y0 + h], color=TEAL, lw=0.8, ls=(0, (3, 3)))
    for i, yy in enumerate([y0 + h * 2 / 3, y0 + h / 3]):
        ax.plot([x0, x0 + w], [yy, yy], color=LIGHT, lw=0.7)
    rows = [("T1", "U1"), ("T2", "U2"), ("T3", "U3")]
    for i, (a, b) in enumerate(rows):
        yy = y0 + h - (i + 0.5) * h / 3
        ax.text(x0 + w * 0.25, yy, a, ha="center", va="center", fontsize=8.5, color=DARK)
        ax.text(x0 + w * 0.75, yy, b, ha="center", va="center", fontsize=8.5, color=DARK)
    ax.text(0.73, 0.31, "empty\nintersection", ha="center", va="center", color=TEAL, fontsize=7.5)
    arrow(ax, 0.73, 0.235, 0.73, 0.205, color=TEAL, ms=9)
    ax.text(0.73, 0.185, "Zero (old fragment, attachment signature) overlap", color=TEAL, fontsize=7.5, ha="center")
    ax.text(0.73, 0.163, "Secondary blind = primary evaluation", color=TEAL, fontsize=8.8, fontweight="bold", ha="center")
    rail_y = 0.115
    for x, lab in [(0.60, "Train"), (0.72, "Development/\ncalibration"), (0.85, "Secondary blind")]:
        box(ax, x - 0.045, rail_y, 0.09, 0.045, lab, (), edge=LIGHT, fill=WHITE, text_color=MID, lw=0.7, fs=6.8, bold=False)
    arrow(ax, 0.645, rail_y + 0.022, 0.675, rail_y + 0.022, color=TEAL, lw=0.8, ms=8)
    arrow(ax, 0.765, rail_y + 0.022, 0.805, rail_y + 0.022, color=TEAL, lw=0.8, ms=8)
    ax.text(0.725, 0.075, "no blind tuning", color=MID, fontsize=7, ha="center", style="italic")
    return save_fig(fig, "Figure_1_paperbanana_scratch")


def figure2():
    fig, ax = setup_canvas(8.0, 4.8)
    ax.text(0.5, 0.965, "Candidate-level scoring pipeline and evidence hierarchy", ha="center", va="top", color=DARK, fontsize=12.8, fontweight="bold")
    ax.text(0.035, 0.84, "Candidate-level scoring pipeline", color=DARK, fontsize=10, fontweight="bold")
    box(ax, 0.035, 0.58, 0.13, 0.10, "Query-candidate pair", ["old fragment + attachment", "candidate"], edge=LIGHT, fill=WHITE, text_color=DARK, lw=0.8, fs=7.0, bold=True)
    arrow(ax, 0.18, 0.63, 0.22, 0.63)
    ax.text(0.31, 0.79, "Base-ranker evidence", color=NAVY, fontsize=8.5, fontweight="bold", ha="center")
    base_y = [0.72, 0.66, 0.60, 0.54, 0.48]
    base_labels = ["Attachment-Frequency", "HGB", "Dual Encoder", "Borda(DE,HGB)", "MLP / Score Blend"]
    for yy, lab in zip(base_y, base_labels):
        box(ax, 0.22, yy - 0.022, 0.18, 0.044, lab, (), edge=NAVY, fill=FILL_BLUE, text_color=DARK, fs=7.2, lw=1.0, bold=False)
        ax.plot([0.40, 0.435], [yy, yy], color=MID, lw=0.8)
    ax.plot([0.435, 0.435], [base_y[-1], base_y[0]], color=MID, lw=0.8)
    arrow(ax, 0.435, 0.60, 0.49, 0.60, color=DARK)
    ax.text(0.445, 0.45, "stack", color=MID, fontsize=6.5, ha="center")
    ax.add_patch(patches.FancyBboxPatch((0.50, 0.42), 0.23, 0.37, boxstyle="round,pad=0.006,rounding_size=0.008", linewidth=1.2, edgecolor=TEAL, facecolor=FILL_TEAL))
    ax.text(0.615, 0.745, "Candidate-level feature", color=TEAL, fontsize=8.0, fontweight="bold", ha="center")
    ax.text(0.615, 0.715, "matrix", color=TEAL, fontsize=8.0, fontweight="bold", ha="center")
    feature_lines = ["model outputs", "train-derived priors", "molecular descriptors", "frozen categorical schema"]
    for i, line in enumerate(feature_lines):
        yy = 0.655 - i * 0.064
        ax.text(0.615, yy, line, fontsize=7.6, color=DARK, ha="center", va="center")
        if i < len(feature_lines) - 1:
            ax.plot([0.525, 0.705], [yy - 0.030, yy - 0.030], color=TEAL, lw=0.65)
    arrow(ax, 0.745, 0.60, 0.775, 0.60)
    box(ax, 0.79, 0.52, 0.10, 0.16, "HistGB scorer", ["candidate-level", "integration"], edge=TEAL, fill=FILL_TEAL, text_color=TEAL, lw=1.05, fs=7.8)
    arrow(ax, 0.90, 0.60, 0.93, 0.60)
    box(ax, 0.94, 0.53, 0.055, 0.14, "Final ranking", ["per-query", "candidate order"], edge=LIGHT, fill=WHITE, text_color=DARK, lw=0.8, fs=7.0)

    ax.text(0.035, 0.345, "Post-audit pruning strip", color=DARK, fontsize=11, fontweight="bold")
    y = 0.18
    box(ax, 0.06, y, 0.17, 0.11, "Initial 82-feature scorer", ["base 77-feature set", "+ prior ranks"], edge=TEAL, fill=FILL_TEAL, text_color=TEAL, lw=1.05, fs=7.7)
    box(ax, 0.095, y - 0.055, 0.10, 0.035, "prospective", (), edge=TEAL, fill=WHITE, text_color=TEAL, lw=0.9, fs=6.6)
    arrow(ax, 0.25, y + 0.055, 0.285, y + 0.055)
    box(ax, 0.30, y, 0.13, 0.11, "Secondary blind\ndiagnostics", (), edge=LIGHT, fill=WHITE, text_color=DARK, lw=0.8, fs=7.8)
    arrow(ax, 0.45, y + 0.055, 0.485, y + 0.055)
    box(ax, 0.50, y, 0.15, 0.11, "Remove prior ranks", ["non-transferable", "shortcut"], edge=RED, fill=FILL_RED, text_color=RED, lw=1.0, fs=7.7)
    box(ax, 0.525, y - 0.055, 0.10, 0.035, "removed add-on", (), edge=RED, fill=WHITE, text_color=RED, lw=0.9, fs=6.4)
    arrow(ax, 0.67, y + 0.055, 0.705, y + 0.055)
    box(ax, 0.72, y, 0.19, 0.11, "Post-audit 77-feature scorer", ["base 77-feature set"], edge=TEAL, fill=FILL_TEAL, text_color=TEAL, lw=1.05, fs=7.6)
    box(ax, 0.76, y - 0.055, 0.11, 0.035, "post-audit locked", (), edge=TEAL, fill=WHITE, text_color=TEAL, lw=0.9, fs=6.4)
    return save_fig(fig, "Figure_2_paperbanana_scratch")


def figure3():
    data = [
        ("Attachment-Frequency", 0.6019, 0.5933, 0.6104, "base"),
        ("HGB", 0.7437, 0.7356, 0.7516, "base"),
        ("Dual Encoder", 0.8055, 0.7986, 0.8122, "base"),
        ("Borda(DE,HGB)", 0.8384, 0.8321, 0.8447, "fusion"),
        ("MLP", 0.8402, 0.8339, 0.8466, "fusion"),
        ("Score Blend", 0.8558, 0.8495, 0.8621, "fusion"),
        ("82-feature scorer\n(prospective)", 0.8851, 0.8796, 0.8903, "eighty2"),
        ("77-feature scorer\n(post-audit)", 0.9243, 0.9199, 0.9287, "seventy7"),
        ("Best-of-DE+HGB\n(diagnostic)", 0.8686, None, None, "diagnostic"),
    ]
    colors = {"base": MID, "fusion": "#496A88", "eighty2": "#C9873A", "seventy7": TEAL, "diagnostic": "#B5B5B5"}
    fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=180)
    y = list(range(len(data)))[::-1]
    ax.set_title("Secondary blind Top-10 performance", loc="left", fontsize=9, fontweight="bold", pad=6)
    for yi, (lab, val, lo, hi, grp) in zip(y, data):
        if lo is None:
            ax.scatter(val, yi, s=20, facecolors=WHITE, edgecolors=colors[grp], linewidths=0.8, zorder=3)
        else:
            ax.errorbar(val, yi, xerr=[[val - lo], [hi - val]], fmt="o", color=colors[grp], ecolor=colors[grp], elinewidth=0.8, capsize=1.8, markersize=3.0, zorder=3)
    for sep in [5.5, 2.5, 0.5]:
        ax.axhline(sep, color=GRID, lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in data], fontsize=7.2)
    ax.set_xlim(0.58, 0.94)
    ax.set_xticks([0.60, 0.70, 0.80, 0.90])
    ax.set_xlabel("Secondary blind Top-10", fontsize=7.5)
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT)
    return save_fig(fig, "Figure_3_paperbanana_scratch")


def figure4():
    fig = plt.figure(figsize=(6.2, 3.0), dpi=180)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.42)
    fig.suptitle("Post-audit prior-rank pruning improves transfer", x=0.02, y=0.98, ha="left", fontsize=9.2, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0])
    labels = ["Drop prior ranks", "Drop model ranks", "Drop model scores"]
    vals = [0.0393, -0.0154, -0.0241]
    yy = [2, 1, 0]
    ax.set_title("A. Ablation of feature families", loc="left", fontsize=8.3, fontweight="bold")
    ax.axvline(0, color=MID, lw=0.7)
    ax.hlines(2, 0.0354, 0.0432, color=TEAL, lw=1.0)
    ax.plot([0.0354, 0.0354], [1.94, 2.06], color=TEAL, lw=0.8)
    ax.plot([0.0432, 0.0432], [1.94, 2.06], color=TEAL, lw=0.8)
    ax.scatter([0.0393], [2], color=TEAL, s=18, zorder=4)
    ax.text(0.0393, 1.82, "+0.0393", color=TEAL, fontsize=6.5, ha="center", va="top")
    for yv, val in zip(yy[1:], vals[1:]):
        ax.hlines(yv, val, 0, color=MID, lw=0.8)
        ax.scatter([val], [yv], color=MID, s=13, zorder=3)
        ax.text(val - 0.002, yv + 0.08, f"{val:+.4f}", color=MID, fontsize=6.3, ha="right")
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(-0.03, 0.045)
    ax.set_xticks([-0.02, 0.00, 0.02, 0.04])
    ax.set_xlabel("Delta Top-10 vs initial 82-feature scorer", fontsize=7)
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_title("B. Reliability audit", loc="left", fontsize=8.3, fontweight="bold")
    ax2.text(0.03, 0.82, "Lost cases vs Score Blend", fontsize=7.4, fontweight="bold", transform=ax2.transAxes)
    ax2.text(0.22, 0.70, "82-feature", fontsize=6.5, color=MID, ha="center", transform=ax2.transAxes)
    ax2.text(0.22, 0.62, "596", fontsize=9.0, fontweight="bold", ha="center", transform=ax2.transAxes)
    ax2.annotate("", xy=(0.48, 0.66), xytext=(0.34, 0.66), xycoords=ax2.transAxes, arrowprops=dict(arrowstyle="->", lw=0.8, color=MID))
    ax2.text(0.58, 0.62, "101", fontsize=8.2, fontweight="bold", color=SOFT_RED, ha="center", transform=ax2.transAxes)
    ax2.text(0.73, 0.70, "77-feature", fontsize=6.5, color=MID, ha="center", transform=ax2.transAxes)
    ax2.text(0.46, 0.52, "5.9x fewer", fontsize=6.6, color=MID, ha="center", transform=ax2.transAxes)
    ax2.text(0.03, 0.40, "Rescue/lost audit", fontsize=7.4, fontweight="bold", transform=ax2.transAxes)
    cols = [0.03, 0.48, 0.68, 0.88]
    headers = ["Reference", "Rescue", "Lost", "Net"]
    for x, h in zip(cols, headers):
        ax2.text(x, 0.30, h, fontsize=6.7, color=TEAL if h == "Rescue" else SOFT_RED if h == "Lost" else DARK, transform=ax2.transAxes, ha="left")
    rows = [("Score Blend", "1016", "101", "+915"), ("82-feature scorer", "626", "102", "+524")]
    for i, row in enumerate(rows):
        yrow = 0.20 - i * 0.12
        ax2.plot([0.03, 0.96], [yrow + 0.055, yrow + 0.055], transform=ax2.transAxes, color=LIGHT, lw=0.5)
        for x, text in zip(cols, row):
            color = TEAL if x == cols[1] else SOFT_RED if x == cols[2] else DARK
            ax2.text(x, yrow, text, fontsize=6.8, color=color, transform=ax2.transAxes, ha="left")
    return save_fig(fig, "Figure_4_paperbanana_scratch")


def figure5():
    fig, ax = setup_canvas(8.4, 4.8)
    ax.text(0.5, 0.965, "Dual-mode provenance triage for exploratory replacement proposals", ha="center", va="top", color=DARK, fontsize=12.4, fontweight="bold")
    xcols = [0.030, 0.345, 0.660]
    widths = [0.29, 0.29, 0.31]
    titles = ["A. Dual-mode routes", "B. Provenance strata", "C. Coverage-limited A4C triage"]
    for x, w, title in zip(xcols, widths, titles):
        panel(ax, x, 0.08, w, 0.80, None, None)
        ax.text(x + 0.015, 0.835, title, fontsize=9.6, fontweight="bold", color=DARK, ha="left", va="center")
    ax.text(0.11, 0.70, "Conservative Mode", color=NAVY, fontsize=8.2, fontweight="bold", ha="center")
    ax.text(0.11, 0.66, "HGB route", color=NAVY, fontsize=7.6, style="italic", ha="center")
    ax.text(0.24, 0.70, "Exploration Mode", color=TEAL, fontsize=8.2, fontweight="bold", ha="center")
    ax.text(0.24, 0.66, "Borda(DE,HGB) route", color=TEAL, fontsize=7.4, style="italic", ha="center")
    ax.plot([0.175, 0.175], [0.18, 0.68], color=LIGHT, lw=0.7, ls=(0, (2, 3)))
    box(ax, 0.065, 0.50, 0.10, 0.08, "HGB", (), edge=NAVY, fill=FILL_BLUE, text_color=NAVY, fs=8.8)
    arrow(ax, 0.115, 0.50, 0.115, 0.43, color=NAVY)
    box(ax, 0.055, 0.31, 0.12, 0.10, "frequency-aligned\nproposals", (), edge=NAVY, fill=FILL_BLUE, text_color=NAVY, fs=7.2, bold=False)
    box(ax, 0.205, 0.50, 0.12, 0.08, "Borda(DE,HGB)", (), edge=TEAL, fill=FILL_TEAL, text_color=TEAL, fs=7.6)
    arrow(ax, 0.265, 0.50, 0.265, 0.44, color=TEAL)
    box(ax, 0.215, 0.37, 0.10, 0.07, "exploratory\nproposals", (), edge=TEAL, fill=FILL_TEAL, text_color=TEAL, fs=7.0, bold=False)
    arrow(ax, 0.265, 0.37, 0.265, 0.31, color=TEAL)
    box(ax, 0.215, 0.24, 0.10, 0.07, "provenance\nstrata", (), edge=TEAL, fill=FILL_TEAL, text_color=TEAL, fs=7.0, bold=False)
    ax.text(0.11, 0.16, "Driven by HGB", fontsize=6.5, color=NAVY, style="italic", ha="center")
    ax.text(0.25, 0.16, "Driven by Borda(DE,HGB)", fontsize=6.3, color=TEAL, style="italic", ha="center")

    strata = [
        (0.66, "G4 shared", r"$K_{\mathrm{HGB}}\cap K_{\mathrm{Borda}}$", "consensus", TEAL, FILL_TEAL),
        (0.45, "G3 DE-elevated", r"$K_{\mathrm{Borda}}\backslash K_{\mathrm{HGB}}$", "similarity-supported\nexpansion", TEAL, FILL_TEAL),
        (0.22, "G2 Borda-emergent", r"$K_{\mathrm{Borda}}\backslash (K_{\mathrm{HGB}}\cup K_{\mathrm{DE}})$", "highest novelty", SOFT_RED, FILL_RED),
    ]
    for y, title, formula, note, col, fill in strata:
        box(ax, 0.395, y - 0.07, 0.21, 0.14, "", (), edge=col, fill=fill, lw=0.9)
        ax.text(0.50, y + 0.035, title, ha="center", va="center", fontsize=9.5, color=col, fontweight="bold")
        ax.text(0.50, y - 0.005, formula, ha="center", va="center", fontsize=7.6, color=DARK)
        ax.text(0.50, y - 0.045, note, ha="center", va="center", fontsize=7.0, color=DARK)

    # Panel C table.
    tx, ty, tw, th = 0.685, 0.21, 0.275, 0.55
    ax.add_patch(patches.Rectangle((tx, ty), tw, th, fill=False, edgecolor=LIGHT, lw=0.55))
    colw = [0.060, 0.135, 0.080]
    for xx in [tx + colw[0], tx + colw[0] + colw[1]]:
        ax.plot([xx, xx], [ty, ty + th], color=LIGHT, lw=0.5)
    for yy in [ty + th * 0.78, ty + th * 0.52, ty + th * 0.26]:
        ax.plot([tx, tx + tw], [yy, yy], color=LIGHT, lw=0.5)
    ax.text(tx + 0.030, ty + th - 0.04, "Group", fontsize=6.8, ha="center")
    ax.text(tx + 0.128, ty + th - 0.04, "Coverage", fontsize=6.8, ha="center")
    ax.text(tx + 0.235, ty + th - 0.04, "Alert signal", fontsize=6.8, ha="center")
    rows = [("G2", "100%", "46.85%", SOFT_RED, 0.66), ("G3", "100%", "9.67%", TEAL, 0.43), ("G4", "5.63%", "17.60%", TEAL, 0.20)]
    for group, cov, alert, col, cy in rows:
        ax.text(tx + 0.030, cy, group, fontsize=9.5, color=col, fontweight="bold", ha="center", va="center")
        ax.text(tx + 0.128, cy + 0.055, cov, fontsize=6.5, color=DARK, ha="center")
        barx, bary = tx + 0.082, cy - 0.005
        ax.add_patch(patches.Rectangle((barx, bary), 0.09, 0.018, fill=True, color=TEAL, alpha=0.95))
        if group == "G4":
            ax.add_patch(patches.Rectangle((barx + 0.014, bary), 0.076, 0.018, fill=False, edgecolor=LIGHT, hatch="////", lw=0.4))
            ax.text(barx, bary - 0.035, "covered", fontsize=5.5, ha="left")
            ax.text(barx + 0.09, bary - 0.035, "unknown", fontsize=5.5, ha="right")
        ax.text(tx + 0.235, cy + 0.015, alert, fontsize=7.2, color=col, fontweight="bold", ha="center")
        if group == "G4":
            ax.text(tx + 0.235, cy - 0.02, "among covered;", fontsize=5.0, color=DARK, ha="center")
            ax.text(tx + 0.235, cy - 0.055, "94.37%\nunknown", fontsize=5.1, color=DARK, ha="center")
    ax.text(0.82, 0.13, "A4C = computational triage only;\nnot experimental validation.", fontsize=6.0, color=MID, ha="center")
    return save_fig(fig, "Figure_5_paperbanana_scratch")


def trim_white(img: Image.Image, border: int = 20) -> Image.Image:
    bg = Image.new(img.mode, img.size, "white")
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if not bbox:
        return img
    left = max(0, bbox[0] - border)
    top = max(0, bbox[1] - border)
    right = min(img.width, bbox[2] + border)
    bottom = min(img.height, bbox[3] + border)
    return img.crop((left, top, right, bottom))


def make_contact_sheet(pngs):
    thumbs = []
    for p in pngs:
        im = trim_white(Image.open(p).convert("RGB"))
        im.thumbnail((880, 560), Image.LANCZOS)
        thumbs.append((p, im.copy()))
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    label_h = 36
    pad = 28
    cols, rows = 2, 3
    cell_w, cell_h = 940, 640
    sheet = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (p, im) in enumerate(thumbs):
        col, row = i % cols, i // cols
        x0 = pad + col * cell_w
        y0 = pad + row * cell_h
        draw.text((x0, y0), f"Figure {i + 1}", fill=(34, 34, 34), font=font)
        x = x0 + (cell_w - im.width) // 2
        y = y0 + label_h + (cell_h - label_h - im.height) // 2
        sheet.paste(im, (x, y))
    sheet.save(OUT / "Figures_1_to_5_paperbanana_scratch_contact_sheet.png")
    sheet.save(OUT / "Figures_1_to_5_paperbanana_scratch_contact_sheet.pdf", "PDF", resolution=150)


def write_audit():
    (OUT / "PAPERBANANA_FROM_SCRATCH_AUDIT.md").write_text(
        "# PaperBanana From-Scratch Figure Suite Audit\n\n"
        "## Scope\n\n"
        "This folder contains a from-scratch redraw of Figures 1-5 using PaperBanana-style design constraints: faithfulness, conciseness, readability, and restrained academic aesthetics. It does not edit the frozen v4 SVGs.\n\n"
        "## Key Decisions\n\n"
        "- Figure 1 focuses only on task definition and leakage-controlled evaluation.\n"
        "- Figure 2 focuses on candidate-level scoring architecture and the post-audit pruning ribbon.\n"
        "- Figure 3 preserves the locked secondary blind Top-10 values and CIs.\n"
        "- Figure 4 preserves the locked ablation and rescue/lost values.\n"
        "- Figure 5 preserves the dual-mode provenance triage and A4C claim boundary.\n\n"
        "## Locked Values Preserved\n\n"
        "- Figure 3: Attachment 0.6019, HGB 0.7437, DE 0.8055, Borda 0.8384, MLP 0.8402, Score Blend 0.8558, 82-feature 0.8851, 77-feature 0.9243, Best-of-DE+HGB diagnostic 0.8686.\n"
        "- Figure 4: Drop prior ranks +0.0393 CI [0.0354, 0.0432], Drop model ranks -0.0154, Drop model scores -0.0241; lost 596 -> 101; rescue/lost/net values 1016/101/+915 and 626/102/+524.\n"
        "- Figure 5: G2 46.85%, G3 9.67%, G4 5.63% coverage and 17.60% among covered with 94.37% unknown.\n\n"
        "## Claim Boundaries\n\n"
        "- No activity-preserving, wet-lab, expert-validation, or toxicity-verdict claim is introduced.\n"
        "- A4C remains computational triage only.\n"
        "- 77-feature scorer remains post-audit/post-selection, not fully prospective.\n"
        "- Best-of-DE+HGB remains diagnostic and is not called Oracle, ceiling, or upper bound.\n",
        encoding="utf-8",
    )


def main():
    pngs = [figure1(), figure2(), figure3(), figure4(), figure5()]
    make_contact_sheet(pngs)
    write_audit()
    for p in sorted(OUT.iterdir()):
        print(p)


if __name__ == "__main__":
    main()
