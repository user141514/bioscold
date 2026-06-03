from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.5,
    }
)

NAVY = "#0B3F99"
TEAL = "#087A74"
RED = "#B31F24"
GRAY = "#4D4D4D"
LIGHT_GRAY = "#D9D9D9"
BLUE_FILL = "#F4F8FF"
TEAL_FILL = "#F2FBFA"
RED_FILL = "#FFF6F4"
NEUTRAL_FILL = "#FAFAFA"


def box(ax, xy, wh, text, edge=GRAY, fill=NEUTRAL_FILL, lw=1.0, size=8.5, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=edge if edge in (NAVY, TEAL, RED) else "#222222",
        fontsize=size,
        fontweight=weight,
        linespacing=1.25,
    )
    return patch


def arrow(ax, p1, p2, color=GRAY, lw=1.1, ms=11, style="-|>"):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        shrinkA=2,
        shrinkB=2,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arr)
    return arr


def label(ax, x, y, s, color="#222222", size=8.5, weight="normal", ha="center"):
    ax.text(x, y, s, ha=ha, va="center", color=color, fontsize=size, fontweight=weight)


fig, ax = plt.subplots(figsize=(7.2, 4.05), dpi=300)
ax.set_xlim(0, 1.04)
ax.set_ylim(0, 1)
ax.axis("off")

label(
    ax,
    0.03,
    0.965,
    "Candidate-level scoring pipeline and evidence hierarchy",
    color="#222222",
    size=9.6,
    weight="bold",
    ha="left",
)

# Main pipeline
query = box(
    ax,
    (0.035, 0.62),
    (0.16, 0.14),
    "Query-candidate pair\nold fragment\nattachment\ncandidate",
    edge=GRAY,
    fill=NEUTRAL_FILL,
    lw=0.9,
    size=5.9,
    weight="bold",
)

label(ax, 0.305, 0.825, "Base-ranker evidence", color=NAVY, size=7.8, weight="bold")
rankers = [
    ("Attachment-\nFrequency", 0.745),
    ("HGB", 0.675),
    ("Dual Encoder", 0.605),
    ("Borda(DE,HGB)", 0.535),
    ("MLP /\nScore Blend", 0.465),
]
ranker_boxes = []
for text, y in rankers:
    ranker_boxes.append(
        box(ax, (0.235, y), (0.17, 0.048), text, edge=NAVY, fill=BLUE_FILL, lw=0.9, size=6.8, weight="bold")
    )

matrix = box(
    ax,
    (0.515, 0.48),
    (0.235, 0.33),
    "Candidate-level\nfeature matrix\n\nmodel outputs\ntrain-derived priors\nmolecular descriptors\nfrozen categorical schema",
    edge=TEAL,
    fill=TEAL_FILL,
    lw=1.2,
    size=7.0,
    weight="bold",
)

histgb = box(
    ax,
    (0.80, 0.58),
    (0.095, 0.105),
    "HistGB\nscorer",
    edge=TEAL,
    fill=TEAL_FILL,
    lw=1.0,
    size=8,
    weight="bold",
)
ranking = box(
    ax,
    (0.92, 0.58),
    (0.07, 0.105),
    "Final\nranking",
    edge=GRAY,
    fill=NEUTRAL_FILL,
    lw=0.9,
    size=6.8,
    weight="bold",
)

arrow(ax, (0.195, 0.695), (0.235, 0.695), color=GRAY)

# Route base rankers through a small combiner node instead of overlapping fan-in.
node_x = 0.455
ax.plot([node_x, node_x], [0.489, 0.769], color=NAVY, linewidth=1.1)
for rb in ranker_boxes:
    y = rb.get_y() + rb.get_height() / 2
    arrow(ax, (rb.get_x() + rb.get_width(), y), (node_x, y), color=NAVY, lw=0.9, ms=8)
arrow(ax, (node_x, 0.625), (0.515, 0.625), color=TEAL, lw=1.1, ms=10)

arrow(ax, (0.75, 0.645), (0.80, 0.635), color=GRAY)
arrow(ax, (0.895, 0.635), (0.92, 0.635), color=GRAY)

# Audit ribbon
ax.plot([0.05, 0.965], [0.355, 0.355], color=LIGHT_GRAY, linewidth=0.8, linestyle=(0, (3, 2)))
label(ax, 0.055, 0.377, "Post-audit pruning ribbon", color=GRAY, size=6.8, weight="bold", ha="left")
arrow(ax, (0.632, 0.48), (0.632, 0.365), color=TEAL, lw=1.0, ms=9)

initial = box(
    ax,
    (0.055, 0.165),
    (0.20, 0.105),
    "Initial 82-feature scorer\nF1-F9 + categorical\n+ prior ranks",
    edge=TEAL,
    fill=TEAL_FILL,
    lw=1.0,
    size=6.2,
    weight="bold",
)
label(ax, 0.155, 0.137, "prospective", color=TEAL, size=6.2)

diag = box(
    ax,
    (0.31, 0.165),
    (0.16, 0.105),
    "Secondary blind\ndiagnostics",
    edge=GRAY,
    fill=NEUTRAL_FILL,
    lw=0.9,
    size=6.5,
    weight="bold",
)

remove = box(
    ax,
    (0.55, 0.165),
    (0.155, 0.105),
    "Remove prior ranks\nnon-transferable\nshortcut",
    edge=RED,
    fill=RED_FILL,
    lw=1.0,
    size=6.2,
    weight="bold",
)
label(ax, 0.627, 0.137, "removed add-on", color=RED, size=6.2)

post = box(
    ax,
    (0.775, 0.165),
    (0.22, 0.105),
    "Post-audit 77-feature scorer\nF1-F9 + categorical",
    edge=TEAL,
    fill=TEAL_FILL,
    lw=1.0,
    size=6.0,
    weight="bold",
)
label(ax, 0.885, 0.137, "post-audit locked", color=TEAL, size=6.2)

arrow(ax, (0.255, 0.218), (0.31, 0.218), color=GRAY)
arrow(ax, (0.475, 0.218), (0.55, 0.218), color=RED)
arrow(ax, (0.705, 0.218), (0.775, 0.218), color=TEAL)

label(ax, 0.52, 0.045, "Score Blend is included as base-ranker evidence; A4C triage is outside this scoring pipeline.", color="#666666", size=5.8)

for ext in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"Figure_2_paperbanana_cleaned_v3.{ext}", bbox_inches="tight", dpi=600)

plt.close(fig)
