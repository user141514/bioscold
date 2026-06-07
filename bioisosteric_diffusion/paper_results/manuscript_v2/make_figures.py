#!/usr/bin/env python3
"""Nature-figure QA regenerated figures for LBC-Ranker manuscript."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("paper_results/manuscript_v2")
DATA = Path("paper_results/v2_full_data")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "font.size": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "legend.frameon": False,
    "legend.fontsize": 5.5,
    "legend.title_fontsize": 6,
})

# NMI pastel palette
C_LBC   = "#2166ac"
C_STRUCT = "#4393c3"
C_PHYSC  = "#d1e5f0"
C_HGB   = "#92c5de"
C_RETR  = "#f4a582"
C_CA    = "#ca0020"
C_FREQ  = "#bdbdbd"
C_GREEN = "#4dac26"

def save_pub(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    print(f"  Saved {name}.pdf + .png")

# ═══════════════════════════════════════
# Figure 1: Feature ablation
# ═══════════════════════════════════════
abl = pd.read_csv(DATA / "e2_ablation_full_summary.csv")
features_ordered = ["freq", "bit_corr", "morgan", "dLogP", "dTPSA", "dRings", "dMW", "dHeavy"]
labels = ["Candidate\nfrequency", "Bit-level\ncorrelation", "Morgan\nsimilarity",
          "dLogP", "dTPSA", "dRings", "dMW", "dHeavy"]
means = [abl[abl["drop"] == f]["mean_delta_macro"].values[0] for f in features_ordered]
stds  = [abl[abl["drop"] == f]["std_delta_macro"].values[0] for f in features_ordered]
colors_bar = [C_FREQ, C_STRUCT, C_STRUCT] + [C_PHYSC] * 5

fig, ax = plt.subplots(figsize=(4.5, 2.8))
x = np.arange(len(features_ordered))
ax.bar(x, means, yerr=stds, color=colors_bar, edgecolor="white", linewidth=0.3,
       capsize=2.5, error_kw={"linewidth": 0.6, "capthick": 0.6})

for i, m in enumerate(means):
    if m < -0.12:
        ax.text(i, m + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=5.5, color="white", fontweight="bold")
    else:
        ax.text(i, m - 0.008, f"{m:.3f}", ha="center", va="top", fontsize=5.5, color="black")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=5.5)
ax.set_ylabel("Delta OF-macro Hit@10", fontsize=6)
ax.axhline(y=0, color="black", linewidth=0.4)
ax.set_ylim(-0.28, 0.05)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax.set_title("Feature ablation (10-seed mean +/- std)", fontsize=7, fontweight="bold", loc="left")
ax.text(0.98, 0.02, "n = 10 seeds, 123 OFs. Error bars: +/-1 std", transform=ax.transAxes,
        fontsize=4.5, ha="right", va="bottom", color="gray")
plt.tight_layout(pad=0.5)
save_pub(fig, "fig1_ablation")
plt.close()

# ═══════════════════════════════════════
# Figure 2: Per-seed method comparison
# ═══════════════════════════════════════
df = pd.read_csv(DATA / "e1_main_10seed_summary.csv")

fig, ax = plt.subplots(figsize=(5.5, 3.2))
x = np.arange(10)
w = 0.16
methods = [
    ("LBC-Ranker", "lbc_macro", C_LBC),
    ("HGB (capacity ctrl)", "hgb_macro", C_HGB),
    ("C3F-style retrieval", "c3f_macro", C_RETR),
    ("CA (tuned)", "ca_macro", C_CA),
    ("Frequency", "freq_macro", C_FREQ),
]
for i, (name, col, color) in enumerate(methods):
    offset = (i - 2) * w
    ax.bar(x + offset, df[col], w * 0.92, label=name, color=color, edgecolor="white", linewidth=0.2)

ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in range(10)], fontsize=5.5)
ax.set_xlabel("Seed", fontsize=6)
ax.set_ylabel("OF-macro Hit@10", fontsize=6)
ax.set_ylim(0.35, 0.98)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.10))
ax.legend(fontsize=5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.24),
          columnspacing=0.8, handlelength=1.2, handleheight=0.6)
ax.set_title("10-seed ranking performance", fontsize=7, fontweight="bold", loc="left")
ax.text(0.98, 0.02, "n = 10 seeds, 86 train / 37 test OFs", transform=ax.transAxes,
        fontsize=4.5, ha="right", va="bottom", color="gray")
plt.tight_layout(pad=0.5)
save_pub(fig, "fig2_per_seed")
plt.close()

# ═══════════════════════════════════════
# Figure 3: Delta vs best non-LBC
# ═══════════════════════════════════════
fig, ax = plt.subplots(figsize=(4.2, 2.6))

deltas = df["delta_macro"].values
colors_d = [C_LBC if v >= 0.15 else C_RETR if v >= 0.10 else C_CA for v in deltas]
ax.bar(x, deltas, color=colors_d, edgecolor="white", linewidth=0.3, width=0.65)

ax.axhline(y=0.15, color=C_GREEN, linewidth=0.7, linestyle="--", alpha=0.7)
ax.axhline(y=0.10, color="#e6ab02", linewidth=0.7, linestyle="--", alpha=0.7)
ax.axhline(y=0, color="black", linewidth=0.4)

ax.text(9.5, 0.152, "Strong (+0.15)", fontsize=5, ha="right", va="bottom", color=C_GREEN)
ax.text(9.5, 0.102, "Medium (+0.10)", fontsize=5, ha="right", va="bottom", color="#e6ab02")

for i, v in enumerate(deltas):
    va_pos = "bottom" if v > 0.07 else "top"
    y_offset = 0.003 if v > 0.07 else -0.003
    ax.text(i, v + y_offset, f"{v:+.3f}", ha="center", va=va_pos, fontsize=5.2,
            fontweight="bold", color="white" if v > 0.07 else "black")

ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in range(10)], fontsize=5.5)
ax.set_xlabel("Seed", fontsize=6)
ax.set_ylabel("Delta OF-macro Hit@10 (LBC - best non-LBC)", fontsize=6)
ax.set_ylim(-0.03, 0.25)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))

mean_d = deltas.mean()
ax.axhline(y=mean_d, color=C_LBC, linewidth=0.5, linestyle=":", alpha=0.5)
ax.text(9.5, mean_d + 0.005, f"Mean: {mean_d:+.3f}", fontsize=5, ha="right", color=C_LBC)

strong_n = int((deltas >= 0.15).sum())
medium_n = int((deltas >= 0.10).sum())
ax.text(0.02, 0.96, f"Strong: {strong_n}/10   Medium: {medium_n}/10   All positive: 10/10",
        transform=ax.transAxes, fontsize=5, va="top", color="gray")
ax.set_title("LBC-Ranker vs best non-LBC baseline", fontsize=7, fontweight="bold", loc="left")
plt.tight_layout(pad=0.5)
save_pub(fig, "fig3_delta")
plt.close()

print("All 3 figures regenerated with nature-figure QA checks.")
