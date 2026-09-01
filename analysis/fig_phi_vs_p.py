#!/usr/bin/env python3
"""Figure: mechanical coverage phi vs single-sample accuracy p (18 cells).

All 18 cells (8 gpt-4.1 from results/kappa_rival_preference.json, 10
Tier-3 from results/tier3_kappa.json), x = single-sample accuracy p,
y = mechanical coverage phi. Encoding: marker shape = model family (all markers hollow, uniform
style); edge color = benchmark (blue = GPQA-Diamond, orange = AIME;
Wong colorblind-safe). The sampling-protocol difference (gpt-4.1 from
the Ding pipeline, K=50; open-weights from the controlled Tier-3
protocol, K=32) is disclosed in the caption text, not visually
encoded, because it is a deterministic function of family here. No
connectivity lines, no regression: the comparison is a family-level
contrast at matched accuracy, not a functional relationship. The
pivotal point (qwen3.5-122b AIME) and the minimum-phi gpt-4.1 AIME
point (gpt-4.1-mini zero-shot) are text-labeled. Dashed reference at
phi = 1. CIs are in the paper's tables (tab:tier3, tab:kappa), not
drawn here.

Legend design: family entries use a NEUTRAL black edge and hollow fill
(shape encodes family only), so the edge color is reserved for the
benchmark channel; benchmark entries are filled color dots.

Output: paper/fig2_phi_vs_p.pdf
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
 "pdf.fonttype": 42,
 "ps.fonttype": 42,
 "font.family": "sans-serif",
 "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
 "font.size": 8,
 "axes.linewidth": 0.6,
 "axes.labelsize": 8,
 "xtick.labelsize": 7.5,
 "ytick.labelsize": 7.5,
 "lines.linewidth": 1.0,
 "lines.markersize": 4,
 "savefig.bbox": "tight",
 "savefig.pad_inches": 0.03,
})

BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#555555"
BLACK = "#111111"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main(:
 ap = argparse.ArgumentParser(
 ap.add_argument("--outdir", default=os.path.join(ROOT, "paper"))
 args = ap.parse_args(

 riv = json.load(open(os.path.join(
 ROOT, "results", "kappa_rival_preference.json")))
 t3 = json.load(open(os.path.join(
 ROOT, "results", "tier3_kappa.json")))

 cells = [] # (family, model, bench, prompt, p, phi, protocol)
 for row in riv["cells"]:
 cells.append(("gpt-4.1", row["model"], row["benchmark"],
 row["prompt"], float(row["p"]),
 float(row["share_explained_mech"]), "ding"))
 for label, row in t3["cells"].items(:
 model, bench, prompt = label.split("|")
 fam = ("qwen3.5" if model.startswith("qwen3.5")
 else "qwen3.8" if model.startswith("qwen3.8")
 else "gemma4")
 cells.append((fam, model, bench, prompt, float(row["p"]),
 float(row["share_explained_mech"]), "tier3"))

 marks = {"gpt-4.1": "o", "qwen3.5": "s", "qwen3.8": "D", "gemma4": "^"}

 fig, ax = plt.subplots(figsize=(3.6, 2.6))
 for fam, model, bench, prompt, p, phi, proto in cells:
 color = BLUE if bench in ("gpqa_diamond", "gpqa") else ORANGE
 ax.scatter(p, phi, marker=marks[fam], s=30, facecolors="white",
 edgecolors=color, linewidths=0.9, zorder=3)

 # label the pivotal point and the minimum-phi gpt-4.1 AIME point
 def label_exact(model, bench, prompt, dx, dy, text):
 for fam, m, b, pr, p, phi, proto in cells:
 if m == model and b == bench and pr == prompt:
 ax.annotate(text, (p, phi), textcoords="offset points",
 xytext=(dx, dy), fontsize=6.5, color="#222222",
 ha="left", zorder=4)
 return

 label_exact("qwen3.5-122b", "aime", "zero_shot", 5, 5, "q3.5-122B")
 label_exact("gpt-4.1-mini", "aime", "zero_shot", 5, -8, "4.1-mini-ZS")
 label_exact("gpt-4.1", "aime", "zero_shot", 5, 4, "4.1-ZS")

 ax.axhline(1.0, color="black", lw=0.6, ls="--", zorder=2)
 ax.text(0.505, 1.012, "$\\phi=1$", fontsize=7, color="black",
 ha="right")
 ax.set_xlabel("single-sample accuracy $p$")
 ax.set_ylabel("mechanical coverage $\\phi$")
 ax.set_xlim(0, 0.6)
 ax.set_ylim(0.55, 1.14)
 ax.spines["top"].set_visible(False)
 ax.spines["right"].set_visible(False)
 ax.tick_params(direction="in", width=0.6)

 from matplotlib.lines import Line2D
 handles = [
 Line2D([], [], marker="o", color=BLACK, mfc="white", ms=3.2,
 label="gpt-4.1", ls=""),
 Line2D([], [], marker="s", color=BLACK, mfc="white", ms=3.2,
 label="Qwen3.5", ls=""),
 Line2D([], [], marker="D", color=BLACK, mfc="white", ms=3.2,
 label="Qwen3.8", ls=""),
 Line2D([], [], marker="^", color=BLACK, mfc="white", ms=3.2,
 label="Gemma4", ls=""),
 Line2D([], [], marker="o", color=BLUE, mfc=BLUE, ms=3.2,
 label="GPQA", ls=""),
 Line2D([], [], marker="o", color=ORANGE, mfc=ORANGE, ms=3.2,
 label="AIME", ls=""),
 ]
 leg = ax.legend(handles=handles, fontsize=6, frameon=True,
 loc="lower right", bbox_to_anchor=(0.995, 0.06),
 handletextpad=0.25, ncol=2, columnspacing=0.6,
 borderaxespad=0.15, labelspacing=0.2)
 leg.get_frame(.set_edgecolor(BLACK)
 leg.get_frame(.set_linewidth(0.5)
 leg.get_frame(.set_linestyle("--")
 leg.get_frame(.set_facecolor("white")

 out = os.path.join(args.outdir, "fig2_phi_vs_p.pdf")
 fig.savefig(out)
 print(f"wrote {out} with {len(cells)} cells")


if __name__ == "__main__":
 main(
