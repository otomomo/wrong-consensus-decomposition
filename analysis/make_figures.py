#!/usr/bin/env python3
"""Paper figures from committed evidence JSONs (no new numbers computed).

Publication-style configuration (publication-figures-guide standards):
Wong (2011) colorblind-safe palette, serif fonts matching the LaTeX
body (Times), embedded TrueType fonts (pdf.fonttype=42), 0.5-0.8pt
spines, top/right spines removed, vector PDF output.

Figure 1: forest plot of phi = kappa_rival/kappa_emp^(t) per cell with
case-clustered bootstrap CIs, split by benchmark (from
results/kappa_rival_preference.json).
Figure 2: difficulty-binned voting gap (consensus - single-sample
accuracy) per cell (from results/backfire_repro.json), quintile bins on
per-case p.

Outputs: paper/fig1_phi_forest.pdf, paper/fig3_backfire_gap.pdf
"""
import argparse
import json

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

BLUE = "#0072B2" # Wong palette (GPQA)
ORANGE = "#D55E00" # Wong palette (AIME)
GRAY = "#666666"

ORDER = [
 ("gpt-4.1", "gpqa_diamond", "zero_shot"),
 ("gpt-4.1-mini", "gpqa_diamond", "chain_of_thought"),
 ("gpt-4.1-mini", "gpqa_diamond", "zero_shot"),
 ("gpt-4.1-nano", "gpqa_diamond", "zero_shot"),
 ("gpt-4.1", "aime", "zero_shot"),
 ("gpt-4.1-mini", "aime", "chain_of_thought"),
 ("gpt-4.1-mini", "aime", "zero_shot"),
 ("gpt-4.1-nano", "aime", "zero_shot"),
]


def short(cell):
 m, b, p = cell
 prompt = "CoT" if p == "chain_of_thought" else "ZS"
 model = {"gpt-4.1": "4.1", "gpt-4.1-mini": "mini",
 "gpt-4.1-nano": "nano"}[m]
 bench = "GPQA" if b == "gpqa_diamond" else "AIME"
 return f"{model} {bench}-{prompt}"


def main(:
 ap = argparse.ArgumentParser(
 ap.add_argument("--rival", required=True)
 ap.add_argument("--backfire", required=True)
 ap.add_argument("--outdir", default="paper")
 args = ap.parse_args(

 rival = json.load(open(args.rival))
 cells = {(c["model"], c["benchmark"], c["prompt"]): c
 for c in rival["cells"]}

 # ---- Figure 1: phi forest plot ----
 rows = [cells[k] for k in ORDER]
 labels = [short(k) for k in ORDER]
 phi = np.array([r["share_explained_mech"] for r in rows])
 lo = np.array([r["share_explained_mech_ci_clustered"][0] for r in rows])
 hi = np.array([r["share_explained_mech_ci_clustered"][1] for r in rows])
 colors = [BLUE if k[1] == "gpqa_diamond" else ORANGE for k in ORDER]

 fig, ax = plt.subplots(figsize=(3.3, 2.9))
 y = np.arange(len(labels))[::-1]
 # group separator between GPQA (top 4) and AIME (bottom 4)
 ax.axhline(3.5, color="0.8", lw=0.6)
 ax.errorbar(phi, y, xerr=[phi - lo, hi - phi], fmt="o",
 color=GRAY, ecolor="0.6", elinewidth=0.8, ms=3.5,
 zorder=3)
 for yi, (p_, c_) in zip(y, zip(phi, colors)):
 ax.plot(p_, yi, "o", color=c_, ms=3.5, zorder=4)
 ax.axvline(1.0, color="0.35", ls="--", lw=0.8)
 ax.set_yticks(y)
 ax.set_yticklabels(labels)
 ax.set_xlabel(r"mechanical coverage $\phi=\Gamma_{\rm rival}/\Gamma_{\rm emp}^{(t)}$")
 ax.set_ylim(-0.6, len(labels) - 0.4)
 ax.set_xlim(0.45, 1.05)
 ax.text(0.455, 3.92, "GPQA-Diamond", color=BLUE, fontsize=7.5,
 va="center", ha="left")
 ax.text(0.455, 3.08, "AIME", color=ORANGE, fontsize=7.5,
 va="center", ha="left")
 ax.spines["top"].set_visible(False)
 ax.spines["right"].set_visible(False)
 fig.tight_layout(
 fig.savefig(f"{args.outdir}/fig1_phi_forest.pdf")
 plt.close(fig)

 # ---- Figure 2: difficulty-binned voting gap ----
 bf = json.load(open(args.backfire))
 fig, axes = plt.subplots(2, 4, figsize=(6.7, 3.0), sharex=True,
 sharey=True)
 for i, (ax, key) in enumerate(zip(axes.ravel(, ORDER)):
 c = bf["cells"]["|".join(key)]
 bins = c["difficulty_bins"]
 mids = [b["case_p_mid"] for b in bins]
 gaps = [b["case_gap"] for b in bins]
 los = [b["case_gap_ci_coupled"][0] for b in bins]
 his = [b["case_gap_ci_coupled"][1] for b in bins]
 ax.axhline(0, color="0.6", lw=0.6)
 ax.errorbar(mids, gaps, yerr=[np.array(gaps) - np.array(los),
 np.array(his) - np.array(gaps)],
 fmt="o-", ms=2.5, lw=0.9, color=BLUE,
 elinewidth=0.7, capsize=1.5)
 ax.set_title(f"({'abcdefgh'[i]}) {short(key)}", fontsize=7.5,
 pad=2)
 ax.spines["top"].set_visible(False)
 ax.spines["right"].set_visible(False)
 ax.tick_params(labelsize=6.5)
 if key[1] == "gpqa_diamond":
 ax.set_xlim(0, 1)
 for ax in axes[:, 0]:
 ax.set_ylabel("voting gap", fontsize=8)
 for ax in axes[1, :]:
 ax.set_xlabel("per-case $p$ (bin mid)", fontsize=8)
 for ax in axes[0, :]:
 ax.tick_params(labelbottom=False)
 for ax in axes[:, 1:].ravel(:
 ax.tick_params(labelleft=False)
 fig.align_ylabels(axes[:, 0])
 fig.tight_layout(
 fig.savefig(f"{args.outdir}/fig3_backfire_gap.pdf")
 plt.close(fig)
 print(f"wrote {args.outdir}/fig1_phi_forest.pdf, "
 f"{args.outdir}/fig3_backfire_gap.pdf")


if __name__ == "__main__":
 main(
