#!/usr/bin/env python3
"""Continuous association of the mechanical coverage phi across the 8 cells
with difficulty (p), answer-space size (C), and the open/closed indicator.

Descriptive (n=8 cells); reported as Spearman correlations.

Output: results/phi_association.json
"""
import argparse
import json

import numpy as np
from scipy import stats


def main(:
 ap = argparse.ArgumentParser(
 ap.add_argument("--input", required=True)
 ap.add_argument("--output", required=True)
 args = ap.parse_args(

 d = json.load(open(args.input))
 cells = d["cells"]
 phi = np.array([c["share_explained_mech"] for c in cells])
 p = np.array([c["p"] for c in cells])
 C = np.array([c["C_options"] for c in cells])
 open_flag = np.array([0 if c["benchmark"] == "gpqa_diamond" else 1
 for c in cells])

 out = {
 "schema_version": "1.0",
 "generated_by": "phi_association.py",
 "args": vars(args),
 "note": ("Descriptive Spearman correlations of phi across the 8 cells "
 "(n=8, observational; the three candidate drivers are "
 "mutually confounded)."),
 "n_cells": int(len(cells)),
 "phi": [float(x) for x in phi],
 "p": [float(x) for x in p],
 "C": [float(x) for x in C],
 "open": [int(x) for x in open_flag],
 "spearman": {
 "phi_vs_p": {"rho": float(stats.spearmanr(phi, p).correlation),
 "p": float(stats.spearmanr(phi, p).pvalue)},
 "phi_vs_C": {"rho": float(stats.spearmanr(phi, C).correlation),
 "p": float(stats.spearmanr(phi, C).pvalue)},
 "phi_vs_open": {"rho": float(stats.spearmanr(phi, open_flag).correlation),
 "p": float(stats.spearmanr(phi, open_flag).pvalue)},
 },
 }
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2)
 print(f"wrote {args.output}")


if __name__ == "__main__":
 main(
