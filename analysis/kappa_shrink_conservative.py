#!/usr/bin/env python3
"""Conservative draw-coverage convention for the shrinkage appendix.

The rival numerator averages only over test runs whose simulation produced a
wrong plurality (a "draw"). The most conservative alternative counts
non-draw test runs as zero in the numerator. Under that convention the
mechanical coverage becomes phi_cons = phi * (n_draw / n_test), and the drop
is phi - phi_cons. This script derives both per cell from the committed
lambda=1 shrinkage evidence (results/kappa_rival_shrink_lam1.0.json), which
itself is produced by kappa_rival_preference.py at n_sim=2e4.

Output: results/kappa_shrink_conservative.json
"""
import argparse
import json


def main( -> None:
 ap = argparse.ArgumentParser(
 ap.add_argument("--input", required=True,
 help="results/kappa_rival_shrink_lam1.0.json")
 ap.add_argument("--output", required=True)
 args = ap.parse_args(

 with open(args.input) as f:
 cells = json.load(f)["cells"]

 rows = []
 for c in cells:
 n_test = c["n_test_runs"]
 n_draw = c["n_test_with_draw"]
 phi = c["share_explained_mech"]
 frac = n_draw / n_test if n_test else float("nan")
 rows.append({
 "model": c["model"], "benchmark": c["benchmark"],
 "prompt": c["prompt"],
 "phi_lambda1": phi,
 "n_test_runs": n_test,
 "n_test_with_draw": n_draw,
 "draw_fraction": frac,
 "phi_conservative": phi * frac if n_test else float("nan"),
 "phi_drop": phi * (1 - frac) if n_test else float("nan"),
 })

 out = {
 "schema_version": "1.0",
 "generated_by": "kappa_shrink_conservative.py",
 "args": vars(args),
 "note": ("phi_conservative = phi * (n_draw / n_test): non-draw test runs "
 "counted as zero in the rival numerator; derived from the "
 "committed lambda=1 shrinkage evidence (n_sim=2e4)."),
 "cells": rows,
 }
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2)
 print(f"wrote {args.output}: {len(rows)} cells")


if __name__ == "__main__":
 main(
