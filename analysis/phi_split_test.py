#!/usr/bin/env python3
"""R5: formal test of the GPQA-vs-AIME split in the mechanical share phi.

Reads the committed rival-preference JSON. Null: benchmark labels are
exchangeable across the 8 cells, so the observed mean difference
d = mean(phi_GPQA) - mean(phi_AIME) is tested by Monte Carlo permutation
(10^4 shuffles of the benchmark labels). Also reports a coupled
bootstrap CI (B=10^4, resampling cells within benchmark) for d.

Output: results/phi_split_test.json
"""
import argparse
import json

import numpy as np


def main(:
 ap = argparse.ArgumentParser(
 ap.add_argument("--input", required=True)
 ap.add_argument("--output", required=True)
 ap.add_argument("--seed", type=int, default=0)
 ap.add_argument("--n-perm", type=int, default=10000)
 ap.add_argument("--bootstrap", type=int, default=10000)
 args = ap.parse_args(

 d = json.load(open(args.input))
 cells = d["cells"]
 phi = np.array([c["share_explained_mech"] for c in cells])
 bench = np.array(["g" if c["benchmark"] == "gpqa_diamond" else "a"
 for c in cells])
 order = [f"{c['model']}|{c['benchmark']}|{c['prompt']}" for c in cells]

 d_obs = float(phi[bench == "g"].mean( - phi[bench == "a"].mean()

 rng = np.random.default_rng(args.seed)
 # exact permutation enumeration over all C(8,4)=70 labelings
 from itertools import combinations
 idx_g = [i for i in range(len(bench)) if bench[i] == "g"]
 n_exceed = 0
 all_d = []
 for combo in combinations(range(len(bench)), len(idx_g)):
 mask = np.zeros(len(bench), dtype=bool)
 mask[list(combo)] = True
 dd = phi[mask].mean( - phi[~mask].mean(
 all_d.append(dd)
 if abs(dd) >= abs(d_obs):
 n_exceed += 1
 p_exact = n_exceed / len(all_d)

 # permutation test (benchmark labels shuffled across the 8 cells)
 count = 0
 for _ in range(args.n_perm):
 perm = rng.permutation(bench)
 dd = phi[perm == "g"].mean( - phi[perm == "a"].mean(
 if abs(dd) >= abs(d_obs):
 count += 1
 p_perm = (count + 1) / (args.n_perm + 1)

 # coupled bootstrap: resample cells within benchmark, recompute d
 phi_g, phi_a = phi[bench == "g"], phi[bench == "a"]
 diffs = np.empty(args.bootstrap)
 for b in range(args.bootstrap):
 gs = rng.choice(len(phi_g), size=len(phi_g), replace=True)
 as_ = rng.choice(len(phi_a), size=len(phi_a), replace=True)
 diffs[b] = phi_g[gs].mean( - phi_a[as_].mean(
 d_lo, d_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

 out = {
 "schema_version": "1.0",
 "generated_by": "phi_split_test.py",
 "args": vars(args),
 "note": ("Permutation test of the benchmark split of phi across the 8 "
 "cells (n=4 per benchmark; exchangeability of benchmark labels "
 "is the null): p_permutation = Monte Carlo (1e4 shuffles), "
 "p_exact_enumeration = exhaustive over all C(8,4)=70 "
 "labelings. Coupled bootstrap resamples cells within "
 "benchmark. Small n: treat p as descriptive."),
 "cells": order,
 "phi": [float(x) for x in phi],
 "benchmark": [b for b in bench],
 "d_obs": d_obs,
 "p_permutation": p_perm,
 "p_exact_enumeration": p_exact,
 "n_exact_labelings": len(all_d),
 "d_ci_bootstrap": [d_lo, d_hi],
 }
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2)
 print(f"wrote {args.output}: d={d_obs:.3f} "
 f"p_perm={p_perm:.4f} d_ci=[{d_lo:.3f},{d_hi:.3f}]")


if __name__ == "__main__":
 main(
