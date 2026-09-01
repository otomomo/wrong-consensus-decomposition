#!/usr/bin/env python3
"""P2: champion fragility -- plurality winner stability across independent
sampling runs.

Same (model, benchmark, case_id, prompt) has two independent K=50 runs
(condition 'a' and 'b' in Ding's per-run data). We ask: does the
plurality (majority) winner flip between the two runs? A high flip rate
means the "champion" answer is fragile -- consensus is not a stable
outcome of sampling, which compounds the low plurality-share finding:
even the winner itself is not reproducible.

Output schema (results/champion_fragility.json):
 schema_version, generated_by, args, note
 cells: {
 "<model>|<benchmark>|<prompt>": {
 n_pairs, n_flips, flip_rate, flip_lo, flip_hi,
 by_consensus_bin: [
 {lo, hi, n_pairs, flip_rate, flip_lo, flip_hi}
 ]
 }
 }

Conventions (see the paper Method):
 - champion = plurality (argmin-tie) label of the K samples of a run
 - paired within identical (model, benchmark, case_id, prompt) only
 - bootstrap B=10^4 percentile CI on flip_rate (coupled: resample pairs)
 - consensus bin = self-consistency C of run 'a', equal-quantile bins
"""
import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats


def wilson_ci(k, n, z=1.96):
 if n == 0:
 return (float("nan"), float("nan"))
 phat = k / n
 denom = 1 + z * z / n
 centre = (phat + z * z / (2 * n)) / denom
 half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
 return (centre - half, centre + half)


def bootstrap_flip_ci(flip, n_boot, seed, lo=0.025, hi=0.975):
 rng = np.random.default_rng(seed)
 n = len(flip)
 if n == 0:
 return (float("nan"), float("nan"))
 draws = rng.integers(0, n, size=(n_boot, n))
 rates = flip[draws].mean(axis=1)
 return (float(np.quantile(rates, lo)), float(np.quantile(rates, hi)))


def main(:
 ap = argparse.ArgumentParser(
 ap.add_argument("--input", required=True)
 ap.add_argument("--output", required=True)
 ap.add_argument("--seed", type=int, default=0)
 ap.add_argument("--bootstrap", type=int, default=10000)
 ap.add_argument("--n-bins", type=int, default=3)
 args = ap.parse_args(

 df = pd.read_parquet(args.input)
 df = df[["model", "benchmark", "case_id", "prompt", "condition",
 "majority_answer", "C", "A"]].copy(
 df["maj"] = df["majority_answer"].astype(str)
 df["C"] = pd.to_numeric(df["C"], errors="coerce")

 piv = df.pivot_table(
 index=["model", "benchmark", "case_id", "prompt"],
 columns="condition",
 values=["maj", "C", "A"],
 aggfunc="first",
 )
 piv = piv.dropna(subset=[("maj", "a"), ("maj", "b")]).reset_index(
 piv["flip"] = (piv[("maj", "a")] != piv[("maj", "b")]).astype(int)
 piv["C_a"] = pd.to_numeric(piv[("C", "a")], errors="coerce")

 out = {
 "schema_version": "1.0",
 "generated_by": "champion_fragility.py",
 "args": vars(args),
 "note": ("champion = plurality label of one K=50 run; paired within "
 "identical (model, benchmark, case_id, prompt); flip = "
 "champion differs between run a and run b; CI = percentile "
 "bootstrap (coupled, B=10^4); consensus bin = C of run a."),
 "cells": {},
 }

 for (model, benchmark, prompt), sub in piv.groupby(
 ["model", "benchmark", "prompt"]):
 flip = sub["flip"].to_numpy(
 n = len(flip)
 n_flips = int(flip.sum()
 rate = n_flips / n if n else float("nan")
 lo, hi = bootstrap_flip_ci(flip, args.bootstrap, args.seed)
 lo_w, hi_w = wilson_ci(n_flips, n)

 bins = []
 cvals = sub["C_a"].to_numpy(
 mask = ~np.isnan(cvals)
 if mask.sum( > 0 and args.n_bins > 1:
 # equal-quantile bins on C of run a
 qs = np.quantile(cvals[mask], np.linspace(0, 1, args.n_bins + 1))
 qs = np.unique(qs)
 idx = np.digitize(cvals, qs, right=False) - 1
 idx = np.clip(idx, 0, len(qs) - 2)
 for b in range(len(qs) - 1):
 sel = (idx == b) & mask
 bflip = flip[sel]
 bn = int(bflip.sum()
 btot = int(len(bflip))
 brate = bn / btot if btot else float("nan")
 blo, bhi = bootstrap_flip_ci(bflip, args.bootstrap, args.seed + b + 1)
 bins.append({
 "lo": float(qs[b]), "hi": float(qs[b + 1]),
 "n_pairs": btot, "n_flips": bn,
 "flip_rate": brate,
 "flip_lo": blo, "flip_hi": bhi,
 })
 else:
 btot = int(len(flip))
 bn = n_flips
 bins.append({
 "lo": float("nan"), "hi": float("nan"),
 "n_pairs": btot, "n_flips": bn,
 "flip_rate": rate, "flip_lo": lo, "flip_hi": hi,
 })

 out["cells"][f"{model}|{benchmark}|{prompt}"] = {
 "model": model, "benchmark": benchmark, "prompt": prompt,
 "n_pairs": n, "n_flips": n_flips, "flip_rate": rate,
 "flip_lo": lo, "flip_hi": hi,
 "wilson_lo": lo_w, "wilson_hi": hi_w,
 "by_consensus_bin": bins,
 }

 with open(args.output, "w") as f:
 json.dump(out, f, indent=2)
 print(f"wrote {args.output} "
 f"({len(out['cells'])} cell(s), "
 f"{sum(c['n_pairs'] for c in out['cells'].values()} pairs)")


if __name__ == "__main__":
 main(
