#!/usr/bin/env python3
"""Pilot-sample stability of the agreement diagnostics (pre-decision angle).

Question: how many pilot cases are needed before the empirical agreement
index kappa_emp and the mechanical coverage phi are stably estimated, and
does a cheap pilot (25 cases) order cells the same way the full dataset does?

Design (fixed before running):
 * subsample CASES (not runs) without replacement, deterministic seeds
 * kappa_emp: exact (deterministic) over all sizes [10,25,50,100,200,full]
 x 20 seeds
 * phi: reduced-precision exploratory run (n_sim=2e4, bootstrap=2e3,
 documented; NOT the canonical paper values, which use n_sim=1e5) over
 sizes [25,100,full] x 5 seeds
 * pilot-vs-full ordering: Spearman over cells of phi(pilot 25) vs
 phi(full), and of kappa_emp(pilot) vs kappa_emp(full)

Inputs: Ding per-run tables (case_results + answer_distributions), the same
frames the parity test uses. Output: results/pilot_stability.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
 sys.path.insert(0, _HERE)
from kappa_tool.decompose import Config, decompose_cell # noqa: E402

ROOT = os.path.dirname(_HERE)
DATA = os.path.join(ROOT, "data", "raw")


def load_frames(:
 df = pd.read_parquet(os.path.join(DATA, "case_results_deid.parquet"))
 df["majority_is_correct"] = df["majority_is_correct"].astype(bool)
 df["A"] = df["A"].astype(float)
 df["C"] = df["C"].astype(float)
 df["n_distinct_answers"] = df["n_distinct_answers"].astype(int)
 mp = df.drop_duplicates(["axis", "condition", "model", "prompt"])[
 ["axis", "condition", "model", "prompt"]]
 dist = pd.read_parquet(os.path.join(DATA, "answer_distributions_deid.parquet"))
 dist = dist.merge(mp, on=["axis", "condition"], how="left")
 df["_correct"] = df["majority_is_correct"]
 dist["_correct"] = dist["majority_is_correct"].astype(bool)
 return df, dist


def subsample_cases(sub, dist, size, seed):
 cases = sub["case_id"].unique(
 rng = np.random.default_rng(seed)
 keep = rng.choice(len(cases), size=min(size, len(cases)), replace=False)
 keep_ids = set(cases[keep])
 sub2 = sub[sub["case_id"].isin(keep_ids)]
 dist2 = dist[dist["case_id"].isin(keep_ids)]
 return sub2, dist2


def kappa_emp_of(sub, c_fixed=None):
 p = float(sub["A"].mean()
 benchmark = str(sub["benchmark"].iloc[0])
 cfg = Config(c_fixed=c_fixed)
 c = cfg.c_fixed if cfg.c_fixed else float(sub["n_distinct_answers"].mean()
 if c <= 1:
 c = 2.0
 denom = (1.0 - p) / (c - 1)
 aw = sub.loc[~sub["majority_is_correct"], "C"].values
 if len(aw) == 0 or denom <= 0:
 return np.nan
 return float(np.mean(aw)) / denom


def main(:
 ap = argparse.ArgumentParser(
 ap.add_argument("--output", default=os.path.join(ROOT, "results", "pilot_stability.json"))
 ap.add_argument("--kappa-sizes", default="10,25,50,100,200,full")
 ap.add_argument("--kappa-seeds", type=int, default=20)
 ap.add_argument("--phi-sizes", default="25,100,full")
 ap.add_argument("--phi-seeds", type=int, default=5)
 ap.add_argument("--cells", default=None,
 help="comma list of model|benchmark|prompt cells; default all 8")
 args = ap.parse_args(

 df, dist = load_frames(
 cells = []
 for (m, b, p), sub in df.groupby(["model", "benchmark", "prompt"], dropna=False):
 cells.append((f"{m}|{b}|{p}", sub, dist[(dist["model"] == m)
 & (dist["benchmark"] == b) & (dist["prompt"] == p)].copy())
 if args.cells:
 want = set(args.cells.split(","))
 cells = [(k, s, d) for k, s, d in cells if k in want]

 cfg = Config(n_sim=20000, bootstrap=2000, c_gpqa=4)
 out = {"generated_by": "pilot_stability.py",
 "note": "kappa_emp exact; phi exploratory (n_sim=2e4, bootstrap=2e3, "
 "documented reduced precision); cases resampled without "
 "replacement; run_key student_id",
 "cells": {}, "pilot_vs_full": {}}

 def size_val(size, sub):
 return len(sub["case_id"].unique() if size == "full" else int(size)

 for label, sub, dsub in cells:
 rec = {"kappa_sizes": {}, "phi_sizes": {}, "pilot_vs_full": {}}
 cfix = 4 if "gpqa_diamond" in label else None
 full_k = kappa_emp_of(sub, c_fixed=cfix)
 # --- kappa_emp stability (exact) ---
 for size in args.kappa_sizes.split(","):
 vals = []
 for seed in range(args.kappa_seeds):
 s2, _ = subsample_cases(sub, dsub, size_val(size, sub),
 1000 * seed + 7)
 vals.append(kappa_emp_of(s2, c_fixed=cfix))
 arr = np.array([v for v in vals if np.isfinite(v)])
 rec["kappa_sizes"][size] = {
 "mean": float(np.mean(arr)) if len(arr) else None,
 "sd": float(np.std(arr)) if len(arr) else None,
 "lo": float(np.percentile(arr, 5)) if len(arr) else None,
 "hi": float(np.percentile(arr, 95)) if len(arr) else None,
 }
 # --- phi stability (exploratory reduced precision) ---
 for size in args.phi_sizes.split(","):
 vals = []
 for seed in range(args.phi_seeds):
 s2, d2 = subsample_cases(sub, dsub, size_val(size, sub),
 2000 * seed + 13)
 if len(s2["case_id"].unique() < 3:
 vals.append(np.nan)
 continue
 rng = np.random.default_rng(4000 * seed + 3)
 row = decompose_cell(s2, d2, cfg, label, rng,
 run_key="student_id")
 vals.append(row["share_explained_mech"])
 arr = np.array([v for v in vals if np.isfinite(v)])
 rec["phi_sizes"][size] = {
 "mean": float(np.mean(arr)) if len(arr) else None,
 "sd": float(np.std(arr)) if len(arr) else None,
 "lo": float(np.percentile(arr, 5)) if len(arr) else None,
 "hi": float(np.percentile(arr, 95)) if len(arr) else None,
 }
 rec["pilot_vs_full"]["kappa_emp_full"] = full_k
 out["cells"][label] = rec
 with open(args.output, "w") as f: # incremental write (crash-safe)
 json.dump(out, f, indent=2)
 print(f"[pilot] {label}: kappa_full={full_k:.3f} "
 f"phi25={rec['phi_sizes'].get('25', {}).get('mean')} "
 f"phi100={rec['phi_sizes'].get('100', {}).get('mean')} "
 f"phi_full={rec['phi_sizes'].get('full', {}).get('mean')}",
 flush=True)

 # pilot-vs-full ordering (Spearman over cells) for kappa and phi
 from scipy import stats as sps
 for q in ("kappa_sizes", "phi_sizes"):
 if q == "kappa_sizes":
 pilot = "25"
 else:
 pilot = "25"
 if all(pilot in c[q] and c[q][pilot]["mean"] is not None
 and c[q]["full"]["mean"] is not None for c in out["cells"].values():
 x = [c[q][pilot]["mean"] for c in out["cells"].values(]
 y = [c[q]["full"]["mean"] for c in out["cells"].values(]
 rho, pv = sps.spearmanr(x, y)
 out["pilot_vs_full"][q] = {"spearman_rho": float(rho), "p": float(pv)}
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2)
 print("wrote", args.output)


if __name__ == "__main__":
 main(
