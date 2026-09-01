#!/usr/bin/env python3
"""Pilot-sample stability of the agreement diagnostics on Tier-3 cells.

Mirrors analysis/pilot_stability.py (Ding 8-cell version) but over the 5
collected Tier-3 cells (controlled V100 sampling: 4 runs/case, K=32, T=0.7).
Same design, same quantities:

 * subsample CASES (not runs) without replacement, deterministic seeds
 * kappa_emp: exact over sizes [10,25,50,100,200,full] x 20 seeds
 * phi (rival share = share_explained_mech = kappa_rival_case /
 kappa_empirical_subset, the paper's phi): reduced-precision exploratory
 (n_sim=2e4, bootstrap=2e3, documented; NOT the canonical values, which
 use n_sim=1e5) over sizes [25,100,full] x 5 seeds
 * pilot(25)-vs-full ordering: Spearman over the 6 cells for kappa_emp and
 for phi

Inputs: data/sampled/tier3/*.jsonl (committed), converted to per-sample CSVs
via tier3_to_samples.py (subprocess, committed script). Output:
results/pilot_stability_tier3.json (incremental writes, crash-safe).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
 sys.path.insert(0, _HERE)
from kappa_tool import load # noqa: E402
from kappa_tool.decompose import Config, decompose_cell # noqa: E402

ROOT = os.path.dirname(_HERE)
TIER3 = os.path.join(ROOT, "data", "sampled", "tier3")
TMP = "/tmp/ac_work"

CELLS = [
 ("tier3_qwen3.5-9b-ctx4k_gpqa.jsonl", "qwen3.5-9b-ctx4k", "gpqa_diamond", 4),
 ("tier3_gemma4:26b_gpqa.jsonl", "gemma4-26b", "gpqa_diamond", 4),
 ("tier3_qwen3.8:27b_gpqa.jsonl", "qwen3.8-27b", "gpqa_diamond", 4),
 ("tier3_qwen3.5:9b_aime.jsonl", "qwen3.5-9b", "aime", None),
 ("tier3_gemma4:26b_aime.jsonl", "gemma4-26b", "aime", None),
 ("tier3_qwen3.5:122b_aime.jsonl", "qwen3.5-122b", "aime", None),
]


def to_samples_csv(jsonl, model, benchmark):
 safe = model.replace(":", "-")
 out = os.path.join(TMP, f"pilot3_{safe}_{benchmark}.csv")
 if not os.path.exists(out):
 cmd = [sys.executable, os.path.join(_HERE, "tier3_to_samples.py"),
 "--input", os.path.join(TIER3, jsonl),
 "--output", out,
 "--model", model, "--benchmark", benchmark]
 subprocess.run(cmd, check=True)
 return out


def subsample_cases(runs, size, seed):
 cases = runs["case_id"].unique(
 rng = np.random.default_rng(seed)
 keep = rng.choice(len(cases), size=min(size, len(cases)), replace=False)
 keep_ids = set(cases[keep])
 return runs[runs["case_id"].isin(keep_ids)]


def kappa_emp_of_runs(sub, c_fixed=None):
 """Exact kappa_emp on a runs frame (no MC): mirrors decompose_cell."""
 p = float(sub["A"].mean()
 if c_fixed is not None:
 c = float(c_fixed)
 else:
 c = float(sub["n_distinct_answers"].mean()
 if c <= 1:
 c = 2.0
 denom = (1.0 - p) / (c - 1)
 aw = sub.loc[~sub["majority_is_correct"].astype(bool), "C"].values
 if len(aw) == 0 or denom <= 0:
 return np.nan
 return float(np.mean(aw)) / denom


def main(:
 ap = argparse.ArgumentParser(
 ap.add_argument("--output",
 default=os.path.join(ROOT, "results",
 "pilot_stability_tier3.json"))
 ap.add_argument("--kappa-sizes", default="10,25,50,100,200,full")
 ap.add_argument("--kappa-seeds", type=int, default=20)
 ap.add_argument("--phi-sizes", default="25,100,full")
 ap.add_argument("--phi-seeds", type=int, default=5)
 args = ap.parse_args(

 out = {"generated_by": "pilot_stability_tier3.py",
 "note": ("kappa_emp exact; phi exploratory (n_sim=2e4, bootstrap="
 "2e3, documented reduced precision); cases resampled "
 "without replacement; phi = kappa_rival_case / "
 "kappa_empirical_subset (share_explained_mech); 5 tier3 "
 "cells, 4 runs/case, K=32, T=0.7"),
 "cells": {}, "pilot_vs_full": {}}

 for jsonl, model, benchmark, cfix in CELLS:
 csv_path = to_samples_csv(jsonl, model, benchmark)
 samples = load.load_samples(csv_path)
 runs = load.aggregate_to_runs(samples)
 label = f"{model}|{benchmark}|zero_shot"
 rec = {"kappa_sizes": {}, "phi_sizes": {}, "pilot_vs_full": {}}
 cfg = Config(seed=0, n_sim=20000, bootstrap=2000, c_fixed=cfix,
 c_gpqa=4)
 full_cases, full_dist = load.split_tables(runs.reset_index(drop=True))
 full_row = decompose_cell(full_cases, full_dist, cfg, label,
 np.random.default_rng(0), run_key="run_id")
 full_k = full_row["kappa_empirical"]
 full_phi = full_row["share_explained_mech"]

 def size_val(size):
 return runs["case_id"].nunique( if size == "full" else int(size)

 for size in args.kappa_sizes.split(","):
 vals = []
 for seed in range(args.kappa_seeds):
 s2 = subsample_cases(runs, size_val(size), 1000 * seed + 7)
 vals.append(kappa_emp_of_runs(s2, c_fixed=cfix))
 arr = np.array([v for v in vals if np.isfinite(v)])
 rec["kappa_sizes"][size] = {
 "mean": float(np.mean(arr)) if len(arr) else None,
 "sd": float(np.std(arr)) if len(arr) else None,
 "lo": float(np.percentile(arr, 5)) if len(arr) else None,
 "hi": float(np.percentile(arr, 95)) if len(arr) else None,
 }

 for size in args.phi_sizes.split(","):
 vals = []
 for seed in range(args.phi_seeds):
 s2 = subsample_cases(runs, size_val(size), 2000 * seed + 13)
 if s2["case_id"].nunique( < 3:
 vals.append(np.nan)
 continue
 c2, d2 = load.split_tables(s2.reset_index(drop=True))
 row = decompose_cell(c2, d2, cfg, label,
 np.random.default_rng(4000 * seed + 3),
 run_key="run_id")
 vals.append(row["share_explained_mech"])
 arr = np.array([v for v in vals if np.isfinite(v)])
 rec["phi_sizes"][size] = {
 "mean": float(np.mean(arr)) if len(arr) else None,
 "sd": float(np.std(arr)) if len(arr) else None,
 "lo": float(np.percentile(arr, 5)) if len(arr) else None,
 "hi": float(np.percentile(arr, 95)) if len(arr) else None,
 }
 rec["pilot_vs_full"]["kappa_emp_full"] = float(full_k)
 rec["pilot_vs_full"]["phi_full"] = float(full_phi)
 out["cells"][label] = rec
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2)
 print(f"[pilot3] {label}: kappa_full={full_k:.3f} phi_full={full_phi:.3f} "
 f"phi25={rec['phi_sizes'].get('25', {}).get('mean')}",
 flush=True)

 from scipy import stats as sps
 for q in ("kappa_sizes", "phi_sizes"):
 if all("25" in c[q] and c[q]["25"]["mean"] is not None
 and c[q]["full"]["mean"] is not None
 for c in out["cells"].values():
 x = [c[q]["25"]["mean"] for c in out["cells"].values(]
 y = [c[q]["full"]["mean"] for c in out["cells"].values(]
 rho, pv = sps.spearmanr(x, y)
 out["pilot_vs_full"][q] = {"spearman_rho": float(rho),
 "p": float(pv)}
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2)
 print("wrote", args.output)


if __name__ == "__main__":
 main(
