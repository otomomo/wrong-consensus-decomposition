"""CLI for kappa_tool.

Two input modes (see decompose.py docstring for the parity contract):

 per-sample mode (default):
 python -m kappa_tool.cli --input samples.parquet --output out.json
 Accepts a per-sample long table (case_id, run_id, answer,
 is_correct|ground_truth, optional model/benchmark/prompt). A/C are
 re-derived by re-counting -- a warning is printed because the rival family
 is then a fresh MC estimate, not bit-locked to the committed JSON.

 parity / DB mode:
 python -m kappa_tool.cli --input-cases data/raw/case_results_deid.parquet \
 --input-dist data/raw/answer_distributions_deid.parquet --output out.json
 Consumes the Ding per-run tables with their STORED A/C floats and mirrors
 kappa_rival_preference.main's data prep exactly, so the rival family is
 bit-exact against results/kappa_rival_preference.json (same seed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
 sys.path.insert(0, _HERE)
# sibling validated scripts live one directory up
_ANALYSIS = os.path.dirname(_HERE)
if _ANALYSIS not in sys.path:
 sys.path.insert(0, _ANALYSIS)

from .decompose import Config, decompose, decompose_cell # noqa: E402


def parse_args( -> argparse.Namespace:
 p = argparse.ArgumentParser(description=__doc__,
 formatter_class=argparse.RawDescriptionHelpFormatter)
 src = p.add_mutually_exclusive_group(required=True)
 src.add_argument("--input", help="per-sample long table (.parquet/.csv/.json/.jsonl)")
 src.add_argument("--input-cases", help="[parity/DB mode] Ding case_results table")
 p.add_argument("--input-dist", help="[parity/DB mode] Ding answer_distributions table "
 "(required with --input-cases)")
 p.add_argument("--output", required=True)
 p.add_argument("--seed", type=int, default=0)
 p.add_argument("--bootstrap", type=int, default=10000)
 p.add_argument("--n-sim", type=int, default=100000)
 p.add_argument("--share-n-sim", type=int, default=4000)
 p.add_argument("--share-bootstrap", type=int, default=2000)
 p.add_argument("--min-wrong", type=int, default=1)
 p.add_argument("--shrink", type=float, default=1.0)
 p.add_argument("--tie-break", choices=["argmin", "random"], default="argmin")
 p.add_argument("--tie-seed", type=int, default=0)
 p.add_argument("--c-gpqa", type=int, default=4)
 p.add_argument("--aime-c-mode", choices=["mean_distinct", "max_distinct", "fixed"],
 default="mean_distinct")
 p.add_argument("--aime-c-fixed", type=float, default=None)
 return p.parse_args(


def _read(path: str) -> pd.DataFrame:
 if path.endswith(".parquet"):
 return pd.read_parquet(path)
 if path.endswith(".csv"):
 return pd.read_csv(path)
 raise ValueError(f"unsupported table format: {path} (use .parquet or .csv)")


def _cfg_from(args: argparse.Namespace) -> Config:
 return Config(
 seed=args.seed, bootstrap=args.bootstrap, n_sim=args.n_sim,
 share_n_sim=args.share_n_sim, share_bootstrap=args.share_bootstrap,
 min_wrong=args.min_wrong, shrink=args.shrink,
 tie_break=args.tie_break, tie_seed=args.tie_seed,
 c_gpqa=args.c_gpqa, aime_c_mode=args.aime_c_mode,
 aime_c_fixed=args.aime_c_fixed)


def run_parity_mode(args: argparse.Namespace) -> dict:
 """Mirror kappa_rival_preference.main's data prep, then decompose per cell.

 Single Generator seeded args.seed, advanced across cells in sorted
 (model, benchmark, prompt) order exactly as the validated script does, so
 the rival family is bit-exact against the committed JSON.
 """
 if not args.input_dist:
 raise SystemExit("--input-cases requires --input-dist")
 cfg = _cfg_from(args)
 rng = np.random.default_rng(cfg.seed)

 df = _read(args.input_cases)
 df["majority_is_correct"] = df["majority_is_correct"].astype(bool)
 df["A"] = df["A"].astype(float)
 df["C"] = df["C"].astype(float)
 df["n_distinct_answers"] = df["n_distinct_answers"].astype(int)

 # map (axis, condition) -> (model, prompt) from case_results
 mp = df.drop_duplicates(["axis", "condition", "model", "prompt"])[
 ["axis", "condition", "model", "prompt"]]
 dist = _read(args.input_dist)
 dist = dist.merge(mp, on=["axis", "condition"], how="left")

 if cfg.tie_break == "random":
 import ast
 tie_rng = np.random.default_rng(cfg.tie_seed)

 def _maj_random(entry):
 d = ast.literal_eval(entry)
 if not d:
 return None
 maxc = max(d.values()
 top = [k for k, v in d.items( if v == maxc]
 if len(top) > 1:
 return top[int(tie_rng.integers(len(top)))]
 return top[0]

 dist["_maj"] = [_maj_random(e) for e in dist["answer_counts"]]
 dist["_correct"] = (dist["_maj"].astype(str) ==
 dist["ground_truth"].astype(str)).astype(bool)
 correct_map = dist.set_index("student_id")["_correct"].to_dict(
 df["_correct"] = df["student_id"].map(correct_map)
 else:
 df["_correct"] = df["majority_is_correct"]
 dist["_correct"] = dist["majority_is_correct"].astype(bool)

 cells = {}
 for (model, benchmark, prompt), sub in df.groupby(
 ["model", "benchmark", "prompt"], dropna=False):
 label = "|".join([model, benchmark, prompt])
 dsub = dist[(dist["model"] == model) & (dist["benchmark"] == benchmark) &
 (dist["prompt"] == prompt)].copy(
 cells[label] = decompose_cell(sub, dsub, cfg, label, rng,
 run_key="student_id")
 return {"schema_version": "1.0", "generated_by": "kappa_tool.cli (parity mode)",
 "config": cfg.__dict__, "cells": cells}


def run_per_sample_mode(args: argparse.Namespace) -> dict:
 from . import load as _load
 print("WARNING: per-sample mode re-derives A/C by re-counting; the rival "
 "family is a fresh MC estimate (not bit-locked to the committed "
 "JSON, which used Ding's stored floats).", file=sys.stderr)
 df = _load.load_samples(args.input)
 return decompose(df, _cfg_from(args), per_sample=True)


def main( -> int:
 args = parse_args(
 if args.input_cases:
 out = run_parity_mode(args)
 else:
 out = run_per_sample_mode(args)
 os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2, sort_keys=False)
 print(f"wrote {args.output}")
 return 0


if __name__ == "__main__":
 raise SystemExit(main()
