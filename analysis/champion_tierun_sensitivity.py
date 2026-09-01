#!/usr/bin/env python3
"""Sensitivity of the champion flip rate to the run chosen per condition.

For cases with multiple runs in a condition, the main analysis takes the
first run (aggfunc='first'). This script recomputes the overall flip rate
with the last run and with a random run per condition, to bound the
arbitrariness of the choice.

Output: results/champion_tierun_sensitivity.json
"""
import argparse
import json

import numpy as np
import pandas as pd


def flip_rate(piv):
 return float((piv["a"] != piv["b"]).mean(), int(len(piv))


def main(:
 ap = argparse.ArgumentParser(
 ap.add_argument("--input", required=True)
 ap.add_argument("--output", required=True)
 ap.add_argument("--seed", type=int, default=0)
 args = ap.parse_args(

 df = pd.read_parquet(args.input)
 df = df[["model", "benchmark", "case_id", "prompt", "condition",
 "majority_answer"]]
 df["maj"] = df["majority_answer"].astype(str)
 rng = np.random.default_rng(args.seed)

 out = {
 "schema_version": "1.0",
 "generated_by": "champion_tierun_sensitivity.py",
 "args": vars(args),
 "note": ("Flip rate under first/last/random run per condition for "
 "cases with multiple runs per condition."),
 "cells": {},
 }
 for bench in ["gpqa_diamond", "aime"]:
 s = df[(df.model == "gpt-4.1-mini") & (df.benchmark == bench)]
 piv_first = s.pivot_table(index=["case_id", "prompt"],
 columns="condition", values="maj",
 aggfunc="first").dropna(
 piv_last = s.pivot_table(index=["case_id", "prompt"],
 columns="condition", values="maj",
 aggfunc="last").dropna(
 piv_rnd = s.pivot_table(index=["case_id", "prompt"],
 columns="condition", values="maj",
 aggfunc=lambda x: x.iloc[
 rng.integers(0, len(x))]).dropna(
 r_first, n = flip_rate(piv_first)
 r_last, _ = flip_rate(piv_last)
 r_rnd, _ = flip_rate(piv_rnd)
 out["cells"][bench] = {
 "flip_first": r_first, "flip_last": r_last, "flip_random": r_rnd,
 "n_pairs": n,
 "max_abs_diff": float(max(abs(r_first - r_last),
 abs(r_first - r_rnd))),
 }
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2)
 print(f"wrote {args.output}")


if __name__ == "__main__":
 main(
