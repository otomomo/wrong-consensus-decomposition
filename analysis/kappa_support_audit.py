#!/usr/bin/env python3
"""Audit the option-space mismatch between the uniform null and the rival null.

Answers R4.3/R6: the uniform null draws over C-1 options with C = mean
per-RUN distinct-answer count (from case_results), while the rival null
draws over the per-CASE wrong-label support aggregated across the case's
other runs. For open-domain AIME these differ substantially (per-case
support >> per-run mean distinct), which explains why a support-uniform
rival (shrink lambda=0) sits below the uniform null kappa_iid.

Output: results/kappa_support_audit.json
"""
import argparse
import ast
import json

import numpy as np
import pandas as pd


def main(:
 ap = argparse.ArgumentParser(
 ap.add_argument("--input-cases", required=True)
 ap.add_argument("--input-dist", required=True)
 ap.add_argument("--output", required=True)
 args = ap.parse_args(

 cases = pd.read_parquet(args.input_cases)
 dist = pd.read_parquet(args.input_dist)
 mp = cases.drop_duplicates(["axis", "condition", "model", "prompt"])[
 ["axis", "condition", "model", "prompt"]]
 dist = dist.merge(mp, on=["axis", "condition"], how="left")
 dist["_wrong"] = ~dist["majority_is_correct"].astype(bool)

 rows = []
 for (model, benchmark, prompt), sub in dist.groupby(
 ["model", "benchmark", "prompt"], dropna=False):
 c = float(cases[(cases["model"] == model)
 & (cases["benchmark"] == benchmark)
 & (cases["prompt"] == prompt)
 ]["n_distinct_answers"].mean()
 case_agg = {}
 for cid, grp in sub.groupby("case_id"):
 agg = {}
 for e in grp["answer_counts"]:
 for l, v in ast.literal_eval(e).items(:
 agg[str(l)] = agg.get(str(l), 0.0) + float(v)
 case_agg[cid] = agg
 n_runs_per_case = sub.groupby("case_id").size(
 supports = []
 p_test_list = []
 p_lopo_list = []
 case_A_sum = sub.groupby("case_id")["A"].sum(
 for cid, grp in sub.groupby("case_id"):
 if n_runs_per_case[cid] < 2:
 continue
 gt = str(grp["ground_truth"].iloc[0])
 for sid, row in grp.iterrows(:
 if not row["_wrong"]:
 continue
 tc = ast.literal_eval(row["answer_counts"])
 support = sum(
 1 for l, v in case_agg[cid].items(
 if l != "_UNPARSEABLE_" and l != gt
 and v - float(tc.get(l, 0.0)) > 0)
 supports.append(support)
 A_test = float(row["A"])
 p_test_list.append(A_test)
 p_lopo_list.append(
 float(np.clip((float(case_A_sum[cid]) - A_test)
 / (n_runs_per_case[cid] - 1),
 1e-6, 1.0 - 1e-6)))
 rows.append({
 "model": model, "benchmark": benchmark, "prompt": prompt,
 "C_mean_per_run_distinct": c,
 "rival_support_mean": float(np.mean(supports)) if supports else np.nan,
 "rival_support_median": float(np.median(supports)) if supports else np.nan,
 "rival_support_max": float(np.max(supports)) if supports else np.nan,
 "n_test_runs": int(len(supports)),
 "p_test_mean": float(np.mean(p_test_list)) if p_test_list else np.nan,
 "p_lopo_mean": float(np.mean(p_lopo_list)) if p_lopo_list else np.nan,
 })

 out = {
 "schema_version": "1.0",
 "generated_by": "kappa_support_audit.py",
 "args": vars(args),
 "note": ("C = mean per-RUN distinct-answer count (cell scalar used by the "
 "uniform null). rival_support_* = size of the per-case wrong-label "
 "support after subtracting the test run's counts and dropping gt "
 "and _UNPARSEABLE_ (the option set the rival null draws from). "
 "p_test = test run's own accuracy (old convention); p_lopo = "
 "leave-one-out case accuracy (new convention)."),
 "cells": rows,
 }
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2)
 print(f"wrote {args.output}")


if __name__ == "__main__":
 main(
