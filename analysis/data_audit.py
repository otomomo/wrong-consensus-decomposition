#!/usr/bin/env python3
"""Data-level audit quantities cited in the paper.

Computes and commits the evidence for four claims:
1. plurality tie rates per benchmark (runs whose top-2 answer counts tie)
2. number of runs whose release majority label is not the argmax of its
 own answer counts (untied)
3. per-axis consensus accuracy for the mini-ZS cells (Sec. Data)
4. runs-per-case min/max across cells

Output: results/data_audit.json
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

 # 1. tie rates (top-2 counts equal)
 tie_rates = {}
 for bench, g in dist.groupby("benchmark"):
 ties = 0
 for e in g["answer_counts"]:
 d = ast.literal_eval(e)
 vals = np.array(sorted(d.values(, reverse=True))
 if len(vals) >= 2 and vals[0] == vals[1]:
 ties += 1
 tie_rates[bench] = {"n_tied": ties, "n_runs": len(g),
 "rate": ties / len(g)}

 # 2. majority label not argmax of own counts (untied)
 bad = 0
 for _, r in dist.iterrows(:
 cnt = ast.literal_eval(r["answer_counts"])
 top = max(cnt.values()
 tops = [k for k, v in cnt.items( if v == top]
 if r["majority_answer"] not in tops:
 bad += 1

 # 3. per-axis consensus accuracy for mini-ZS
 sub = cases[(cases.model == "gpt-4.1-mini") & (cases.prompt == "zero_shot")]
 axis_acc = {}
 for bench, g in sub.groupby("benchmark"):
 for (ax, cond), gg in g.groupby(["axis", "condition"]):
 axis_acc[f"{bench}|{ax}/{cond}"] = {
 "n_runs": len(gg), "n_cases": int(gg["case_id"].nunique(),
 "consensus_acc": float(
 gg["majority_is_correct"].astype(bool).mean(),
 }

 # 4. runs per case
 rpc = cases.groupby(["model", "benchmark", "prompt",
 "case_id"]).size(

 out = {
 "schema_version": "1.0",
 "generated_by": "data_audit.py",
 "args": vars(args),
 "note": ("Data-level facts cited in Sec. Data: plurality tie rates, "
 "release majority labels that are not the argmax of their own "
 "counts, per-axis consensus accuracies for mini-ZS, and the "
 "runs-per-case range."),
 "tie_rates": tie_rates,
 "majority_not_argmax_count": bad,
 "total_runs": int(len(dist)),
 "axis_consensus_acc": axis_acc,
 "runs_per_case": {"min": int(rpc.min(), "max": int(rpc.max()},
 }
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2)
 print(f"wrote {args.output}")


if __name__ == "__main__":
 main(
