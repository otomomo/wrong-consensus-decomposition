#!/usr/bin/env python3
"""P3: Jensen gap / soft-vs-hard agreement as a confidence signal.

Ding's per-run data carries the full answer-count distribution of each
K=50 run (column `answer_counts`). From it we reconstruct the empirical
answer distribution p_c = count_c / K over all draws, and compare two
agreement signals for predicting whether the plurality answer is correct
(`majority_is_correct`):

 hard confidence H = max_c p_c (= self-consistency C)
 soft confidence S = sum_c p_c^2 (prob. two i.i.d. draws agree)
 Jensen gap J = S - H^2 (>=0, mass beyond the plurality)

The Jensen gap is exactly the excess agreement contributed by
non-plurality answers; a large gap means the empirical distribution is
spread across many candidates, i.e. the plurality winner is fragile (ties
to P2 champion fragility) and its self-consistency H overstates how
committed the model is.

Because only the aggregate counts exist (no per-sample softmax logits),
"soft" here is the second-order agreement statistic, not softmax-weighting
of individual samples. We are explicit that this is a comparison of two
scoring rules on the SAME hard winner, not a new voting method (see
the paper Method).

Output (results/jensen_gap.json):
 schema_version, generated_by, args, note
 cells: {
 "<model>|<benchmark>": {
 model, benchmark, n_runs, n_correct,
 auc_hard, auc_soft, # AUROC predicting majority_is_correct
 mean_jensen_gap, # mean(S - H^2) over runs
 mean_jensen_gap_incorrect, # J on majority-incorrect runs
 mean_jensen_gap_correct, # J on majority-correct runs
 mean_H, mean_S, mean_S_minus_H
 }
 }
 global: aggregated n_runs, auc_hard, auc_soft, mean_jensen_gap
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_counts(s):
 try:
 d = json.loads(s) if isinstance(s, str) else s
 except (TypeError, ValueError):
 return {}
 return {k: float(v) for k, v in d.items(}


def _auc(y, s):
 s = np.asarray(s, dtype=float)
 y = np.asarray(y, dtype=int)
 m = ~np.isnan(s)
 if m.sum( < 2 or len(np.unique(y[m])) < 2:
 return float("nan")
 return float(roc_auc_score(y[m], s[m]))


def main(:
 ap = argparse.ArgumentParser(
 ap.add_argument("--input", required=True,
 help="answer_distributions parquet (per-run answer counts)")
 ap.add_argument("--input-cases", required=True,
 help="case_results parquet (axis,condition)->(model,prompt)")
 ap.add_argument("--output", required=True)
 ap.add_argument("--min-runs", type=int, default=20)
 args = ap.parse_args(

 df = pd.read_parquet(args.input)
 # dist lacks model/prompt; map through (axis, condition) using the case
 # table (same runs; the axis->model map is condition-dependent).
 cases = pd.read_parquet(args.input_cases)
 mp = cases.drop_duplicates(["axis", "condition", "model", "prompt"])[
 ["axis", "condition", "model", "prompt"]]
 df = df.merge(mp, on=["axis", "condition"], how="left")

 rows = []
 for _, r in df.iterrows(:
 counts = parse_counts(r["answer_counts"])
 k = int(r["K"])
 if k <= 0:
 continue
 p = np.array(list(counts.values()) / k
 if p.size == 0:
 continue
 H = float(p.max()
 S = float((p * p).sum()
 J = S - H * H
 rows.append({
 "model": r["model"], "benchmark": r["benchmark"],
 "H": H, "S": S, "J": J,
 "y": int(bool(r["majority_is_correct"])),
 })
 d = pd.DataFrame(rows)

 cells = {}
 for (model, benchmark), sub in d.groupby(["model", "benchmark"]):
 y = sub["y"].to_numpy(
 cells[f"{model}|{benchmark}"] = {
 "model": model, "benchmark": benchmark,
 "n_runs": int(len(sub)),
 "n_correct": int(y.sum(),
 "auc_hard": _auc(y, sub["H"]),
 "auc_soft": _auc(y, sub["S"]),
 "mean_jensen_gap": float(sub["J"].mean(),
 "mean_jensen_gap_incorrect": float(sub.loc[y == 0, "J"].mean() if (y == 0).any( else float("nan"),
 "mean_jensen_gap_correct": float(sub.loc[y == 1, "J"].mean() if (y == 1).any( else float("nan"),
 "mean_H": float(sub["H"].mean(),
 "mean_S": float(sub["S"].mean(),
 "mean_S_minus_H": float((sub["S"] - sub["H"]).mean(),
 }

 dbig = d[d.groupby(["model", "benchmark"])["y"].transform("size") >= args.min_runs]
 out = {
 "schema_version": "1.0",
 "generated_by": "jensen_gap.py",
 "args": vars(args),
 "note": ("Hard confidence H = max_c p_c (self-consistency C); soft "
 "confidence S = sum_c p_c^2 (pairwise agreement over the "
 "empirical distribution); Jensen gap J = S - H^2 >= 0 is the "
 "agreement mass beyond the plurality, a direct measure of "
 "winner fragility. Soft is a scoring rule on the same hard "
 "winner, NOT a new voting method. p_c from answer_counts of "
 "each K=50 run; all draws including _UNPARSEABLE_ included. "
 "Aggregated at (model, benchmark); model/prompt resolved "
 "through the (axis, condition) mapping of case_results "
 "(answer_distributions carries no model/prompt column)."),
 "cells": cells,
 "global": {
 "n_runs": int(len(dbig)),
 "auc_hard": _auc(dbig["y"], dbig["H"]),
 "auc_soft": _auc(dbig["y"], dbig["S"]),
 "mean_jensen_gap": float(dbig["J"].mean(),
 "mean_jensen_gap_incorrect": float(dbig.loc[dbig["y"] == 0, "J"].mean(),
 "mean_jensen_gap_correct": float(dbig.loc[dbig["y"] == 1, "J"].mean(),
 },
 }

 with open(args.output, "w") as f:
 json.dump(out, f, indent=2)
 print(f"wrote {args.output} "
 f"({len(out['cells'])} cell(s), {out['global']['n_runs']} runs)")


if __name__ == "__main__":
 main(
