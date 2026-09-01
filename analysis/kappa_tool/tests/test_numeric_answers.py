#!/usr/bin/env python3
"""Regression test: all-numeric answer columns must stay strings through
load_samples (the 2026-08-22 float64-coercion bug that zeroed the
majority-correct rate on two Tier-3 AIME cells).

Fixture: a per-sample CSV whose answer and ground_truth columns are
all numeric strings; one case's majority is the correct answer. The
aggregated majority_is_correct must be 1/2, not 0.
"""
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ANALYSIS = os.path.dirname(os.path.dirname(_HERE))
if _ANALYSIS not in sys.path:
 sys.path.insert(0, _ANALYSIS)
from kappa_tool import load # noqa: E402


def main(:
 csv_path = "/tmp/ac_work/test_numeric_answers.csv"
 rows = []
 for case in ("q1", "q2"):
 for run in (0, 1):
 if case == "q1":
 votes = ["704"] * 30 + ["705"] * 2
 else:
 votes = ["704"] * 10 + ["705"] * 22
 for v in votes:
 rows.append({"case_id": case, "run_id": f"{case}_r{run}",
 "answer": v, "option_id": 10**6,
 "is_correct": "true" if v == "704" else "false",
 "ground_truth": "704", "model": "m",
 "benchmark": "aime", "prompt": "zero_shot"})
 pd.DataFrame(rows).to_csv(csv_path, index=False)

 df = load.load_samples(csv_path)
 assert df["answer"].dtype == object, f"answer dtype {df['answer'].dtype}"
 assert list(df["answer"].iloc[:2]) == ["704", "704"], "strings corrupted"
 runs = load.aggregate_to_runs(df)
 assert runs["majority_is_correct"].sum( == 2, (
 f"expected 2 correct majorities (q1's two runs), "
 f"got {runs['majority_is_correct'].sum(}")
 assert runs.loc[runs["case_id"] == "q1", "majority_is_correct"].iloc[0]
 assert not runs.loc[runs["case_id"] == "q2", "majority_is_correct"].iloc[0]
 print("test_numeric_answers: PASS")


if __name__ == "__main__":
 main(
