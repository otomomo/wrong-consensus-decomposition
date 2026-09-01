#!/usr/bin/env python3
"""Champion-flip sensitivity: pair only A/b against B/a (drop the C/a arm).

The main champion_fragility pairing takes, per (case, prompt), the first run
of condition 'a' and the first run of condition 'b'. For gpt-4.1-mini ZS,
condition 'b' exists only under axis A, while condition 'a' exists under axes
B and C. The C/a arm is lower-accuracy on AIME (see results/data_audit.json),
so the cross-batch flip rate is recomputed on the subset where the 'a' side is
the case's first B/a run (cases without any B/a run are dropped).

Output: results/champion_axis_sensitivity.json
"""
import argparse
import json
import pandas as pd


def main( -> None:
 ap = argparse.ArgumentParser(
 ap.add_argument("--input", required=True)
 ap.add_argument("--output", required=True)
 args = ap.parse_args(

 df = pd.read_parquet(args.input)
 mini = df[(df["model"] == "gpt-4.1-mini") &
 (df["prompt"] == "zero_shot")].copy(

 cells = {}
 for benchmark, m in mini.groupby("benchmark"):
 pairs = []
 for _, g in m.groupby("case_id"):
 b = g[(g["condition"] == "b") & (g["axis"] == "A")]
 aB = g[(g["condition"] == "a") & (g["axis"] == "B")]
 if len(b) and len(aB):
 pairs.append((str(aB.iloc[0]["majority_answer"]),
 str(b.iloc[0]["majority_answer"])))
 n_flips = sum(1 for ma, mb in pairs if ma != mb)
 cells[benchmark] = {
 "n_pairs": len(pairs),
 "n_flips": n_flips,
 "flip_rate": n_flips / len(pairs) if pairs else float("nan"),
 }

 out = {
 "schema_version": "1.0",
 "generated_by": "champion_axis_sensitivity.py",
 "args": vars(args),
 "note": ("gpt-4.1-mini ZS only; pairs the case's first B/a run against "
 "its first A/b run (C/a arm dropped); cases without a B/a run "
 "are excluded."),
 "cells": cells,
 }
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2)
 print(f"wrote {args.output}: {cells}")


if __name__ == "__main__":
 main(
