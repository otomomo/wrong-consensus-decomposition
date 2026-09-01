#!/usr/bin/env python3
"""Convert tier-3 sampled JSONL into the exact input-cases / input-dist CSV schema
used by kappa_rival_preference.py, then run the canonical script (with the
hierarchical bootstrap) once per cell and aggregate the per-cell JSON output.

This script does NOT modify the canonical analysis code: it only builds inputs
and drives `kappa_rival_preference.py` via its CLI.
"""
import ast
import glob
import json
import os
import subprocess
import sys
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IN_DIR = os.path.join(ROOT, "data", "sampled", "tier3")
OUT_DIR = os.path.join(ROOT, "results", "checkpoints")
TMP_DIR = "/tmp/ac_work/tier3_inputs"
CANON = os.path.join(HERE, "kappa_rival_preference.py")
N_SIM = 100000
N_BOOT = 10000
HIER_BOOT = 500
HIER_NSIM = 10000

CASE_COLS = [
 "student_id", "axis", "axis_ok", "meta_ok", "case_id", "benchmark",
 "condition", "model", "prompt", "K", "majority_answer", "ground_truth",
 "subject", "year", "n_distinct_answers", "n_unparseable", "C", "A",
 "R_0", "R_1", "R_2", "R_wilson", "majority_is_correct", "R_wilson_recomputed",
]
DIST_COLS = [
 "student_id", "axis", "condition", "case_id", "benchmark", "K",
 "answer_counts", "total", "ground_truth", "majority_answer",
 "majority_is_correct", "C", "A", "subject", "year", "n_unparseable",
 "n_distinct_answers",
]


def slug(model: str) -> str:
 return model.replace(":", "_").replace("/", "_").replace(" ", "")


def build_cell(jsonl_path: str, model: str, benchmark: str):
 rows_case = []
 rows_dist = []
 with open(jsonl_path) as f:
 for line in f:
 line = line.strip(
 if not line:
 continue
 d = json.loads(line)
 idx = d["idx"]
 run = d["run"]
 gt = str(d["gt"])
 votes = [str(v) for v in d["votes"]]
 total = len(votes)
 counts = Counter(votes)
 nu = counts.get("_UNPARSEABLE_", 0)
 correct_votes = counts.get(gt, 0)
 maxc = max(counts.values()
 top = sorted([k for k, v in counts.items( if v == maxc])
 majority = top[0] # argmin over ALL labels (incl. _UNPARSEABLE_), deid convention
 c_share = maxc / total
 n_distinct = len(counts) # incl. _UNPARSEABLE_ (deid convention)
 A = correct_votes / total # single-sample accuracy, NOT majority
 correct = (majority == gt)
 sid = f"S{idx:04d}_r{run}"
 cid = f"{benchmark}_{idx:05d}"
 ac = json.dumps(dict(counts))
 case_id_attrs = dict(
 student_id=sid, axis="C", axis_ok="True", meta_ok="True",
 case_id=cid, benchmark=benchmark, condition="a", model=model,
 prompt="zero_shot", K=str(total), majority_answer=majority,
 ground_truth=gt, subject="", year="",
 n_distinct_answers=str(n_distinct), n_unparseable=str(nu),
 C=f"{c_share:.10f}", A=f"{A:.10f}",
 R_0="0.0", R_1="0.0", R_2="0.0", R_wilson="0.0",
 majority_is_correct="True" if correct else "False",
 R_wilson_recomputed="0.0",
 )
 rows_case.append(case_id_attrs)
 rows_dist.append(dict(
 student_id=sid, axis="C", condition="a", case_id=cid,
 benchmark=benchmark, K=str(total), answer_counts=ac,
 total=str(total), ground_truth=gt, majority_answer=majority,
 majority_is_correct="True" if correct else "False",
 C=f"{c_share:.10f}", A=f"{A:.10f}",
 subject="", year="", n_unparseable=str(nu),
 n_distinct_answers=str(n_distinct),
 ))
 df_case = pd.DataFrame(rows_case, columns=CASE_COLS)
 df_dist = pd.DataFrame(rows_dist, columns=DIST_COLS)
 # Write PARQUET with native dtypes (bool/float/int), matching the deid
 # parquet inputs used by the gpt-4.1 batch. The majority_is_correct bool
 # is mapped from its literal string below (see loop); the canonical
 # script's own `.astype(bool)` string trap is avoided on both read and write.
 for df in (df_case, df_dist):
 # Map literal-string "True"/"False" to real Python bools. Do NOT use
 # astype(bool): on a string column it turns "False" into True (the
 # canonical script's own trap). The numeric columns hold
 # number-shaped strings and are safe to coerce via astype.
 df["majority_is_correct"] = df["majority_is_correct"].map(
 {"True": True, "False": False}
 )
 df["A"] = df["A"].astype(float)
 df["C"] = df["C"].astype(float)
 df["n_distinct_answers"] = df["n_distinct_answers"].astype(int)
 df["K"] = df["K"].astype(int)
 df["n_unparseable"] = df["n_unparseable"].astype(int)
 s = slug(model)
 case_parquet = os.path.join(TMP_DIR, f"{s}_{benchmark}_cases.parquet")
 dist_parquet = os.path.join(TMP_DIR, f"{s}_{benchmark}_dist.parquet")
 df_case.to_parquet(case_parquet, index=False)
 df_dist.to_parquet(dist_parquet, index=False)
 return case_parquet, dist_parquet


def worker(cell):
 jsonl_path, model, benchmark = cell
 s = slug(model)
 out_json = os.path.join(OUT_DIR, f"tier3_hier_{s}_{benchmark}.json")
 if os.path.exists(out_json):
 try:
 with open(out_json) as f:
 if json.load(f).get("cells"):
 print(f"SKIP {model} {benchmark} (checkpoint exists)", flush=True)
 return out_json
 except Exception:
 pass
 case_csv, dist_csv = build_cell(jsonl_path, model, benchmark)
 cmd = [
 sys.executable, CANON,
 "--input-cases", case_csv,
 "--input-dist", dist_csv,
 "--output", out_json,
 "--hierarchical",
 "--hier-bootstrap", str(HIER_BOOT),
 "--hier-n-sim", str(HIER_NSIM),
 "--bootstrap", str(N_BOOT),
 "--n-sim", str(N_SIM),
 "--seed", "0",
 "--min-wrong", "1",
 "--shrink", "1.0",
 "--tie-break", "argmin",
 "--tie-seed", "0",
 "--c-gpqa", "4",
 "--aime-c-mode", "mean_distinct",
 ]
 r = subprocess.run(cmd, capture_output=True, text=True)
 if r.returncode != 0:
 sys.stderr.write(f"FAIL {model} {benchmark}:\n{r.stderr}\n")
 return None
 return out_json


def discover_cells(:
 cells = []
 for p in sorted(glob.glob(os.path.join(IN_DIR, "tier3_*.jsonl"))):
 base = os.path.basename(p)
 if not base.endswith(".jsonl"):
 continue
 stem = base[len("tier3_"):-len(".jsonl")]
 model = None
 benchmark = None
 for bm in ("aime", "gpqa"):
 if stem.endswith("_" + bm):
 model = stem[: -(len(bm) + 1)]
 benchmark = bm
 break
 if model and benchmark:
 cells.append((p, model, benchmark))
 return cells


def main(:
 os.makedirs(TMP_DIR, exist_ok=True)
 os.makedirs(OUT_DIR, exist_ok=True)
 cells = discover_cells(
 print(f"discovered {len(cells)} tier-3 cells", flush=True)
 for c in cells:
 print(" ", c[1], c[2], flush=True)

 import concurrent.futures as cf
 results = []
 with cf.ProcessPoolExecutor(max_workers=min(len(cells), 10)) as ex:
 futs = {ex.submit(worker, c): c for c in cells}
 for fut in cf.as_completed(futs):
 c = futs[fut]
 try:
 rj = fut.result(
 if rj:
 results.append((c, rj))
 print(f"done {c[1]} {c[2]} -> {rj}", flush=True)
 except Exception as e:
 sys.stderr.write(f"EXC {c[1]} {c[2]}: {e}\n")

 # aggregate
 agg = {"schema_version": "tier3_hier_phi", "cells": []}
 for (model, benchmark, _), rj in results:
 with open(rj) as f:
 out = json.load(f)
 for row in out.get("cells", []):
 agg["cells"].append(row)
 agg_path = os.path.join(ROOT, "results", "tier3_hier_phi.json")
 with open(agg_path, "w") as f:
 json.dump(agg, f, indent=2, sort_keys=False)
 print(f"wrote {agg_path} with {len(agg['cells'])} cells", flush=True)


if __name__ == "__main__":
 main(
