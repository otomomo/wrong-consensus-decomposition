#!/usr/bin/env python3
"""Append newly completed Tier-3 cells to the canonical decomposition.

Runs tier3_kappa.py's pipeline for cells completed after the first batch
(27b-aime; later: 122b-aime, 31b cells), merging into results/tier3_kappa.json
incrementally. Canonical precision: n_sim=1e5, bootstrap=1e4, seed 0.
"""
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
 sys.path.insert(0, _HERE)
from kappa_tool import load
from kappa_tool.decompose import Config, decompose

ROOT = os.path.dirname(_HERE)
TMP = "/tmp/ac_work"

CELLS = [
 ("data/sampled/tier3/tier3_qwen3.8:27b_aime.jsonl",
 "qwen3.8-27b", "aime", None),
 ("data/sampled/tier3/tier3_qwen3.5:122b_aime.jsonl",
 "qwen3.5-122b", "aime", None),
]


def to_samples_csv(jsonl, model, benchmark):
 safe = model.replace(":", "-")
 out = os.path.join(TMP, f"t3_{safe}_{benchmark}.csv")
 cmd = [sys.executable, os.path.join(_HERE, "tier3_to_samples.py"),
 "--input", os.path.join(ROOT, jsonl),
 "--output", out,
 "--model", model, "--benchmark", benchmark]
 subprocess.run(cmd, check=True)
 return out


def main(:
 out = json.load(open(os.path.join(ROOT, "results", "tier3_kappa.json")))
 for jsonl, model, benchmark, cfix in CELLS:
 label = f"{model}|{benchmark}|zero_shot"
 if label in out["cells"]:
 print(f"[t3+] skip {label} (exists)")
 continue
 csv_path = to_samples_csv(jsonl, model, benchmark)
 cfg = Config(seed=0, n_sim=100000, bootstrap=10000, c_fixed=cfix,
 c_gpqa=4)
 d = decompose(load.load_samples(csv_path), cfg, per_sample=True)
 row = next(iter(d["cells"].values())
 out["cells"][label] = row
 print(f"[t3+] {label}: p={row['p']:.4f} "
 f"kappa_emp={row['kappa_empirical']:.3f} "
 f"kappa_rival={row['kappa_rival_case']:.3f} "
 f"phi={row['share_explained_mech']:.3f} "
 f"(n_test={row['n_test_runs']})", flush=True)
 with open(os.path.join(ROOT, "results", "tier3_kappa.json"), "w") as f:
 json.dump(out, f, indent=2, default=str)
 print("merged")


if __name__ == "__main__":
 main(
