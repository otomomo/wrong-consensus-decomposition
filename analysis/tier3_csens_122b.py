#!/usr/bin/env python3
"""C-sensitivity of the pivotal 122b-aime cell.

phi is C-invariant by construction (numerator and denominator share the
same (1-p)/(C-1) scale, which cancels in the ratio); this checks it
directly on the corrected cell: c_fixed=9 and c_fixed=20 vs
mean_distinct, n_sim=2e3 (matching the Ding csens evidence).
Writes results/tier3_csens_122b.json.
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
 sys.path.insert(0, _HERE)
from kappa_tool import load
from kappa_tool.decompose import Config, decompose

ROOT = os.path.dirname(_HERE)
CSV = "/tmp/ac_work/t3_qwen3.5-122b_aime.csv"


def main(:
 import subprocess
 subprocess.run([sys.executable, os.path.join(_HERE, "tier3_to_samples.py"),
 "--input", os.path.join(ROOT, "data", "sampled", "tier3",
 "tier3_qwen3.5:122b_aime.jsonl"),
 "--output", CSV,
 "--model", "qwen3.5-122b", "--benchmark", "aime"],
 check=True)
 out = {"generated_by": "tier3_csens_122b.py",
 "note": "C-sensitivity of the pivotal 122b-aime cell: "
 "c_fixed=9/20 vs mean_distinct; n_sim=2e3",
 "cells": {}}
 for tag, cfix in (("c_fixed_9", 9), ("c_fixed_20", 20),
 ("mean_distinct", None)):
 cfg = Config(seed=0, n_sim=2000, bootstrap=2000, c_fixed=cfix,
 c_gpqa=4)
 d = decompose(load.load_samples(CSV), cfg, per_sample=True)
 row = next(iter(d["cells"].values())
 out["cells"][tag] = {"phi": row["share_explained_mech"]}
 print(f"{tag}: phi={row['share_explained_mech']:.6f}")
 json.dump(out, open(os.path.join(ROOT, "results",
 "tier3_csens_122b.json"), "w"),
 indent=2, default=str)


if __name__ == "__main__":
 main(
