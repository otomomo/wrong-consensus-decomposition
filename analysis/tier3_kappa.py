#!/usr/bin/env python3
"""Tier-3 kappa decomposition over all 10 collected cells (canonical precision).

Self-contained: reads the committed raw votes (data/sampled/tier3/*.jsonl),
converts each to the per-sample CSV via tier3_to_samples.py (dedup by (idx, run)
+ option-order tie-break, rule 19), and decomposes at canonical precision. One
command regenerates results/tier3_kappa.json (tab:tier3, the canonical evidence
for the Tier-3 table). No /tmp dependency, no manual append steps.

n_sim=1e5, bootstrap=1e4, seed 0; c_fixed=4 for gpqa_diamond, mean_distinct for
aime; 27b-gpqa input deduped upstream in tier3_to_samples.py. Per-cell RNG is
seeded inside decompose, so cell order does not affect the values.

Usage:
 python3 analysis/tier3_kappa.py # regenerate results/tier3_kappa.json
 python3 analysis/tier3_kappa.py --check # diff vs existing, DO NOT write
 python3 analysis/tier3_kappa.py --output PATH # write to a custom path
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
 sys.path.insert(0, _HERE)
from kappa_tool import load
from kappa_tool.decompose import Config, decompose

ROOT = os.path.dirname(_HERE)

# (jsonl_relpath, model_label, benchmark_label, c_fixed) — order matches the
# committed results/tier3_kappa.json cell order. benchmark_label is "gpqa_diamond"
# or "aime" (feeds both the decompose label and the C rule).
CELLS = [
 ("data/sampled/tier3/tier3_qwen3.5-9b-ctx4k_gpqa.jsonl", "qwen3.5-9b-ctx4k", "gpqa_diamond", 4),
 ("data/sampled/tier3/tier3_gemma4:26b_gpqa.jsonl", "gemma4-26b", "gpqa_diamond", 4),
 ("data/sampled/tier3/tier3_qwen3.8:27b_gpqa.jsonl", "qwen3.8-27b", "gpqa_diamond", 4),
 ("data/sampled/tier3/tier3_qwen3.5:9b_aime.jsonl", "qwen3.5-9b", "aime", None),
 ("data/sampled/tier3/tier3_gemma4:26b_aime.jsonl", "gemma4-26b", "aime", None),
 ("data/sampled/tier3/tier3_qwen3.8:27b_aime.jsonl", "qwen3.8-27b", "aime", None),
 ("data/sampled/tier3/tier3_qwen3.5:122b_aime.jsonl", "qwen3.5-122b", "aime", None),
 ("data/sampled/tier3/tier3_gemma4:31b_gpqa.jsonl", "gemma4-31b", "gpqa_diamond", 4),
 ("data/sampled/tier3/tier3_gemma4:31b_aime.jsonl", "gemma4-31b", "aime", None),
 ("data/sampled/tier3/tier3_qwen3.5:122b_gpqa.jsonl", "qwen3.5-122b", "gpqa_diamond", 4),
]

NOTE = ("n_sim=1e5, bootstrap=1e4, seed 0, c_fixed=4 for gpqa, mean_distinct for "
 "aime; controlled V100 sampling, 4 runs/case, K=32, T=0.7; 27b-gpqa "
 "deduped; self-contained driver over committed data/sampled/tier3/*.jsonl "
 "(no /tmp dependency)")


def to_samples_csv(jsonl_abs: str, out_csv: str, model: str, benchmark: str) -> None:
 cmd = [sys.executable, os.path.join(_HERE, "tier3_to_samples.py"),
 "--input", jsonl_abs, "--output", out_csv,
 "--model", model, "--benchmark", benchmark]
 r = subprocess.run(cmd, capture_output=True, text=True)
 if r.returncode != 0:
 raise RuntimeError(f"tier3_to_samples failed for {jsonl_abs}:\n{r.stderr}")


def compute_cells(tmpdir: str) -> dict:
 cells = {}
 for jsonl_rel, model, bench, cfix in CELLS:
 jsonl_abs = os.path.join(ROOT, jsonl_rel)
 out_csv = os.path.join(tmpdir, f"t3_{model}_{bench}.csv")
 to_samples_csv(jsonl_abs, out_csv, model, bench)
 cfg = Config(seed=0, n_sim=100000, bootstrap=10000, c_fixed=cfix, c_gpqa=4)
 d = decompose(load.load_samples(out_csv), cfg, per_sample=True)
 row = next(iter(d["cells"].values())
 cells[f"{model}|{bench}|zero_shot"] = row
 print(f"[t3] {model}|{bench}: p={row['p']:.4f} "
 f"phi={row['share_explained_mech']:.4f}", flush=True)
 return cells


def main( -> int:
 ap = argparse.ArgumentParser(
 description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
 ap.add_argument("--output",
 default=os.path.join(ROOT, "results", "tier3_kappa.json"))
 ap.add_argument("--check", action="store_true",
 help="diff the regenerated cells against --output and DO NOT write")
 args = ap.parse_args(

 tmpdir = tempfile.mkdtemp(prefix="t3canon_")
 try:
 cells = compute_cells(tmpdir)
 finally:
 shutil.rmtree(tmpdir, ignore_errors=True)

 if args.check:
 try:
 ref = json.load(open(args.output))["cells"]
 except FileNotFoundError:
 print(f"no existing {args.output}; nothing to compare")
 return 0
 bad = 0
 for label, row in cells.items(:
 if label not in ref:
 print(f" NEW {label} (not in existing)"); bad += 1; continue
 r = ref[label]
 ok = (abs(row["p"] - r["p"]) < 1e-9 and
 abs(row["share_explained_mech"] - r["share_explained_mech"]) < 1e-9)
 print(f" [{'OK ' if ok else 'CHANGED'}] {label}: "
 f"p={row['p']:.4f}/{r['p']:.4f} "
 f"phi={row['share_explained_mech']:.4f}/{r['share_explained_mech']:.4f}")
 if not ok:
 bad += 1
 extra = [l for l in ref if l not in cells]
 if extra:
 print(" in existing but not regenerated:", extra); bad += len(extra)
 print(f"CHECK: {bad} mismatch(es) out of {len(cells)} regenerated cells")
 return 1 if bad else 0

 out = {"generated_by": "tier3_kappa.py", "note": NOTE, "cells": cells}
 os.makedirs(os.path.dirname(args.output), exist_ok=True)
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2, default=str)
 print(f"wrote {args.output} ({len(cells)} cells)")
 return 0


if __name__ == "__main__":
 sys.exit(main()
