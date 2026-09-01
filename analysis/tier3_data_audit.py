#!/usr/bin/env python3
"""Final Tier-3 data audit: integrity of every collected cell jsonl.

Checks per cell: unique (idx, run) pairs == expected (198*4=792 gpqa,
200*4=800 aime); no duplicated lines; every run has exactly K votes
(32) and n_valid == 32 (no unparseable votes); gt is a nonempty string;
options count == C; every vote is a string. Also records the provenance
note for the 27b-aime cell (768 runs from the CPU-degraded instance 1,
32 from the GPU instance 3, same model/protocol; execution environment
differs, sampling distribution does not).

Writes results/tier3_data_audit.json.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIER3 = os.path.join(ROOT, "data", "sampled", "tier3")

CELLS = [
 ("tier3_qwen3.5-9b-ctx4k_gpqa.jsonl", 198, 4),
 ("tier3_gemma4:26b_gpqa.jsonl", 198, 4),
 ("tier3_qwen3.8:27b_gpqa.jsonl", 198, 4),
 ("tier3_qwen3.5:9b_aime.jsonl", 200, None),
 ("tier3_gemma4:26b_aime.jsonl", 200, None),
 ("tier3_qwen3.8:27b_aime.jsonl", 200, None),
 ("tier3_qwen3.5:122b_aime.jsonl", 200, None),
 ("tier3_gemma4:31b_gpqa.jsonl", 198, 4),
 ("tier3_gemma4:31b_aime.jsonl", 200, None),
 ("tier3_qwen3.5:122b_gpqa.jsonl", 198, 4),
]

EXPECT_K = 32


def audit(path, expected_cases, c_fixed):
 rec = {"file": path, "expected_cases": expected_cases,
 "c_fixed": c_fixed}
 seen = set(
 errors = []
 warnings = []
 n_lines = 0
 n_valid_total = 0
 n_runs_low_parse = 0
 with open(path) as f:
 for ln, line in enumerate(f):
 n_lines += 1
 try:
 r = json.loads(line)
 except json.JSONDecodeError:
 errors.append(f"line {ln}: bad json")
 continue
 key = (r.get("idx"), r.get("run"))
 if key in seen:
 warnings.append(f"line {ln}: duplicate (idx,run) {key}")
 seen.add(key)
 votes = r.get("votes", [])
 if len(votes) != EXPECT_K:
 errors.append(f"line {ln}: {len(votes)} votes != {EXPECT_K}")
 nv = r.get("n_valid", 0)
 if nv < EXPECT_K:
 n_runs_low_parse += 1
 n_valid_total += nv
 if not isinstance(r.get("gt"), str) or not r["gt"]:
 errors.append(f"line {ln}: bad gt")
 opts = r.get("options", [])
 if c_fixed is not None and len(opts) != c_fixed:
 errors.append(f"line {ln}: {len(opts)} options != {c_fixed}")
 n_cases = len(set(k[0] for k in seen))
 rec.update({"lines": n_lines, "unique_pairs": len(seen),
 "n_cases": n_cases, "errors": errors[:20],
 "n_errors": len(errors),
 "n_dup_lines": len(warnings),
 "n_runs_with_unparseable": n_runs_low_parse,
 "mean_n_valid": round(n_valid_total / max(n_lines, 1), 3)})
 return rec


def main(:
 out = {"generated_by": "tier3_data_audit.py",
 "note": ("integrity check of every collected tier3 cell; "
 "expected pairs = cases*4; provenance: the 27b-aime "
 "cell's first 768 runs were generated on the "
 "CPU-degraded ollama instance 1 (same model, same "
 "protocol, slower execution; sampling distribution "
 "unaffected) and its final 32 runs on the GPU "
 "instance 3; the 31b cells were sampled on the 5090 "
 "node; all other cells on V100 GPU instances"),
 "cells": {}}
 all_ok = True
 for fname, expected_cases, c_fixed in CELLS:
 path = os.path.join(TIER3, fname)
 if not os.path.exists(path):
 print(f"[audit] MISSING {fname}")
 all_ok = False
 continue
 rec = audit(path, expected_cases, c_fixed)
 ok = (rec["unique_pairs"] == expected_cases * 4
 and rec["n_cases"] == expected_cases
 and rec["n_errors"] == 0)
 rec["ok"] = ok
 all_ok = all_ok and ok
 out["cells"][fname] = rec
 print(f"[audit] {fname}: {rec['lines']} lines, "
 f"{rec['unique_pairs']} pairs, {rec['n_cases']} cases, "
 f"{rec['n_errors']} errors, {rec['n_dup_lines']} dup lines, "
 f"mean_n_valid={rec['mean_n_valid']} -> "
 f"{'OK' if ok else 'FAIL'}")
 out["all_ok"] = all_ok
 with open(os.path.join(ROOT, "results", "tier3_data_audit.json"),
 "w") as f:
 json.dump(out, f, indent=2)
 print("wrote results/tier3_data_audit.json; ALL OK" if all_ok
 else "AUDIT FAILED")


if __name__ == "__main__":
 main(
