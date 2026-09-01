#!/usr/bin/env python3
"""Convert Ollama self-consistency JSONL to the kappa_tool per-sample table.

Input format (data/sampled/per_llm/llm_sc_*.jsonl), one line per question:
 {"idx": int, "gt": int, "C": int, "question": str, "options": [str, ...],
 "votes": [int, ...], "n_valid": int}
Each question carries ONE run of K votes (option ids 0..C-1); ``gt`` is the
correct option id. There is no per-case replication, so the leak-free rival
null is undefined on this arm (as disclosed in the paper, Appendix B).

Output: a per-sample long CSV with columns
 case_id, run_id, answer, is_correct, ground_truth[, model, benchmark, prompt]
compatible with kappa_tool.load.aggregate_to_runs (and the paper's
conventions: answers are option STRINGS from ``options``; ties broken by
argmin per rule 19 downstream).

Usage:
 python ollama_to_samples.py --input X.jsonl --output X_samples.csv \
 [--model qwen3.5-9b] [--benchmark mmlu] [--prompt ctx4k]
"""
import argparse
import json


def main( -> None:
 ap = argparse.ArgumentParser(
 ap.add_argument("--input", required=True)
 ap.add_argument("--output", required=True)
 ap.add_argument("--model", default="qwen3.5-9b")
 ap.add_argument("--benchmark", default=None,
 help="defaults to the dataset token parsed from the filename")
 ap.add_argument("--prompt", default="ctx4k")
 ap.add_argument("--keep-invalid", action="store_true",
 help="keep failed-parse votes as _UNPARSEABLE_ (Ding "
 "convention); default drops them (original Qwen-arm "
 "convention, matching llm_selfconsistency.py)")
 args = ap.parse_args(

 bench = args.benchmark
 if bench is None:
 name = args.input.rsplit("/", 1)[-1]
 for cand in ("mmlu_pro", "mmlu", "aime", "gpqa"):
 if cand in name:
 bench = cand
 break
 if bench is None:
 raise SystemExit("cannot parse benchmark from filename; use --benchmark")

 rows = []
 with open(args.input) as f:
 for line in f:
 rec = json.loads(line)
 case_id = f"q{rec['idx']:04d}"
 gt = rec["gt"]
 options = [str(o) for o in rec["options"]]
 K = len(rec["votes"])
 run_id = f"{case_id}_r0"
 gt_ans = options[gt] if 0 <= gt < len(options) else str(gt)
 for v in rec["votes"]:
 if 0 <= v < len(options):
 ans = options[v]
 oid = v
 ok = str(v == gt).lower(
 elif args.keep_invalid:
 # Ding convention: failed parses stay in the vote pool
 ans = "_UNPARSEABLE_"
 oid = 10 ** 6
 ok = "false"
 else:
 # original Qwen-arm convention: drop failed parses
 continue
 rows.append({
 "case_id": case_id, "run_id": run_id, "answer": ans,
 "option_id": oid,
 "is_correct": ok,
 "ground_truth": gt_ans,
 "model": args.model, "benchmark": bench,
 "prompt": args.prompt,
 })
 import csv
 with open(args.output, "w", newline="") as f:
 w = csv.DictWriter(f, fieldnames=list(rows[0].keys())
 w.writeheader(
 w.writerows(rows)
 n_cases = len({r["case_id"] for r in rows})
 print(f"wrote {args.output}: {len(rows)} samples, {n_cases} cases, "
 f"K={len(rows)//max(n_cases,1)}, benchmark={bench}")


if __name__ == "__main__":
 main(
