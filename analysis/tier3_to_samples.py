#!/usr/bin/env python3
"""Convert Tier-3 jsonl (extract_tier3.py output) to the kappa_tool per-sample
CSV, with dedup by (idx, run) and per-question R runs preserved as run_id.

Tier-3 record: {"idx", "run", "gt", "C", "options", "votes", "n_valid"}
 votes: answer STRINGS (option text for gpqa, numeric strings for aime,
 "__UNPARSEABLE__" for failed parses)
 gt: ground-truth STRING (option text / numeric string)

Dedup: the 27b-gpqa file contains duplicate (idx, run) pairs from an early
process collision; keep the FIRST occurrence of each (idx, run).

Output columns: case_id, run_id, answer, option_id, is_correct,
 ground_truth, model, benchmark, prompt
 option_id: order-of-first-appearance among the question's OPTIONS (so the
 argmin tie-break of rule 19 works); answers not in options get
 a large sentinel (unparseable convention).
"""
import argparse
import csv
import json
import os


def main(:
 ap = argparse.ArgumentParser(
 ap.add_argument("--input", required=True)
 ap.add_argument("--output", required=True)
 ap.add_argument("--model", required=True)
 ap.add_argument("--benchmark", required=True)
 ap.add_argument("--prompt", default="zero_shot")
 args = ap.parse_args(

 seen = set(
 rows = []
 ndup = 0
 with open(args.input) as f:
 for line in f:
 rec = json.loads(line)
 key = (rec["idx"], rec["run"])
 if key in seen:
 ndup += 1
 continue
 seen.add(key)
 opt_id = {o: i for i, o in enumerate(rec["options"])}
 case_id = f"q{rec['idx']:04d}"
 run_id = f"{case_id}_r{rec['run']}"
 for v in rec["votes"]:
 if v in opt_id:
 oid = opt_id[v]
 elif v == "__UNPARSEABLE__":
 oid = 10 ** 6
 else:
 # numeric bench: answer not among options -> treat by string
 oid = 10 ** 6
 rows.append({
 "case_id": case_id, "run_id": run_id, "answer": v,
 "option_id": oid,
 "is_correct": str(v == rec["gt"]).lower(,
 "ground_truth": rec["gt"],
 "model": args.model, "benchmark": args.benchmark,
 "prompt": args.prompt,
 })
 with open(args.output, "w", newline="") as f:
 w = csv.DictWriter(f, fieldnames=["case_id", "run_id", "answer",
 "option_id", "is_correct",
 "ground_truth", "model",
 "benchmark", "prompt"])
 w.writeheader(
 w.writerows(rows)
 print(f"wrote {args.output}: {len(rows)} samples, {len(seen)} (q,run) pairs, "
 f"{ndup} duplicate lines dropped")


if __name__ == "__main__":
 main(
