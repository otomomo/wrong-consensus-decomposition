#!/usr/bin/env python3
"""
analysis/extract_tier3.py — Tier-3 controlled reproduction sampling
========================================================================
Samples multi-run-per-question self-consistency data on GPQA-Diamond and
AIME with multiple model families, mirroring the Ding 2026 protocol
(R independent runs per case, K samples per run) so the FULL kappa_tool
pipeline (including the leak-free rival null) applies.

Design (fixed before running; do not tune after seeing results):
 benchmarks:
 gpqa : GPQA-Diamond, official gated release (Idavidrein/gpqa,
 gpqa_diamond.csv), C=4 multiple choice, 198 questions
 aime : AIME 1983-2024 (di-zhang-fdu/AIME_1983_2024), open numeric
 answers, 200 problems stratified by decade, seed 42
 models : qwen3.5-9b-ctx4k, qwen3.8:27b, gemma4:26b, gemma4:31b,
 qwen3.5:122b (per-cell choice)
 protocol: R=4 runs/question x K=32 samples/run, temperature 0.7,
 num_predict capped at 256 tokens
 prompt : same zero-shot option/number format across all cells
 parsing : letters A-D for gpqa (explicit "answer is X" preferred, else
 last standalone letter); last-number extraction for aime.
 Correct numeric votes are canonicalized to the gt string;
 wrong votes keep their raw string form (float-form wrong
 answers are genuinely non-integer, ~0.2-2% of wrong votes,
 not label-splitting artifacts; verified 2026-08-22).
 Unparseable outputs are kept as "__UNPARSEABLE__".

Output (resumable JSONL, one line per (question, run)):
 data/sampled/tier3/tier3_{model}_{bench}.jsonl
 per line: {"idx": q_idx, "run": r, "gt": str, "C": int, "options": [...],
 "votes": [...], "n_valid": int}

Usage (on a GPU node, under tmux/nohup):
 python3 extract_tier3.py --bench gpqa --model qwen3.5-9b-ctx4k \
 --ollama_url http://127.0.0.1:11434
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve(.parent.parent
OUT_ROOT = REPO_ROOT / "data" / "sampled" / "tier3"
LETTERS = "ABCDEFGHIJ"

BENCH = {
 "gpqa": {
 # official gated GPQA-Diamond release (198 q), exact Ding benchmark;
 # requires HF_TOKEN of an account that accepted the GPQA terms
 # (passed via env HF_TOKEN, never committed)
 "repo": "Idavidrein/gpqa", "file": "gpqa_diamond.csv",
 "n": 198, "mode": "letter", "gated": True,
 },
 "aime": {
 # full AIME 1983-2024, 933 problems, numeric answers (aligns Ding's
 # open-domain benchmark); ungated mirror, CSV not parquet
 "repo": "di-zhang-fdu/AIME_1983_2024",
 "file": "AIME_Dataset_1983_2024.csv", "n": 200, "mode": "number",
 },
}


def load_questions(bench, cache, log):
 from huggingface_hub import hf_hub_download
 cfg = BENCH[bench]
 cache.mkdir(parents=True, exist_ok=True)
 local = REPO_ROOT / "data" / "tier3_static" / cfg["file"]
 if local.exists(:
 p = str(local)
 log(f"{bench}: using local file {p} (no network)")
 else:
 p = hf_hub_download(cfg["repo"], cfg["file"], repo_type="dataset",
 cache_dir=str(cache))
 rows = []
 if str(p).endswith(".csv") and cfg["mode"] == "letter":
 import csv as _csv
 recs = list(_csv.DictReader(open(p)))
 log(f"{bench}: {len(recs)} CSV rows (gpqa-diamond) from {p}")
 for i, rec in enumerate(recs):
 correct = str(rec["Correct Answer"]).strip(
 wrongs = [str(rec[f"Incorrect Answer {j}"]).strip(
 for j in (1, 2, 3)]
 opts = [correct] + wrongs
 rnd = random.Random(1000 + i)
 rnd.shuffle(opts)
 rows.append({"q": str(rec["Question"]), "options": opts,
 "gt": correct, "C": len(opts)})
 return rows
 if str(p).endswith(".csv"):
 import csv as _csv
 recs = list(_csv.DictReader(open(p)))
 n = cfg["n"]
 if len(recs) > n: # stratified by decade, deterministic seed
 by_year = {}
 for rec in recs:
 by_year.setdefault(rec["Year"][:3] + "0s", []).append(rec)
 rnd = random.Random(42)
 picked = []
 per = max(1, n // len(by_year))
 for decade in sorted(by_year):
 pool = by_year[decade]
 picked += rnd.sample(pool, min(per, len(pool)))
 picked = picked[:n]
 else:
 picked = recs
 log(f"{bench}: {len(picked)}/{len(recs)} CSV rows (stratified) from {p}")
 for rec in picked:
 ans = rec["Answer"].strip(.replace(",", "")
 rows.append({"q": rec["Question"], "options": [],
 "gt": ans, "C": None})
 return rows
 import pyarrow.parquet as pq
 t = pq.read_table(p)
 n = cfg["n"]
 idxs = list(range(t.num_rows))
 if t.num_rows > n:
 idxs = sorted(random.Random(42).sample(range(t.num_rows), n))
 for i in idxs:
 if cfg["mode"] == "letter":
 correct = str(t.column("Correct Answer").take([i])[0].as_py()
 wrongs = [str(t.column(f"Incorrect Answer {j}").take([i])[0].as_py()
 for j in (1, 2, 3)]
 opts = [correct] + wrongs
 # deterministic option order per question (seed = row index)
 rnd = random.Random(1000 + i)
 rnd.shuffle(opts)
 rows.append({"q": str(t.column("Question").take([i])[0].as_py(),
 "options": opts, "gt": correct, "C": len(opts)})
 log(f"{bench}: {len(rows)} questions from {p}")
 return rows


def build_prompt(row):
 if row["options"]:
 opt_txt = "\n".join(f"{LETTERS[i]}. {o}" for i, o in
 enumerate(row["options"]))
 return (f"Question: {row['q']}\nOptions:\n{opt_txt}\n\n"
 f"Answer with ONLY the letter of the correct option. "
 f"Do not explain.")
 return (f"Question: {row['q']}\n\n"
 f"Answer with ONLY the final numeric value. Do not explain.")


def parse_answer(text, row):
 t = text.strip(
 if row["options"]:
 m = re.search(r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Ja-j])", t,
 re.IGNORECASE)
 if m:
 i = LETTERS.index(m.group(1).upper()
 return row["options"][i] if i < len(row["options"]) else "__UNPARSEABLE__"
 ls = re.findall(r"\b([A-Ja-j])\b", t)
 if ls:
 i = LETTERS.index(ls[-1].upper()
 return row["options"][i] if i < len(row["options"]) else "__UNPARSEABLE__"
 return "__UNPARSEABLE__"
 # numeric: last number in the response, normalized
 nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", t)
 if not nums:
 return "__UNPARSEABLE__"
 return nums[-1].replace(",", "")


def norm_num(s):
 try:
 return float(str(s).replace(",", ""))
 except ValueError:
 return None


def sample_once(row, model, url, temperature, log):
 import requests
 payload = {"model": model, "prompt": build_prompt(row), "stream": False,
 "keep_alive": "10h", "think": False,
 "options": {"temperature": temperature, "num_predict": 256}}
 try:
 r = requests.post(f"{url}/api/generate", json=payload, timeout=600)
 r.raise_for_status(
 ans = parse_answer(r.json(["response"], row)
 # numeric benchmark: normalize model answer vs gt for exact match
 if not row["options"] and ans != "__UNPARSEABLE__":
 v = norm_num(ans)
 g = norm_num(row["gt"])
 if v is not None and g is not None and v == g:
 ans = row["gt"] # canonical string so equality is exact
 return ans
 except Exception as e: # noqa: BLE001
 log(f"sample error: {e}")
 return "__UNPARSEABLE__"


def main(:
 ap = argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--bench", required=True, choices=list(BENCH))
 ap.add_argument("--model", required=True)
 ap.add_argument("--runs", type=int, default=4)
 ap.add_argument("--k", type=int, default=32)
 ap.add_argument("--temperature", type=float, default=0.7)
 ap.add_argument("--ollama_url", default="http://127.0.0.1:11434")
 ap.add_argument("--workers", type=int, default=4)
 args = ap.parse_args(

 ts = time.strftime("%Y%m%d_%H%M%S")
 logdir = REPO_ROOT / "data" / "logs"
 logdir.mkdir(parents=True, exist_ok=True)
 import logging
 logging.basicConfig(level=logging.INFO,
 format="%(asctime)s [%(levelname)s] %(message)s",
 handlers=[logging.FileHandler(
 logdir / f"tier3_{ts}.log"), logging.StreamHandler(])
 log = logging.getLogger(.info

 OUT_ROOT.mkdir(parents=True, exist_ok=True)
 cache = REPO_ROOT / "data" / "tier3_dl"
 questions = load_questions(args.bench, cache, log)
 out_path = OUT_ROOT / f"tier3_{args.model.replace('/', '_')}_{args.bench}.jsonl"

 done = set(
 if out_path.exists(:
 for line in open(out_path):
 r = json.loads(line)
 done.add((r["idx"], r["run"]))
 log(f"{args.bench}/{args.model}: {len(done)} (q,run) pairs done, resuming")

 jobs = [(qi, r, q) for qi, q in enumerate(questions)
 for r in range(args.runs) if (qi, r) not in done]

 def work(j):
 qi, r, q = j
 votes = [sample_once(q, args.model, args.ollama_url,
 args.temperature, log) for _ in range(args.k)]
 valid = sum(1 for v in votes if v != "__UNPARSEABLE__")
 return {"idx": qi, "run": r, "gt": q["gt"], "C": q["C"],
 "options": q["options"], "votes": votes, "n_valid": valid}

 t0 = time.time(
 with ThreadPoolExecutor(max_workers=args.workers) as pool:
 for k, rec in enumerate(pool.map(work, jobs)):
 with open(out_path, "a") as f:
 f.write(json.dumps(rec) + "\n")
 if (k + 1) % 20 == 0:
 log(f"{args.bench}/{args.model}: {k+1}/{len(jobs)} runs done "
 f"({time.time(-t0:.0f}s)")
 log(f"{args.bench}/{args.model}: complete -> {out_path}")


if __name__ == "__main__":
 main(
