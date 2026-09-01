#!/usr/bin/env python3
"""analysis/extract_gpt41_api.py — controlled gpt-4.1-family sampling via
OpenAI API, mirroring extract_tier3.py exactly (same prompts, parsing,
option shuffle, stratification, output format) so the resulting cells
feed the identical tier3_to_samples -> kappa_tool pipeline.

Designed to be handed to a collaborator who has an OpenAI API key; the
key is read from the OPENAI_API_KEY environment variable and never
appears in this file or any log.

Protocol (identical to Tier-3): R=4 runs/question, K=32 votes/run,
temperature 0.7, zero-shot, num_predict capped at 256 tokens. Models:
gpt-4.1-mini (default) / gpt-4.1 / gpt-4.1-nano via --model.

Data: same official GPQA-Diamond CSV (198 q; gated, obtain via
hf_hub_download with your own HF token, or place it at
data/tier3_static/gpqa_diamond.csv) and the same AIME CSV (200
stratified, seed 42). SHA256 of the question file is printed to the log
for provenance.

Output: data/sampled/gpt41/gpt41_{model}_{bench}.jsonl, one line per
(question, run): {"idx", "run", "gt", "C", "options", "votes",
"n_valid"}; resumable (skips done pairs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve(.parent.parent
OUT_ROOT = REPO_ROOT / "data" / "sampled" / "gpt41"
LETTERS = "ABCDEFGHIJ"

BENCH = {
 "gpqa": {"repo": "Idavidrein/gpqa", "file": "gpqa_diamond.csv",
 "n": 198, "mode": "letter", "gated": True},
 "aime": {"repo": "di-zhang-fdu/AIME_1983_2024",
 "file": "AIME_Dataset_1983_2024.csv", "n": 200,
 "mode": "number"},
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
 nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", t)
 if not nums:
 return "__UNPARSEABLE__"
 return nums[-1].replace(",", "")


def norm_num(s):
 try:
 return float(str(s).replace(",", ""))
 except ValueError:
 return None


def sample_once(row, model, api_key, temperature, log, url):
 import requests
 payload = {"model": model,
 "messages": [{"role": "user",
 "content": build_prompt(row)}],
 "temperature": temperature, "max_tokens": 256}
 headers = {"Authorization": f"Bearer {api_key}"}
 for attempt in range(3):
 try:
 r = requests.post(f"{url}/chat/completions", json=payload,
 headers=headers, timeout=300)
 r.raise_for_status(
 ans = parse_answer(r.json(["choices"][0]["message"]["content"],
 row)
 if not row["options"] and ans != "__UNPARSEABLE__":
 v = norm_num(ans)
 g = norm_num(row["gt"])
 if v is not None and g is not None and v == g:
 ans = row["gt"]
 return ans
 except Exception as e: # noqa: BLE001
 log(f"sample error (attempt {attempt+1}): {e}")
 time.sleep(5 * (attempt + 1))
 return "__UNPARSEABLE__"


def main(:
 ap = argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--bench", required=True, choices=list(BENCH))
 ap.add_argument("--model", default="gpt-4.1-mini")
 ap.add_argument("--runs", type=int, default=4)
 ap.add_argument("--k", type=int, default=32)
 ap.add_argument("--temperature", type=float, default=0.7)
 ap.add_argument("--workers", type=int, default=8)
 ap.add_argument("--api-base",
 default="https://api.openai.com/v1")
 args = ap.parse_args(

 api_key = os.environ.get("OPENAI_API_KEY")
 if not api_key:
 sys.exit("set OPENAI_API_KEY env var")

 ts = time.strftime("%Y%m%d_%H%M%S")
 logdir = REPO_ROOT / "data" / "logs"
 logdir.mkdir(parents=True, exist_ok=True)
 import logging
 logging.basicConfig(level=logging.INFO,
 format="%(asctime)s [%(levelname)s] %(message)s",
 handlers=[logging.FileHandler(
 logdir / f"gpt41_{ts}.log"), logging.StreamHandler(])
 log = logging.getLogger(.info

 OUT_ROOT.mkdir(parents=True, exist_ok=True)
 cache = REPO_ROOT / "data" / "tier3_dl"
 questions = load_questions(args.bench, cache, log)

 # provenance: sha256 of the question file as loaded
 qfile = REPO_ROOT / "data" / "tier3_static" / BENCH[args.bench]["file"]
 qhash = hashlib.sha256(qfile.read_bytes().hexdigest([:16] \
 if qfile.exists( else "downloaded"
 log(f"questions file sha256[0:16]={qhash}; model={args.model}; "
 f"T={args.temperature}; K={args.k}; R={args.runs}; "
 f"api_base={args.api_base}")

 out_path = OUT_ROOT / f"gpt41_{args.model.replace('/', '_')}_{args.bench}.jsonl"
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
 votes = [sample_once(q, args.model, api_key, args.temperature,
 log, args.api_base) for _ in range(args.k)]
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
