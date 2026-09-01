#!/usr/bin/env python3
"""κ-plurality decomposition for LLM self-consistency.

Given de-identified per-run rows (Ding 2026), compute:
 * p single-sample accuracy = mean over runs of A (n_correct/K)
 * majority acc mean(majority_is_correct) (the "consensus" / deployment label)
 * κ_empirical E[α|wrong] · (C-1)/(1-p), α = self-consistency = n_majority/K
 * κ_iid_perq κ under the per-question difficulty-matched i.i.d.-MC null:
 each simulated question draws a per-question accuracy from
 the observed {p_i} (mean A over runs of that case), then K
 independent voters at that accuracy. PRIMARY null.
 * κ_iid_pooled κ under the pooled-p i.i.d.-MC null (every voter correct
 w.p. the cell-mean p). CONTRAST / failed-baseline.
 * plurality_share κ_iid_perq / κ_empirical
 * wrong_consensus empirical / perq / pooled (the backfire decomposition)

Rationale (perq is load-bearing): pooled-p null assumes all questions share one
difficulty, so for a high-accuracy model plurality is almost always correct and
wrong-consensus collapses to ~0 — it cannot predict any backfire. The
per-question null preserves the observed difficulty spread, reproduces most of
the empirical wrong-consensus, and leaves the residual to shared bias. We
therefore report perq as the primary null and keep pooled-p as a deliberately
failed baseline showing why difficulty-matching is required.

Conventions:
 * p = single-sample accuracy, NOT consensus accuracy
 * C = number of answer options; GPQA = 4; AIME via --aime-c-mode
 * i.i.d.-MC tie-break: np.argmax(counts) (smallest class id), rule 19
 * headline quantity plurality_share = κ_iid / κ_empirical

Output: results/kappa_decompose.json (canonical committed evidence).
Bootstrap CIs: B=10^4 on κ_empirical; coupled bootstrap for the share (rules 10).

Usage:
 python analysis/kappa_decompose.py \
 --input data/raw/case_results_deid.parquet \
 --output results/kappa_decompose.json \
 --seed 0 --bootstrap 10000 --n-sim 200000
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd


def parse_args( -> argparse.Namespace:
 p = argparse.ArgumentParser(description=__doc__,
 formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument("--input", required=True,
 help="Path to case_results_deid.parquet (or .csv)")
 p.add_argument("--output", required=True,
 help="Path to write evidence JSON")
 p.add_argument("--seed", type=int, default=0,
 help="RNG seed for bootstrap + i.i.d.-MC")
 p.add_argument("--bootstrap", type=int, default=10000,
 help="Bootstrap resamples B for CIs")
 p.add_argument("--n-sim", type=int, default=200000,
 help="Number of i.i.d.-MC plurality draws per cell")
 p.add_argument("--share-n-sim", type=int, default=4000,
 help="Per-question i.i.d.-MC draws per bootstrap replicate "
 "when building the perq share CI (the expensive step)")
 p.add_argument("--share-bootstrap", type=int, default=2000,
 help="Replicates for the perq share CI (runtime-knob). "
 "The κ_emp CI always uses --bootstrap (B=10^4).")
 p.add_argument("--c-gpqa", type=int, default=4,
 help="C (answer options) for GPQA Diamond")
 p.add_argument("--aime-c-mode", choices=["mean_distinct", "max_distinct", "fixed"],
 default="mean_distinct",
 help="How to set C for AIME")
 p.add_argument("--aime-c-fixed", type=float, default=None,
 help="Fixed C for AIME if --aime-c-mode fixed")
 p.add_argument("--group-cols", nargs="+",
 default=["model", "benchmark", "prompt"],
 help="Columns that define a cell")
 p.add_argument("--wilson", action="store_true",
 help="Also report Wilson-score CI for consensus accuracy")
 return p.parse_args(


# ---------------------------------------------------------------------------
# i.i.d.-MC plurality model
# ---------------------------------------------------------------------------

def iid_mc_plurality(p: float, c: int, K: int, n: int, rng: np.random.Generator,
 ) -> Dict[str, float]:
 """Simulate n realizations of K i.i.d. voters.

 Each voter is correct with prob p; if wrong, answers uniformly over the
 (c-1) incorrect options. Majority answer = argmax(counts) (smallest index
 on ties). Returns conditional-on-wrong statistics and the plurality share
 numerator κ_iid = E[α|wrong]·(C-1)/(1-p).
 """
 correct_opt = 0 # w.l.o.g. option 0 is correct
 wrong_opts = np.arange(1, c, dtype=np.int64)
 # per-voter class probabilities: option 0 -> p, others -> (1-p)/(c-1)
 prob = np.full(c, (1.0 - p) / (c - 1))
 prob[0] = p

 draws = rng.choice(c, size=(n, K), p=prob) # (n, K) voter answers
 # majority per realization (smallest index on ties); one-pass bincount
 # (bit-identical to the former per-column loop, verified by parity tests)
 rows = np.repeat(np.arange(n), K)
 counts = np.bincount(rows * c + draws.ravel(,
 minlength=n * c).reshape(n, c)
 majority = np.argmax(counts, axis=1) # argmax -> smallest index
 n_maj = np.max(counts, axis=1) # (n,)
 alpha = n_maj / K # agreement (self-consistency)
 wrong = majority != correct_opt # plurality is incorrect

 n_wrong = int(wrong.sum()
 if n_wrong == 0:
 return {"n_wrong": 0, "E_alpha_given_wrong": np.nan,
 "kappa_iid": np.nan,
 "wrong_consensus_rate": 0.0}
 E_alpha_wrong = float(np.mean(alpha[wrong]))
 denom = (1.0 - p) / (c - 1)
 kappa_iid = E_alpha_wrong / denom if denom > 0 else np.nan
 return {"n_wrong": n_wrong,
 "E_alpha_given_wrong": E_alpha_wrong,
 "kappa_iid": kappa_iid,
 "wrong_consensus_rate": float(wrong.mean()}


def iid_mc_plurality_perq_sample(p_i: np.ndarray, c: int, K: int, n: int,
 rng: np.random.Generator) -> Dict[str, float]:
 """Per-question difficulty-matched i.i.d.-MC null (PRIMARY), vectorized.

 Each simulated question draws a per-question accuracy from the empirical
 distribution {p_i}, then K independent voters at that accuracy.
 """
 p = rng.choice(p_i, size=n)[:, None] # (n,1) per-question accuracy
 # multinomial: option 0 correct w.p. p, others (1-p)/(c-1)
 u = rng.random((n, K)) # (n,K) uniforms
 p_full = np.broadcast_to(p, u.shape) # (n,K)
 # invert the CDF: threshold bin 0 at p, then split (1-p) evenly over 1..c-1
 preds = np.zeros((n, K), dtype=np.int64)
 wrong = u >= p_full
 with np.errstate(divide="ignore", invalid="ignore"):
 preds[wrong] = 1 + ((u[wrong] - p_full[wrong]) /
 (1.0 - p_full[wrong]) * (c - 1)).astype(np.int64)
 preds[wrong] = np.clip(preds[wrong], 1, c - 1)

 rows = np.repeat(np.arange(n), K)
 counts = np.bincount(rows * c + preds.ravel(,
 minlength=n * c).reshape(n, c)
 majority = np.argmax(counts, axis=1)
 n_maj = np.max(counts, axis=1)
 alpha = n_maj / K
 wrong = majority != 0

 n_wrong = int(wrong.sum()
 if n_wrong == 0:
 return {"n_wrong": 0, "E_alpha_given_wrong": np.nan,
 "kappa_iid": np.nan,
 "wrong_consensus_rate": 0.0}
 E_alpha_wrong = float(np.mean(alpha[wrong]))
 denom = (1.0 - p.mean() / (c - 1)
 kappa_iid = E_alpha_wrong / denom if denom > 0 else np.nan
 return {"n_wrong": n_wrong,
 "E_alpha_given_wrong": E_alpha_wrong,
 "kappa_iid": kappa_iid,
 "wrong_consensus_rate": float(wrong.mean()}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _bootstrap(x: np.ndarray, n_boot: int, rng: np.random.Generator,
 stat=np.mean) -> Tuple[float, float, float]:
 """Percentile bootstrap of stat(x): returns (est, lo, hi)."""
 est = float(stat(x))
 draws = rng.choice(x, size=(n_boot, len(x)), replace=True)
 vals = np.array([stat(d) for d in draws])
 lo, hi = np.percentile(vals, [2.5, 97.5])
 return est, float(lo), float(hi)


def main( -> None:
 args = parse_args(
 rng = np.random.default_rng(args.seed)

 df = pd.read_parquet(args.input) if args.input.endswith(".parquet") \
 else pd.read_csv(args.input, dtype=str)
 # Coerce to canonical dtypes / views
 df["majority_is_correct"] = df["majority_is_correct"].astype(bool)
 df["C"] = df["C"].astype(float) # self-consistency alpha (rule: C col is alpha)
 df["A"] = df["A"].astype(float) # sample accuracy
 df["n_distinct_answers"] = df["n_distinct_answers"].astype(int)

 cells: Dict[str, Dict[str, float]] = {}
 groupby = df.groupby(args.group_cols, dropna=False)

 for key, sub in groupby:
 if not isinstance(key, tuple):
 key = (key,)
 label = "|".join(str(k) for k in key)
 model, benchmark, prompt = key[0], key[1], key[2]
 K = int(sub["K"].iloc[0])

 # p = single-sample accuracy
 p = float(sub["A"].mean()
 # consensus / deployment accuracy
 consensus = float(sub["majority_is_correct"].mean()

 # C per benchmark
 if benchmark == "gpqa_diamond":
 c = float(args.c_gpqa)
 else: # aime
 if args.aime_c_mode == "mean_distinct":
 c = float(sub["n_distinct_answers"].mean()
 elif args.aime_c_mode == "max_distinct":
 c = float(sub["n_distinct_answers"].max()
 else: # fixed
 assert args.aime_c_fixed is not None, "--aime-c-mode fixed needs --aime-c-fixed"
 c = float(args.aime_c_fixed)
 if c <= 1:
 c = 2.0

 # runs where plurality is wrong
 wrong = sub[~sub["majority_is_correct"]]
 alpha_wrong = wrong["C"].values # E[α|wrong] numerator samples
 n_wrong = int(len(wrong))

 denom = (1.0 - p) / (c - 1)
 if n_wrong == 0 or denom <= 0:
 row = {"n_runs": int(len(sub)), "n_wrong": n_wrong,
 "p": p, "consensus_acc": consensus,
 "C_options": c, "kappa_empirical": np.nan,
 "kappa_iid": np.nan, "plurality_share": np.nan,
 "k_emp_lo": np.nan, "k_emp_hi": np.nan,
 "share_lo": np.nan, "share_hi": np.nan}
 cells[label] = row
 continue

 # empirical κ
 k_emp = float(np.mean(alpha_wrong)) / denom
 k_emp_est, k_emp_lo, k_emp_hi = _bootstrap(alpha_wrong, args.bootstrap, rng)
 k_emp_est = float(np.mean(alpha_wrong)) / denom
 k_emp_lo = k_emp_lo / denom
 k_emp_hi = k_emp_hi / denom

 # per-question difficulties: mean A over runs of the same case
 # (rule 18: p_i must come from per-case runs, not the pooled cell mean).
 p_i = sub.groupby("case_id")["A"].mean(.to_numpy(

 # PRIMARY null: per-question difficulty-matched i.i.d.-MC
 mc_perq = iid_mc_plurality_perq_sample(
 p_i, int(c), K, args.n_sim, rng)
 k_iid_perq = mc_perq["kappa_iid"]
 wc_perq = mc_perq["wrong_consensus_rate"]

 # CONTRAST null: pooled-p i.i.d.-MC (all questions share one difficulty)
 mc_pool = iid_mc_plurality(p, int(c), K, args.n_sim, rng)
 k_iid_pool = mc_pool["kappa_iid"]
 wc_pool = mc_pool["wrong_consensus_rate"]

 # coupled bootstrap of the perq share: resample runs, recompute the
 # per-case difficulty set {p_i} and κ_emp, then a fresh perq simulation.
 n_boot = args.share_bootstrap
 r = rng.integers(0, len(sub), size=(n_boot, len(sub)))
 shares = np.empty(n_boot)
 for b in range(n_boot):
 xb = sub.iloc[r[b]]
 p_i_b = xb.groupby("case_id")["A"].mean(.to_numpy(
 if len(p_i_b) == 0:
 shares[b] = np.nan
 continue
 p_b = float(xb["A"].mean()
 aw_b = xb[~xb["majority_is_correct"]]["C"].values
 denom_b = (1.0 - p_b) / (c - 1)
 if len(aw_b) == 0 or denom_b <= 0:
 shares[b] = np.nan
 continue
 k_emp_b = float(np.mean(aw_b)) / denom_b
 mc_perq_b = iid_mc_plurality_perq_sample(
 p_i_b, int(c), K, args.share_n_sim, rng)
 k_iid_b = mc_perq_b["kappa_iid"]
 if np.isnan(k_iid_b) or k_emp_b <= 0:
 shares[b] = np.nan
 continue
 shares[b] = k_iid_b / k_emp_b
 shares_clean = shares[~np.isnan(shares)]
 if len(shares_clean) == 0:
 share_est = share_lo = share_hi = np.nan
 else:
 share_est = float(np.nanmean(shares))
 share_lo, share_hi = np.percentile(shares_clean, [2.5, 97.5])

 row = {
 "model": model, "benchmark": benchmark, "prompt": prompt,
 "n_runs": int(len(sub)), "n_cases": int(len(p_i)), "n_wrong": n_wrong,
 "K": K, "p": p, "consensus_acc": consensus,
 "C_options": c, "E_alpha_given_wrong": float(np.mean(alpha_wrong)),
 "kappa_empirical": k_emp_est,
 "k_emp_lo": k_emp_lo, "k_emp_hi": k_emp_hi,
 "kappa_iid_perq": k_iid_perq,
 "kappa_iid_pooled": k_iid_pool,
 "wrong_consensus_emp": float((~sub["majority_is_correct"]).mean(),
 "wrong_consensus_perq": wc_perq,
 "wrong_consensus_pooled": wc_pool,
 "plurality_share": share_est,
 "share_lo": share_lo, "share_hi": share_hi,
 }
 cells[label] = row

 out = {
 "schema_version": "2.0",
 "generated_by": os.path.basename(__file__),
 "args": vars(args),
 "group_cols": args.group_cols,
 "note": ("alpha = self-consistency (Ding 'C'); p = single-sample accuracy "
 "(Ding 'A'); denom = (1-p)/(C-1); kappa_emp = E[alpha|wrong]/denom. "
 "kappa_iid_perq = per-question difficulty-matched i.i.d.-MC (PRIMARY "
 "null); kappa_iid_pooled = pooled-p i.i.d.-MC (contrast / failed "
 "baseline). plurality_share = kappa_iid_perq/kappa_emp. "
 "wrong_consensus_* = share of majority-wrong runs under each null."),
 "cells": cells,
 }
 os.makedirs(os.path.dirname(args.output), exist_ok=True)
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2, sort_keys=False)
 print(f"wrote {args.output}")


if __name__ == "__main__":
 main(
