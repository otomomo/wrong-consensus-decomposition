#!/usr/bin/env python3
"""Rival κ null: per-case answer-preference i.i.d.-MC (leak-free hold-out).

Responds to the reviewer CRITICAL: the primary null `iid_mc_plurality_perq_sample`
assumes that when a voter is wrong she chooses uniformly among the (C-1) incorrect
options. Empirically this is false for GPQA (a fixed letter space {A,B,C,D}): wrong
answers concentrate per question on a single preferred wrong letter (~0.84 mass),
even though the *global* wrong-letter distribution is roughly uniform. So the
primary null under-predicts wrong-answer agreement, and "κ_iid" is too small; part
of what we labelled "shared bias" may be a purely mechanical per-case
answer-layout preference that needs NO cross-run correlation.

This script builds the honest mechanical counterpart:

 * κ_emp: E[α|wrong]·(C-1)/(1-p) on observed runs (same as kappa_decompose).
 * κ_iid_perq: difficulty-matched i.i.d., uniform distractor (imported primary null).
 * κ_rival_case: same as perq BUT the wrong-voter's distractor follows the
 per-case observed option preference q (estimated LEAVING OUT the
 test run), one test run at a time. If per-case layout preference
 is mechanical and stable across runs, this reproduces κ_emp.
 * κ_rival_pool: same but q pooled over ALL cases in the cell (≈uniform lettuce).
 A negative control: shows the preference is per-case, not a
 global "model likes letter B" bias.

Attribution (honest, rule 20):
 * mechanical_addon = κ_rival_case - κ_iid_perq (uniform->preference relaxation)
 * shared_residual = κ_emp - κ_rival_case (agreement NOT explained by a
 fixed per-case preference = true cross-run correlated bias)
 * If κ_rival_case ≈ κ_emp -> mechanical dominates; SHARED-BIAS LABEL TOO STRONG.
 * If κ_rival_case ≪ κ_emp -> shared bias survives; mechanism confirmed.

Leak-free: the preference for a test run is estimated only from the OTHER runs of
the same (cell,case), never from the test run's own votes. This forbids the trivial
self-prediction that would reproduce α by construction.

Multi-class nuance: a voter is correct w.p. p_i (reusing the run's own accuracy =
difficulty-matched), else draws one of the C-1 options with probability ∝ its
observed share among the test run's OTHER runs' non-correct, non-unparseable votes.
Unparseable votes are outside the option space (in both nulls); we renormalize the
preference over observed non-ground-truth OPTION letters only, mirroring the primary
null's "wrong over C-1 options" structure.

Conventions: p = single-sample accuracy; C per benchmark;
np.argmax tie-break; share = κ_iid/κ_emp. Bootstrap B=10^4 on
κ_emp and κ_rival.

Usage:
 python analysis/kappa_rival_preference.py \
 --input-cases data/raw/case_results_deid.parquet \
 --input-dist data/raw/answer_distributions_deid.parquet \
 --output results/kappa_rival_preference.json \
 --seed 0 --bootstrap 10000 --n-sim 100000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import ast
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from kappa_decompose import iid_mc_plurality_perq_sample # primary null, re-used


# ---------------------------------------------------------------------------
# i.i.d.-MC plurality with a non-uniform per-case option preference
# ---------------------------------------------------------------------------

def iid_mc_plurality_pref(p: float, c: int, K: int, n: int,
 q: Dict[str, float], gt: str,
 rng: np.random.Generator,
 tie_break: str = "argmin") -> Dict[str, float]:
 """Simulate n realizations of K i.i.d. voters with option-preference q.

 A voter is correct (picks the ground-truth option) w.p. `p`; otherwise she
 draws one of the (c-1) WRONG options with probability proportional to the
 provided per-case preference `q` (a map label -> weight over wrong options,
 already renormalized to sum to 1). Majority = argmax(counts) smallest-index
 (np.argmax; the labels are encoded 0..c-1 by sorted order, rule 19).

 Returns the same conditional-on-wrong statistics as kappa_decompose's nulls,
 plus n_ready (how many simulated rows had a well-defined majority).
 """
 # encode preference labels to indices 0..c-1 (gt -> 0 w.l.o.g.)
 wrong_labels = sorted(q.keys()
 if c < 2 or len(wrong_labels) == 0:
 # degenerate cell: no usable preference -> treat as uniform fallback
 ws = (1.0 - p) / (c - 1) if c > 1 else 0.0
 prob = np.full(c, ws)
 if c >= 1:
 prob[0] = p
 prob = prob / prob.sum(
 draws = rng.choice(c, size=(n, K), p=prob)
 n_c = c
 else:
 n_c = len(wrong_labels) + 1 # gt + wrong options
 w = np.array([q[l] for l in wrong_labels], dtype=float)
 w = w / w.sum(
 # per-voter probs: option 0 = gt w.p. p ; wrong option k w.p. (1-p)*w_k
 prob = np.empty(n_c)
 prob[0] = p
 prob[1:] = (1.0 - p) * w
 prob = prob / prob.sum(
 draws = rng.choice(n_c, size=(n, K), p=prob)

 rows = np.repeat(np.arange(n), K)
 counts = np.bincount(rows * n_c + draws.ravel(,
 minlength=n * n_c).reshape(n, n_c)
 if tie_break == "random":
 # uniform jitter <1 breaks ties uniformly; integer gaps >=1 keep
 # non-tied maxima unchanged
 counts_j = counts.astype(float) + rng.uniform(0.0, 1.0, size=counts.shape)
 majority = np.argmax(counts_j, axis=1)
 else:
 majority = np.argmax(counts, axis=1) # smallest index on ties
 n_maj = np.max(counts, axis=1)
 alpha = n_maj / K
 wrong = majority != 0 # plurality != gt(encoded 0)

 E_wrong = int(wrong.sum()
 if E_wrong == 0:
 return {"n_wrong": 0, "E_alpha_given_wrong": np.nan,
 "kappa": np.nan, "wrong_consensus_rate": 0.0,
 "mc_se": np.nan}
 E_alpha_wrong = float(np.mean(alpha[wrong]))
 mc_se = float(np.std(alpha[wrong])) / np.sqrt(E_wrong)
 denom = (1.0 - p) / (c - 1) # same normalization as primary
 kappa = E_alpha_wrong / denom if denom > 0 else np.nan
 return {"n_wrong": E_wrong,
 "E_alpha_given_wrong": E_alpha_wrong,
 "kappa": kappa,
 "wrong_consensus_rate": float(wrong.mean(),
 "mc_se": mc_se}


def option_preference(wrong_votes: pd.Series, gt: str) -> Dict[str, float]:
 """Build a preference dict over non-gt OPTION labels from a str->count dict.

 Unparseable votes are excluded (outside the option space), matching the
 primary null's 'wrong over C-1 options' structure. Returns {} if no votes
 land on a parseable wrong option (caller must branch on emptiness).
 """
 prefs: Dict[str, float] = {}
 for entry in wrong_votes:
 d = ast.literal_eval(entry)
 for label, cnt in d.items(:
 if label == "_UNPARSEABLE_":
 continue
 if gt is not None and str(label) == str(gt):
 continue # correct voters, not a 'wrong' option
 prefs[str(label)] = prefs.get(str(label), 0.0) + float(cnt)
 return prefs


def _aggregate_labels(counts_series: pd.Series) -> Dict[str, float]:
 """Sum all label->count dicts (incl. gt & unparseable) over a series of runs."""
 agg: Dict[str, float] = {}
 for entry in counts_series:
 for label, cnt in ast.literal_eval(entry).items(:
 agg[str(label)] = agg.get(str(label), 0.0) + float(cnt)
 return agg


def _subtract(agg: Dict[str, float], counts: Dict[str, float],
 gt: str) -> Dict[str, float]:
 """agg minus one run's counts, dropping gt & unparseable -> option preference.

 O(C) per call. `counts` is the dict-string of a single run (already parsed).
 """
 prefs: Dict[str, float] = {}
 for label, val in agg.items(:
 if label == "_UNPARSEABLE_" or label == str(gt):
 continue
 v = val - float(counts.get(label, 0.0))
 if v > 0:
 prefs[label] = v
 return prefs


def _shrink(q: Dict[str, float], lam: float) -> Dict[str, float]:
 """Shrink a preference dict toward uniform over its own support.

 lam=1 -> q unchanged; lam=0 -> uniform over the support of q.
 Mirrors a Dirichlet(lam*alpha) posterior contraction: the estimate
 q_hat is noisy (small counts per case), and lam<1 asks how kappa_rival
 behaves if that noise is shrunk toward a flat prior.
 """
 if lam >= 1.0 or len(q) <= 1:
 return dict(q)
 n = len(q)
 total = sum(q.values()
 if total <= 0:
 return dict(q)
 out = {}
 for label, v in q.items(:
 out[label] = lam * v + (1.0 - lam) * (total / n)
 return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args( -> argparse.Namespace:
 p = argparse.ArgumentParser(description=__doc__,
 formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument("--input-cases", required=True)
 p.add_argument("--input-dist", required=True)
 p.add_argument("--output", required=True)
 p.add_argument("--seed", type=int, default=0)
 p.add_argument("--bootstrap", type=int, default=10000)
 p.add_argument("--n-sim", type=int, default=100000,
 help="i.i.d.-MC draws per (cell, test run)")
 p.add_argument("--min-wrong", type=int, default=1,
 help="min wrong runs per case to include it as a hold-out test "
 "(the preference is estimated from ALL other runs' votes, "
 "so a case with a single wrong run is still usable provided "
 "it has >=2 total runs)")
 p.add_argument("--shrink", type=float, default=1.0,
 help="shrink the estimated per-case preference toward uniform "
 "over its support: lam in [0,1]; 1.0 = no shrinkage "
 "(sensitivity for q_hat estimation noise)")
 p.add_argument("--tie-break", choices=["argmin", "random"], default="argmin",
 help="plurality tie-break rule; random = uniform among tied "
 "labels, applied to BOTH the empirical majority (recomputed "
 "from answer_counts) and the simulated runs (sensitivity "
 "for the argmin small-label bias)")
 p.add_argument("--tie-seed", type=int, default=0,
 help="seed for the random tie-break draws")
 p.add_argument("--c-gpqa", type=int, default=4)
 p.add_argument("--aime-c-mode", choices=["mean_distinct", "max_distinct", "fixed"],
 default="mean_distinct")
 p.add_argument("--aime-c-fixed", type=float, default=None)
 return p.parse_args(


def _C_for(custom, benchmark: str, sub: pd.DataFrame, args) -> float:
 """Replicate kappa_decompose C rules."""
 if benchmark == "gpqa_diamond":
 return float(args.c_gpqa)
 if args.aime_c_mode == "mean_distinct":
 return float(sub["n_distinct_answers"].mean()
 if args.aime_c_mode == "max_distinct":
 return float(sub["n_distinct_answers"].max()
 assert args.aime_c_fixed is not None
 return float(args.aime_c_fixed)


def _bootstrap_stat(x: np.ndarray, stat, n_boot: int,
 rng: np.random.Generator) -> Tuple[float, float, float]:
 if len(x) == 0:
 return float("nan"), float("nan"), float("nan")
 est = float(stat(x))
 draws = rng.choice(x, size=(n_boot, len(x)), replace=True)
 vals = np.array([stat(d) for d in draws])
 return est, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _clustered_bootstrap(values: np.ndarray, case_ids: np.ndarray,
 n_boot: int, rng: np.random.Generator,
 stat=np.mean) -> Tuple[float, float, float]:
 """Case-clustered bootstrap: resample cases with replacement, then take
 all units of the sampled cases. `case_ids` parallel to `values`."""
 if len(values) == 0:
 return float("nan"), float("nan"), float("nan")
 uniq = np.unique(case_ids)
 by_case = {cid: values[case_ids == cid] for cid in uniq}
 n_cases = len(uniq)
 est = float(stat(values))
 vals = np.empty(n_boot)
 for b in range(n_boot):
 sample = rng.choice(uniq, size=n_cases, replace=True)
 pooled = np.concatenate([by_case[cid] for cid in sample])
 vals[b] = stat(pooled)
 return est, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _clustered_ratio_bootstrap(num_vals: np.ndarray, num_cases: np.ndarray,
 den_vals: np.ndarray, den_cases: np.ndarray,
 n_boot: int, rng: np.random.Generator
 ) -> Tuple[float, float, float]:
 """Coupled case-clustered bootstrap for ratio = mean(num)/mean(den).
 Resamples the shared case population; units of the same case move
 together in both numerator and denominator."""
 uniq = np.unique(np.concatenate([num_cases, den_cases]))
 n_by_case = {c: num_vals[num_cases == c] for c in uniq}
 d_by_case = {c: den_vals[den_cases == c] for c in uniq}
 keep = [c for c in uniq if len(n_by_case[c]) > 0 and len(d_by_case[c]) > 0]
 keep = np.asarray(keep)
 if len(keep) == 0:
 return float("nan"), float("nan"), float("nan")
 nk = len(keep)
 num_all = np.concatenate([n_by_case[c] for c in keep])
 den_all = np.concatenate([d_by_case[c] for c in keep])
 est = float(num_all.mean( / den_all.mean() if den_all.mean( > 0 else float("nan")
 vals = np.empty(n_boot)
 for b in range(n_boot):
 sample = rng.choice(keep, size=nk, replace=True)
 nums = np.concatenate([n_by_case[c] for c in sample])
 dens = np.concatenate([d_by_case[c] for c in sample])
 dm = dens.mean(
 vals[b] = nums.mean( / dm if dm > 0 else np.nan
 vals = vals[np.isfinite(vals)]
 if len(vals) == 0:
 return est, float("nan"), float("nan")
 return est, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main( -> None:
 args = parse_args(
 rng = np.random.default_rng(args.seed)

 df = pd.read_parquet(args.input_cases) if args.input_cases.endswith(".parquet") \
 else pd.read_csv(args.input_cases, dtype=str)
 df["majority_is_correct"] = df["majority_is_correct"].astype(bool)
 df["A"] = df["A"].astype(float)
 df["C"] = df["C"].astype(float)
 df["n_distinct_answers"] = df["n_distinct_answers"].astype(int)

 # map (axis, condition) -> (model, prompt) from case_results (they are the same
 # runs; the dist table lacks model/prompt but shares axis/condition/student_id)
 mp = df.drop_duplicates(["axis", "condition", "model", "prompt"])[
 ["axis", "condition", "model", "prompt"]]
 dist = pd.read_parquet(args.input_dist) if args.input_dist.endswith(".parquet") \
 else pd.read_csv(args.input_dist, dtype=str)
 dist = dist.merge(mp, on=["axis", "condition"], how="left")

 # random tie-break: recompute the empirical majority per run from the
 # answer counts (alpha is unchanged -- the share of the plurality label --
 # only the identity of the winner, and hence wrong-run status, changes).
 tie_rng = np.random.default_rng(args.tie_seed)
 if args.tie_break == "random":
 def _maj_random(entry):
 d = ast.literal_eval(entry)
 if not d:
 return None
 maxc = max(d.values()
 top = [k for k, v in d.items( if v == maxc]
 if len(top) > 1:
 return top[int(tie_rng.integers(len(top)))]
 return top[0]
 dist["_maj"] = [ _maj_random(e) for e in dist["answer_counts"] ]
 dist["_correct"] = (dist["_maj"].astype(str) ==
 dist["ground_truth"].astype(str)).astype(bool)
 correct_map = dist.set_index("student_id")["_correct"].to_dict(
 df["_correct"] = df["student_id"].map(correct_map)
 else:
 df["_correct"] = df["majority_is_correct"]
 dist["_correct"] = dist["majority_is_correct"].astype(bool)

 rows = []
 for (model, benchmark, prompt), sub in df.groupby(
 ["model", "benchmark", "prompt"], dropna=False):
 label = "|".join([model, benchmark, prompt])
 K = int(sub["K"].iloc[0])
 p = float(sub["A"].mean()
 consensus = float(sub["majority_is_correct"].mean()
 c = _C_for(None, benchmark, sub, args)
 if c <= 1:
 c = 2.0
 denom = (1.0 - p) / (c - 1)

 # empirical kappa (same recipe as kappa_decompose)
 alpha_wrong = sub.loc[~sub["_correct"], "C"].values
 wrong_case_ids = sub.loc[~sub["_correct"], "case_id"].to_numpy(
 n_wrong = int(len(alpha_wrong))
 if n_wrong == 0 or denom <= 0:
 rows.append({"model": model, "benchmark": benchmark, "prompt": prompt,
 "n_runs": int(len(sub)), "n_wrong": n_wrong,
 "C_options": c, "p": p, "consensus_acc": consensus,
 "kappa_empirical": np.nan,
 "kappa_iid_perq": np.nan,
 "kappa_rival_case": np.nan, "kappa_rival_pool": np.nan,
 "mechanical_addon_case": np.nan,
 "shared_residual_case": np.nan,
 "share_explained_mech": np.nan,
 "n_test_cases": 0})
 continue
 k_emp = float(np.mean(alpha_wrong)) / denom

 # primary null (uniform distractor) for the cell
 p_i = sub.groupby("case_id")["A"].mean(.to_numpy(
 perq = iid_mc_plurality_perq_sample(p_i, int(c), K, args.n_sim, rng)
 k_perq = perq["kappa_iid"]

 # ---- per-case rival (leak-free hold-out) ----
 # dist per (cell, case_id, student_id=run): answer counts + ground truth
 dsub = dist[(dist["model"] == model) & (dist["benchmark"] == benchmark) &
 (dist["prompt"] == prompt)].copy(
 dsub["_wrong"] = ~dsub["_correct"].astype(bool)

 # candidate test runs: cases with >= min_wrong wrong runs and >=2 total
 # runs (the LOO preference and LOO accuracy need at least one other run).
 wrong_per_case = dsub.groupby("case_id")["_wrong"].sum(
 n_runs_per_case = dsub.groupby("case_id").size(
 case_A_sum = dsub.groupby("case_id")["A"].sum(
 usable = []
 for case_id, nw_case in wrong_per_case.items(:
 if nw_case < args.min_wrong or n_runs_per_case[case_id] < 2:
 continue
 for sid in dsub.loc[(dsub["case_id"] == case_id) & dsub["_wrong"],
 "student_id"]:
 usable.append((case_id, sid))

 if not usable:
 k_rival_case = np.nan
 k_rival_pool = np.nan
 mech_add = np.nan
 shared_res = np.nan
 share_mech = np.nan
 k_emp_subset = np.nan
 n_test = 0
 test_alphas = np.array([])
 test_case_ids = np.array([], dtype=str)
 kappa_case_arr = np.array([])
 kappa_pool_arr = np.array([])
 mc_se_rival = np.nan
 mc_se_pool = np.nan
 share_lo = share_hi = np.nan
 k_emp_lo = k_emp_hi = np.nan
 k_rival_lo = k_rival_hi = np.nan
 k_pool_lo = k_pool_hi = np.nan
 share_cl_lo = share_cl_hi = np.nan
 cl_emp_lo = cl_emp_hi = np.nan
 cl_sub_lo = cl_sub_hi = np.nan
 cl_riv_lo = cl_riv_hi = np.nan
 else:
 kappa_case_draws: list = [] # kappa numerator samples (E[alpha|wrong])
 kappa_pool_draws: list = []
 draw_case_ids_l: list = [] # case ids of tests WITH a case draw
 mc_var_case: list = [] # per-test MC variance of E_alpha
 mc_var_pool: list = []
 test_alphas_l: list = [] # empirical alpha of tests with a draw
 test_case_ids_l: list = []
 # precompute label aggregates ONCE per cell (O(cell)), reuse per test.
 cell_agg = _aggregate_labels(dsub["answer_counts"])
 case_agg: Dict = {}
 for cid, grp in dsub.groupby("case_id"):
 case_agg[cid] = _aggregate_labels(grp["answer_counts"])
 parsed_cache: Dict = {} # sid -> parsed counts dict
 for cid, sid in usable:
 case_rows = dsub[dsub["case_id"] == cid]
 n_case = int(n_runs_per_case[cid])
 gt = str(case_rows.loc[case_rows["student_id"] == sid,
 "ground_truth"].iloc[0])
 test = case_rows[case_rows["student_id"] == sid]
 A_test = float(test["A"].iloc[0])
 # leave-one-out case accuracy, symmetric with the LOO preference
 p_lopo = float(np.clip((float(case_A_sum[cid]) - A_test)
 / (n_case - 1), 1e-6, 1.0 - 1e-6))
 # leak-free: preference trained on the case MINUS this test run
 test_counts = parsed_cache.get(sid)
 if test_counts is None:
 test_counts = ast.literal_eval(
 test["answer_counts"].iloc[0])
 parsed_cache[sid] = test_counts
 q_case = _subtract(case_agg[cid], test_counts, gt)
 # pooled-cell preference (negative control): all OTHER runs in cell
 q_pool = _subtract(cell_agg, test_counts, gt)
 if args.shrink < 1.0:
 q_case = _shrink(q_case, args.shrink)
 q_pool = _shrink(q_pool, args.shrink)

 # The test-run population is defined by the preference support
 # alone (lambda-invariant and deterministic), NOT by simulation
 # outcomes: kappa_empirical_subset must be the same quantity at
 # every shrink level so that the phi(lambda) columns share one
 # denominator.
 got_case = bool(q_case)
 if got_case:
 test_alphas_l.append(float(test["C"].iloc[0]))
 test_case_ids_l.append(cid)
 for q, tag in ((q_case, "case"), (q_pool, "pool")):
 if not q: # no wrong-vote info -> skip
 continue
 mc = iid_mc_plurality_pref(
 p_lopo, int(c), K, args.n_sim, q, gt, rng,
 tie_break=args.tie_break)
 e_w = mc["E_alpha_given_wrong"]
 if np.isfinite(e_w) and mc["n_wrong"] > 0:
 if tag == "case":
 kappa_case_draws.append(e_w)
 draw_case_ids_l.append(cid)
 mc_var_case.append(float(mc["mc_se"]) ** 2)
 else:
 kappa_pool_draws.append(e_w)
 mc_var_pool.append(float(mc["mc_se"]) ** 2)

 kappa_case_arr = np.asarray(kappa_case_draws, dtype=float)
 kappa_pool_arr = np.asarray(kappa_pool_draws, dtype=float)
 draw_case_ids = np.asarray(draw_case_ids_l)
 test_alphas = np.asarray(test_alphas_l, dtype=float)
 test_case_ids = np.asarray(test_case_ids_l)
 n_test = int(len(test_alphas))

 k_rival_case = float(np.mean(kappa_case_draws)) / denom \
 if kappa_case_draws else np.nan
 k_rival_pool = float(np.mean(kappa_pool_draws)) / denom \
 if kappa_pool_draws else np.nan
 k_emp_subset = float(np.mean(test_alphas)) / denom \
 if len(test_alphas) else np.nan
 mech_add = (k_rival_case - k_perq) if np.isfinite(k_rival_case) else np.nan
 # phi uses the SAME test-run population on both sides (R1)
 share_mech = (k_rival_case / k_emp_subset) \
 if (np.isfinite(k_rival_case) and np.isfinite(k_emp_subset)
 and k_emp_subset > 0) else np.nan
 shared_res = (k_emp - k_rival_case) if np.isfinite(k_rival_case) else np.nan
 # MC standard error of the simulated means (aggregated over tests)
 mc_se_rival = float(np.sqrt(np.mean(mc_var_case) / max(len(mc_var_case), 1))) \
 / denom if mc_var_case else np.nan
 mc_se_pool = float(np.sqrt(np.mean(mc_var_pool) / max(len(mc_var_pool), 1))) \
 / denom if mc_var_pool else np.nan

 # run-level bootstrap CI (as before), phi on the test-run subset
 _emp, k_emp_lo, k_emp_hi = _bootstrap_stat(
 alpha_wrong, np.mean, args.bootstrap, rng)
 _r, k_rival_lo, k_rival_hi = _bootstrap_stat(
 kappa_case_arr, np.mean, args.bootstrap, rng)
 _p, k_pool_lo, k_pool_hi = _bootstrap_stat(
 kappa_pool_arr, np.mean, args.bootstrap, rng)
 k_emp_lo, k_emp_hi = k_emp_lo / denom, k_emp_hi / denom
 k_rival_lo, k_rival_hi = k_rival_lo / denom, k_rival_hi / denom
 k_pool_lo, k_pool_hi = k_pool_lo / denom, k_pool_hi / denom
 if (np.isfinite(k_rival_case) and np.isfinite(k_emp_subset)
 and k_emp_subset > 0 and len(test_alphas)):
 emps = rng.choice(test_alphas, size=(args.bootstrap, len(test_alphas)),
 replace=True).mean(axis=1) / denom
 rvals = rng.choice(kappa_case_arr,
 size=(args.bootstrap, len(kappa_case_arr)),
 replace=True).mean(axis=1) / denom
 share_boot = rvals / np.where(emps > 0, emps, np.nan)
 share_lo, share_hi = np.percentile(share_boot, [2.5, 97.5])
 else:
 share_lo, share_hi = np.nan, np.nan

 # case-clustered bootstrap (R3): resample cases with replacement
 _, cl_emp_lo, cl_emp_hi = _clustered_bootstrap(
 alpha_wrong, wrong_case_ids, args.bootstrap, rng)
 _, cl_riv_lo, cl_riv_hi = _clustered_bootstrap(
 kappa_case_arr, draw_case_ids, args.bootstrap, rng)
 _, cl_sub_lo, cl_sub_hi = _clustered_bootstrap(
 test_alphas, test_case_ids, args.bootstrap, rng)
 _, share_cl_lo, share_cl_hi = _clustered_ratio_bootstrap(
 kappa_case_arr, draw_case_ids,
 test_alphas, test_case_ids, args.bootstrap, rng)
 cl_emp_lo, cl_emp_hi = cl_emp_lo / denom, cl_emp_hi / denom
 cl_riv_lo, cl_riv_hi = cl_riv_lo / denom, cl_riv_hi / denom
 cl_sub_lo, cl_sub_hi = cl_sub_lo / denom, cl_sub_hi / denom

 rows.append({
 "model": model, "benchmark": benchmark, "prompt": prompt,
 "n_runs": int(len(sub)), "n_cases": int(len(p_i)),
 "n_wrong": n_wrong, "K": K, "p": p, "consensus_acc": consensus,
 "C_options": c,
 "kappa_empirical": k_emp,
 "kappa_empirical_subset": k_emp_subset,
 "kappa_iid_perq": k_perq,
 "kappa_rival_case": k_rival_case,
 "kappa_rival_pool": k_rival_pool,
 "mechanical_addon_case": mech_add,
 "shared_residual_case": shared_res,
 "share_explained_mech": share_mech,
 "share_explained_mech_ci": [share_lo, share_hi],
 "kappa_empirical_ci": [k_emp_lo, k_emp_hi],
 "kappa_rival_case_ci": [k_rival_lo, k_rival_hi],
 "kappa_rival_pool_ci": [k_pool_lo, k_pool_hi],
 "kappa_empirical_ci_clustered": [cl_emp_lo, cl_emp_hi],
 "kappa_empirical_subset_ci_clustered": [cl_sub_lo, cl_sub_hi],
 "kappa_rival_case_ci_clustered": [cl_riv_lo, cl_riv_hi],
 "share_explained_mech_ci_clustered": [share_cl_lo, share_cl_hi],
 "mc_se_rival_case": mc_se_rival,
 "mc_se_rival_pool": mc_se_pool,
 "n_test_runs": n_test,
 "n_test_with_draw": int(len(kappa_case_arr)) if len(kappa_case_arr) else 0,
 "n_test_cases": int(len(np.unique(test_case_ids))) if len(test_case_ids) else 0,
 })

 out = {
 "schema_version": "1.1",
 "generated_by": os.path.basename(__file__),
 "args": vars(args),
 "note": (
 "kappa_empirical = E[alpha|wrong]/((1-p)/(C-1)) over ALL wrong runs. "
 "kappa_empirical_subset = same over the held-out TEST runs only (the "
 "same population that produces kappa_rival_case). "
 "share_explained_mech = rival_case / empirical_subset, so the ratio "
 "compares the same test-run population on both sides. "
 "kappa_iid_perq = primary null (uniform distractor). "
 "kappa_rival_case = leak-free per-case answer-preference null: each test "
 "run simulated i.i.d. with wrong voters following the per-case option "
 "preference estimated from the OTHER runs of that case, and with "
 "leave-one-out case accuracy (no self-prediction). "
 "kappa_rival_pool = same but preference pooled over all cases in the cell "
 "(negative control). mechanical_addon_case = rival_case - perq. "
 "shared_residual_case = empirical - rival_case. "
 "CIs: run-level percentile bootstrap (B=10^4) plus case-clustered "
 "bootstrap; mc_se_* = Monte Carlo SE of the simulated means (from the "
 "per-draw std, aggregated over test runs). If rival_case ~= empirical, "
 "the 'shared bias' label over-claims."),
 "cells": rows,
 }
 os.makedirs(os.path.dirname(args.output), exist_ok=True)
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2, sort_keys=False)
 print(f"wrote {args.output}")


if __name__ == "__main__":
 main(