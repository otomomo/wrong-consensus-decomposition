#!/usr/bin/env python3
"""Parallel driver for kappa_rival_preference: per-cell multiprocessing + checkpointing.

Usage:
 python3 kappa_rival_parallel.py \
 --input-cases data/raw/case_results_deid.parquet \
 --input-dist data/raw/answer_distributions_deid.parquet \
 --output results/kappa_rival_preference.json \
 --hierarchical --hier-bootstrap 500 --hier-n-sim 10000 \
 --bootstrap 10000 --n-sim 100000 --seed 0 \
 --workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import ast
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kappa_decompose import iid_mc_plurality_perq_sample
from kappa_rival_preference import (
 iid_mc_plurality_pref,
 _aggregate_labels,
 _subtract,
 _shrink,
 _bootstrap_stat,
 _clustered_bootstrap,
 _clustered_ratio_bootstrap,
 _hierarchical_bootstrap_phi,
 _C_for,
)

logging.basicConfig(
 level=logging.INFO,
 format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
 datefmt="%H:%M:%S",
)
log = logging.getLogger("kappa_parallel")


def parse_args( -> argparse.Namespace:
 p = argparse.ArgumentParser(
 p.add_argument("--input-cases", required=True)
 p.add_argument("--input-dist", required=True)
 p.add_argument("--output", required=True)
 p.add_argument("--seed", type=int, default=0)
 p.add_argument("--tie-seed", type=int, default=42)
 p.add_argument("--tie-break", default="argmin", choices=["argmin", "random"])
 p.add_argument("--bootstrap", type=int, default=10000)
 p.add_argument("--n-sim", type=int, default=100000)
 p.add_argument("--min-wrong", type=int, default=2)
 p.add_argument("--shrink", type=float, default=1.0)
 p.add_argument("--hierarchical", action="store_true")
 p.add_argument("--hier-bootstrap", type=int, default=500)
 p.add_argument("--hier-n-sim", type=int, default=10000)
 p.add_argument("--c-gpqa", type=int, default=4)
 p.add_argument("--aime-c-mode", choices=["mean_distinct", "max_distinct", "fixed"],
 default="mean_distinct")
 p.add_argument("--aime-c-fixed", type=float, default=None)
 p.add_argument("--workers", type=int, default=4)
 p.add_argument("--checkpoint-dir", default=None)
 return p.parse_args(


def _load_data(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
 df = pd.read_parquet(args.input_cases) if args.input_cases.endswith(".parquet") \
 else pd.read_csv(args.input_cases, dtype=str)
 df["majority_is_correct"] = df["majority_is_correct"].astype(bool)
 df["A"] = df["A"].astype(float)
 df["C"] = df["C"].astype(float)
 df["n_distinct_answers"] = df["n_distinct_answers"].astype(int)

 mp = df.drop_duplicates(["axis", "condition", "model", "prompt"])[
 ["axis", "condition", "model", "prompt"]]
 dist = pd.read_parquet(args.input_dist) if args.input_dist.endswith(".parquet") \
 else pd.read_csv(args.input_dist, dtype=str)
 dist = dist.merge(mp, on=["axis", "condition"], how="left")

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
 dist["_maj"] = [_maj_random(e) for e in dist["answer_counts"]]
 dist["_correct"] = (dist["_maj"].astype(str) ==
 dist["ground_truth"].astype(str)).astype(bool)
 correct_map = dist.set_index("student_id")["_correct"].to_dict(
 df["_correct"] = df["student_id"].map(correct_map)
 else:
 df["_correct"] = df["majority_is_correct"]
 dist["_correct"] = dist["majority_is_correct"].astype(bool)
 return df, dist


def _process_cell(task: dict) -> dict:
 """Process one cell. Called in a worker process."""
 label = task["label"]
 t0 = time.time(
 log.info("START %s (n_runs=%d, n_cases=%d)", label,
 task["n_runs"], task["n_cases"])

 args = task["args"]
 seed = task["seed"]
 rng = np.random.default_rng(seed)
 sub = task["sub"]
 dsub = task["dsub"] # pre-filtered dist for this cell

 model, benchmark, prompt = task["model"], task["benchmark"], task["prompt"]
 sub = task["sub"]
 K = int(sub["K"].iloc[0])
 p = float(sub["A"].mean()
 consensus = float(sub["majority_is_correct"].mean()
 c = _C_for(None, benchmark, sub, args)
 if c <= 1:
 c = 2.0
 denom = (1.0 - p) / (c - 1)

 alpha_wrong = sub.loc[~sub["_correct"], "C"].values
 wrong_case_ids = sub.loc[~sub["_correct"], "case_id"].to_numpy(
 n_wrong = int(len(alpha_wrong))

 if n_wrong == 0 or denom <= 0:
 log.info("SKIP %s (n_wrong=0 or denom<=0)", label)
 return {
 "model": model, "benchmark": benchmark, "prompt": prompt,
 "label": label, "n_runs": int(len(sub)), "n_wrong": 0,
 "share_explained_mech": np.nan,
 "share_explained_mech_ci_hierarchical": None,
 }

 k_emp = float(np.mean(alpha_wrong)) / denom

 p_i = sub.groupby("case_id")["A"].mean(.to_numpy(
 perq = iid_mc_plurality_perq_sample(p_i, int(c), K, args.n_sim, rng)
 k_perq = perq["kappa_iid"]
 log.info("%s: kappa_emp=%.4f, kappa_perq=%.4f", label, k_emp, k_perq)

 dsub["_wrong"] = ~dsub["_correct"].astype(bool)

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

 nan_fields = {
 "k_rival_case": np.nan, "k_rival_pool": np.nan,
 "mech_add": np.nan, "shared_res": np.nan,
 "share_mech": np.nan, "k_emp_subset": np.nan,
 "n_test": 0, "test_alphas": np.array([]),
 "test_case_ids": np.array([], dtype=str),
 "kappa_case_arr": np.array([]), "kappa_pool_arr": np.array([]),
 "mc_se_rival": np.nan, "mc_se_pool": np.nan,
 "share_lo": np.nan, "share_hi": np.nan,
 "k_emp_lo": np.nan, "k_emp_hi": np.nan,
 "k_rival_lo": np.nan, "k_rival_hi": np.nan,
 "k_pool_lo": np.nan, "k_pool_hi": np.nan,
 "share_cl_lo": np.nan, "share_cl_hi": np.nan,
 "cl_emp_lo": np.nan, "cl_emp_hi": np.nan,
 "cl_sub_lo": np.nan, "cl_sub_hi": np.nan,
 "cl_riv_lo": np.nan, "cl_riv_hi": np.nan,
 "phi_hier": np.nan, "phi_hier_lo": np.nan, "phi_hier_hi": np.nan,
 }

 if not usable:
 result = nan_fields
 else:
 kappa_case_draws = []
 kappa_pool_draws = []
 draw_case_ids_l = []
 mc_var_case = []
 mc_var_pool = []
 test_alphas_l = []
 test_case_ids_l = []

 cell_agg = _aggregate_labels(dsub["answer_counts"])
 case_agg = {}
 for cid, grp in dsub.groupby("case_id"):
 case_agg[cid] = _aggregate_labels(grp["answer_counts"])
 parsed_cache = {}

 n_usable = len(usable)
 for idx, (cid, sid) in enumerate(usable):
 if idx % 50 == 0:
 log.info("%s: test run %d/%d", label, idx, n_usable)
 case_rows = dsub[dsub["case_id"] == cid]
 n_case = int(n_runs_per_case[cid])
 gt = str(case_rows.loc[case_rows["student_id"] == sid,
 "ground_truth"].iloc[0])
 test = case_rows[case_rows["student_id"] == sid]
 A_test = float(test["A"].iloc[0])
 p_lopo = float(np.clip((float(case_A_sum[cid]) - A_test)
 / (n_case - 1), 1e-6, 1.0 - 1e-6))
 test_counts = parsed_cache.get(sid)
 if test_counts is None:
 test_counts = ast.literal_eval(test["answer_counts"].iloc[0])
 parsed_cache[sid] = test_counts
 q_case = _subtract(case_agg[cid], test_counts, gt)
 q_pool = _subtract(cell_agg, test_counts, gt)
 if args.shrink < 1.0:
 q_case = _shrink(q_case, args.shrink)
 q_pool = _shrink(q_pool, args.shrink)

 got_case = bool(q_case)
 if got_case:
 test_alphas_l.append(float(test["C"].iloc[0]))
 test_case_ids_l.append(cid)
 for q, tag in ((q_case, "case"), (q_pool, "pool")):
 if not q:
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
 share_mech = (k_rival_case / k_emp_subset) \
 if (np.isfinite(k_rival_case) and np.isfinite(k_emp_subset)
 and k_emp_subset > 0) else np.nan
 shared_res = (k_emp - k_rival_case) if np.isfinite(k_rival_case) else np.nan
 mc_se_rival = float(np.sqrt(np.mean(mc_var_case) / max(len(mc_var_case), 1))) \
 / denom if mc_var_case else np.nan
 mc_se_pool = float(np.sqrt(np.mean(mc_var_pool) / max(len(mc_var_pool), 1))) \
 / denom if mc_var_pool else np.nan

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

 # Hierarchical bootstrap
 phi_hier = phi_hier_lo = phi_hier_hi = np.nan
 if args.hierarchical and usable:
 log.info("%s: starting hierarchical bootstrap (B=%d, n_sim=%d)",
 label, args.hier_bootstrap, args.hier_n_sim)
 t_hier = time.time(
 case_ids_arr = np.array([cid for cid, _ in usable])
 phi_hier, phi_hier_lo, phi_hier_hi = _hierarchical_bootstrap_phi(
 dsub, case_ids_arr, usable, case_A_sum, n_runs_per_case,
 c, K, args.hier_bootstrap, args.hier_n_sim, rng)
 log.info("%s: hierarchical done in %.1fs (phi=%.4f, CI=[%.4f, %.4f])",
 label, time.time( - t_hier,
 phi_hier, phi_hier_lo, phi_hier_hi)

 result = {
 "k_rival_case": k_rival_case, "k_rival_pool": k_rival_pool,
 "mech_add": mech_add, "shared_res": shared_res,
 "share_mech": share_mech, "k_emp_subset": k_emp_subset,
 "n_test": n_test, "test_alphas": test_alphas,
 "test_case_ids": test_case_ids,
 "kappa_case_arr": kappa_case_arr, "kappa_pool_arr": kappa_pool_arr,
 "mc_se_rival": mc_se_rival, "mc_se_pool": mc_se_pool,
 "share_lo": share_lo, "share_hi": share_hi,
 "k_emp_lo": k_emp_lo, "k_emp_hi": k_emp_hi,
 "k_rival_lo": k_rival_lo, "k_rival_hi": k_rival_hi,
 "k_pool_lo": k_pool_lo, "k_pool_hi": k_pool_hi,
 "share_cl_lo": share_cl_lo, "share_cl_hi": share_cl_hi,
 "cl_emp_lo": cl_emp_lo, "cl_emp_hi": cl_emp_hi,
 "cl_sub_lo": cl_sub_lo, "cl_sub_hi": cl_sub_hi,
 "cl_riv_lo": cl_riv_lo, "cl_riv_hi": cl_riv_hi,
 "phi_hier": phi_hier, "phi_hier_lo": phi_hier_lo,
 "phi_hier_hi": phi_hier_hi,
 }

 elapsed = time.time( - t0
 log.info("DONE %s in %.1fs (share_mech=%s, phi_hier=%s)",
 label, elapsed,
 f"{result.get('share_mech', float('nan')):.4f}"
 if np.isfinite(result.get('share_mech', np.nan)) else "nan",
 f"{result.get('phi_hier', float('nan')):.4f}"
 if np.isfinite(result.get('phi_hier', np.nan)) else "nan")

 # Format output row
 row = {
 "model": model, "benchmark": benchmark, "prompt": prompt,
 "label": label,
 "n_runs": int(len(sub)), "n_cases": int(sub["case_id"].nunique(),
 "n_wrong": n_wrong, "K": K, "p": p, "consensus_acc": consensus,
 "C_options": c,
 "kappa_empirical": k_emp,
 "kappa_empirical_subset": result.get("k_emp_subset", np.nan),
 "kappa_iid_perq": k_perq,
 "kappa_rival_case": result.get("k_rival_case", np.nan),
 "kappa_rival_pool": result.get("k_rival_pool", np.nan),
 "mechanical_addon_case": result.get("mech_add", np.nan),
 "shared_residual_case": result.get("shared_res", np.nan),
 "share_explained_mech": result.get("share_mech", np.nan),
 "share_explained_mech_ci": [result.get("share_lo", np.nan),
 result.get("share_hi", np.nan)],
 "kappa_empirical_ci": [result.get("k_emp_lo", np.nan),
 result.get("k_emp_hi", np.nan)],
 "kappa_rival_case_ci": [result.get("k_rival_lo", np.nan),
 result.get("k_rival_hi", np.nan)],
 "kappa_rival_pool_ci": [result.get("k_pool_lo", np.nan),
 result.get("k_pool_hi", np.nan)],
 "kappa_empirical_ci_clustered": [result.get("cl_emp_lo", np.nan),
 result.get("cl_emp_hi", np.nan)],
 "kappa_empirical_subset_ci_clustered": [
 result.get("cl_sub_lo", np.nan), result.get("cl_sub_hi", np.nan)],
 "kappa_rival_case_ci_clustered": [result.get("cl_riv_lo", np.nan),
 result.get("cl_riv_hi", np.nan)],
 "share_explained_mech_ci_clustered": [
 result.get("share_cl_lo", np.nan), result.get("share_cl_hi", np.nan)],
 "share_explained_mech_ci_hierarchical":
 [result.get("phi_hier_lo", np.nan), result.get("phi_hier_hi", np.nan)]
 if np.isfinite(result.get("phi_hier_lo", np.nan)) else None,
 "mc_se_rival_case": result.get("mc_se_rival", np.nan),
 "mc_se_rival_pool": result.get("mc_se_pool", np.nan),
 "n_test_runs": result.get("n_test", 0),
 "n_test_with_draw": int(len(result.get("kappa_case_arr", []))),
 "n_test_cases": int(len(np.unique(result.get("test_case_ids", []))))
 if len(result.get("test_case_ids", [])) else 0,
 }
 return row


def _worker(task: dict) -> dict:
 """Top-level worker (must be picklable). Writes checkpoint on success."""
 label = task["label"]
 ckpt_dir = task["ckpt_dir"]
 result = _process_cell(task)
 ckpt_path = os.path.join(ckpt_dir, f"{label.replace('/', '_').replace('|', '_')}.json")
 tmp_path = ckpt_path + ".tmp"
 with open(tmp_path, "w") as f:
 json.dump(result, f, indent=2)
 f.write("\n")
 os.replace(tmp_path, ckpt_path)
 return result


def main(:
 args = parse_args(
 t_start = time.time(
 log.info("=== kappa_rival_parallel starting ===")
 log.info("args: %s", vars(args))

 # Load data once in parent
 df, dist = _load_data(args)
 log.info("Data loaded: %d case rows, %d dist rows", len(df), len(dist))

 # Identify cells
 groups = df.groupby(["model", "benchmark", "prompt"], dropna=False)
 cell_keys = list(groups.groups.keys()
 log.info("Found %d cells", len(cell_keys))

 # Checkpoint dir
 ckpt_dir = args.checkpoint_dir or os.path.join(
 os.path.dirname(args.output), "checkpoints")
 os.makedirs(ckpt_dir, exist_ok=True)

 # Build tasks
 tasks = []
 pending = []
 for i, key in enumerate(cell_keys):
 model, benchmark, prompt = key
 label = "|".join([model, benchmark, prompt])
 sub = df[(df["model"] == model) & (df["benchmark"] == benchmark) &
 (df["prompt"] == prompt)]
 safe_label = label.replace("/", "_").replace("|", "_")
 ckpt_path = os.path.join(ckpt_dir, f"{safe_label}.json")

 if os.path.exists(ckpt_path):
 log.info("SKIP (checkpoint exists): %s", label)
 continue

 task = {
 "label": label,
 "model": model,
 "benchmark": benchmark,
 "prompt": prompt,
 "n_runs": len(sub),
 "n_cases": sub["case_id"].nunique(,
 "seed": args.seed + i * 1000,
 "args": args,
 "ckpt_dir": ckpt_dir,
 }
 # Attach sub-frame + pre-filtered dsub (pickled to worker)
 task["sub"] = sub.reset_index(drop=True)
 dsub_cell = dist[(dist["model"] == model) &
 (dist["benchmark"] == benchmark) &
 (dist["prompt"] == prompt)].reset_index(drop=True)
 task["dsub"] = dsub_cell
 tasks.append(task)
 pending.append(label)

 log.info("%d cells to process, %d already done",
 len(tasks), len(cell_keys) - len(tasks))

 if not tasks:
 log.info("All cells complete, assembling output...")
 else:
 # Parallel execution
 n_workers = min(args.workers, len(tasks))
 log.info("Starting pool with %d workers", n_workers)
 with Pool(processes=n_workers) as pool:
 pool.map(_worker, tasks)
 log.info("All cells processed")

 # Assemble final output from checkpoints
 rows = []
 for key in cell_keys:
 model, benchmark, prompt = key
 label = "|".join([model, benchmark, prompt])
 safe_label = label.replace("/", "_").replace("|", "_")
 ckpt_path = os.path.join(ckpt_dir, f"{safe_label}.json")
 if os.path.exists(ckpt_path):
 with open(ckpt_path) as f:
 rows.append(json.load(f))
 else:
 log.warning("Missing checkpoint for %s", label)

 # Reorder to match original cell order
 out = {
 "schema_version": "1.1",
 "generated_by": "kappa_rival_parallel.py",
 "args": vars(args),
 "note": (
 "kappa_empirical = E[alpha|wrong]/((1-p)/(C-1)) over ALL wrong runs. "
 "kappa_empirical_subset = same over the held-out TEST runs only. "
 "share_explained_mech = rival_case / empirical_subset. "
 "kappa_iid_perq = primary null (uniform distractor). "
 "kappa_rival_case = leak-free per-case answer-preference null. "
 "kappa_rival_pool = same but preference pooled over all cases. "
 "mechanical_addon_case = rival_case - perq. "
 "shared_residual_case = empirical - rival_case. "
 "CIs: run-level percentile bootstrap + case-clustered bootstrap. "
 "share_explained_mech_ci_hierarchical: hierarchical bootstrap "
 "(LOO re-estimation of q_hat)."),
 "cells": rows,
 }
 os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2, sort_keys=False)
 log.info("Wrote %s (%d cells)", args.output, len(rows))
 log.info("=== DONE in %.1f min ===", (time.time( - t_start) / 60)


if __name__ == "__main__":
 main(
