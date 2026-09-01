#!/usr/bin/env python3
"""P4: over-dispersion (Dirichlet-multinomial) rival null.

The i.i.d. rival null fixes the per-case wrong-label preference q_hat but
not run-to-run heterogeneity of that preference. This reference adds the
heterogeneity channel: each simulated run first draws its own wrong-label
preference q* ~ Dirichlet(alpha * q_hat) and then draws its wrong votes
from q*. alpha -> inf recovers the multinomial rival; small alpha =
strong run-level heterogeneity. The per-cell alpha is moment-matched to
the observed within-case cross-run dispersion of the plurality share
(the self-consistency alpha of wrong runs within a case), so the null
grants the data exactly the run-level heterogeneity it exhibits.

If the Dirichlet-multinomial share phi_dm sits near 1 on AIME, the
"preference-unexplained residual" is largely run-level heterogeneity;
if it stays far below 1, the residual survives this channel too.

Output: results/kappa_rival_dispersion.json
"""
import argparse
import ast
import json

import numpy as np
import pandas as pd

from kappa_rival_preference import (_C_for, _aggregate_labels, _subtract,
 iid_mc_plurality_pref)

ALPHA_GRID = [0.2, 0.5, 1.0, 1.5, 3.0, 6.0, 12.0, 25.0, 50.0, float("inf")]


def dm_plurality_stats(p, K, n_sim, q_wrong, alpha, rng):
 """Dirichlet-multinomial plurality simulation for one (case, test).

 Returns (E_alpha_given_wrong, var_alpha_across_runs, n_wrong).
 Tie-break: argmin (smallest index, gt=0), matching the main pipeline.
 """
 labels = sorted(q_wrong.keys()
 w = np.array([q_wrong[l] for l in labels], dtype=float)
 if w.sum( <= 0 or len(labels) == 0:
 return np.nan, np.nan, 0
 w = w / w.sum(
 n_c = len(labels) + 1
 alphas = np.empty(n_sim)
 n_wrong = 0
 e_wrong = 0.0
 for m in range(n_sim):
 if np.isinf(alpha):
 qstar = w
 else:
 qstar = rng.dirichlet(alpha * w)
 prob = np.empty(n_c)
 prob[0] = p
 prob[1:] = (1.0 - p) * qstar
 prob /= prob.sum(
 votes = rng.choice(n_c, size=K, p=prob)
 counts = np.bincount(votes, minlength=n_c)
 maj = int(np.argmax(counts))
 alphas[m] = counts[maj] / K
 if maj != 0:
 n_wrong += 1
 e_wrong += alphas[m]
 if n_wrong == 0:
 return np.nan, np.nan, 0
 return e_wrong / n_wrong, float(np.var(alphas)), n_wrong


def main(:
 ap = argparse.ArgumentParser(
 ap.add_argument("--input-cases", required=True)
 ap.add_argument("--input-dist", required=True)
 ap.add_argument("--output", required=True)
 ap.add_argument("--seed", type=int, default=0)
 ap.add_argument("--n-sim", type=int, default=2000)
 ap.add_argument("--min-wrong", type=int, default=1)
 ap.add_argument("--c-gpqa", type=int, default=4)
 ap.add_argument("--aime-c-mode", choices=["mean_distinct", "max_distinct", "fixed"],
 default="mean_distinct")
 ap.add_argument("--aime-c-fixed", type=float, default=None)
 args = ap.parse_args(

 rng = np.random.default_rng(args.seed)
 cases = pd.read_parquet(args.input_cases)
 dist = pd.read_parquet(args.input_dist)
 mp = cases.drop_duplicates(["axis", "condition", "model", "prompt"])[
 ["axis", "condition", "model", "prompt"]]
 dist = dist.merge(mp, on=["axis", "condition"], how="left")
 dist["_wrong"] = ~dist["majority_is_correct"].astype(bool)

 rows = []
 for (model, benchmark, prompt), sub in cases.groupby(
 ["model", "benchmark", "prompt"], dropna=False):
 K = int(sub["K"].iloc[0])
 p = float(sub["A"].mean()
 c = _C_for(None, benchmark, sub, args)
 if c <= 1:
 c = 2.0
 denom = (1.0 - p) / (c - 1)

 dsub = dist[(dist["model"] == model) & (dist["benchmark"] == benchmark) &
 (dist["prompt"] == prompt)].copy(
 case_agg = {cid: _aggregate_labels(g["answer_counts"])
 for cid, g in dsub.groupby("case_id")}
 n_runs_case = dsub.groupby("case_id").size(
 case_A_sum = dsub.groupby("case_id")["A"].sum(

 # observed within-case cross-run dispersion of the plurality share
 obs_vars = []
 for cid, g in dsub.groupby("case_id"):
 w = g[g["_wrong"]]
 if len(w) >= 3:
 obs_vars.append(float(np.var(w["C"].to_numpy()))
 var_obs = float(np.mean(obs_vars)) if obs_vars else np.nan

 # test runs: wrong runs of cases with >=min_wrong wrong runs, >=2 runs
 wrong_per_case = dsub.groupby("case_id")["_wrong"].sum(
 usable = []
 for cid, nw in wrong_per_case.items(:
 if nw < args.min_wrong or n_runs_case[cid] < 2:
 continue
 gt = str(dsub.loc[dsub["case_id"] == cid, "ground_truth"].iloc[0])
 for sid in dsub.loc[(dsub["case_id"] == cid) & dsub["_wrong"],
 "student_id"]:
 usable.append((cid, sid, gt))

 # variance matching on a subset of cases (>=3 wrong runs), split
 # into fit and held-out halves: alpha is fitted on the FIT half,
 # kappa_dm is evaluated on the HELD-OUT half (de-circularizes the
 # calibrated null: the parameter never touches the data it explains)
 var_cases = []
 for cid, g in dsub.groupby("case_id"):
 w = g[g["_wrong"]]
 if len(w) >= 3:
 gt = str(g["ground_truth"].iloc[0])
 q = _subtract(case_agg[cid],
 {"": 0.0}, gt) # no test subtraction here
 var_cases.append((cid, q, float(np.var(w["C"].to_numpy())))
 rng_split = np.random.default_rng(args.seed + 1)
 order = rng_split.permutation(len(var_cases))
 n_fit = len(var_cases) // 2
 fit_cases = [var_cases[i] for i in order[:n_fit]]
 eval_cases = [var_cases[i] for i in order[n_fit:]]
 eval_cids = {c[0] for c in eval_cases}

 # fit alpha: minimize |var_dm(alpha) - var_obs| over the grid
 best_alpha, best_gap = float("inf"), float("inf")
 var_dm_at = {}
 for alpha in ALPHA_GRID:
 varl = []
 for _, q, _ in fit_cases:
 _, vdm, _ = dm_plurality_stats(p, K, args.n_sim, q, alpha, rng)
 if np.isfinite(vdm):
 varl.append(vdm)
 vdm = float(np.mean(varl)) if varl else np.nan
 var_dm_at[str(alpha)] = vdm
 if np.isfinite(vdm) and np.isfinite(var_obs):
 gap = abs(vdm - var_obs)
 if gap < best_gap:
 best_gap, best_alpha = gap, alpha

 # kappa_dm at the fitted alpha, evaluated on HELD-OUT cases only
 draws = []
 for cid, sid, gt in usable:
 if cid not in eval_cids:
 continue
 row = dsub[(dsub["case_id"] == cid) &
 (dsub["student_id"] == sid)].iloc[0]
 A_test = float(row["A"])
 n_case = int(n_runs_case[cid])
 p_lopo = float(np.clip((float(case_A_sum[cid]) - A_test)
 / (n_case - 1), 1e-6, 1.0 - 1e-6))
 tc = ast.literal_eval(row["answer_counts"])
 q = _subtract(case_agg[cid], tc, gt)
 if not q:
 continue
 e_w, _, n_w = dm_plurality_stats(p_lopo, K, args.n_sim, q,
 best_alpha, rng)
 if np.isfinite(e_w) and n_w > 0:
 draws.append(e_w)

 k_dm = float(np.mean(draws)) / denom if draws else np.nan
 # empirical subset on the same tests (alpha unchanged by dispersion)
 test_sids = {sid for cid, sid, _ in usable if cid in eval_cids}
 alpha_t = sub.loc[sub["student_id"].isin(test_sids) &
 ~sub["majority_is_correct"], "C"].values
 k_emp_t = float(np.mean(alpha_t)) / denom if len(alpha_t) else np.nan
 phi_dm = (k_dm / k_emp_t) if (np.isfinite(k_dm) and np.isfinite(k_emp_t)
 and k_emp_t > 0) else np.nan

 rows.append({
 "model": model, "benchmark": benchmark, "prompt": prompt,
 "p": p, "C_options": c,
 "var_obs_plurality_share": var_obs,
 "var_dm_grid": var_dm_at,
 "fitted_alpha": best_alpha,
 "n_fit_cases": len(fit_cases),
 "n_eval_cases": len(eval_cases),
 "kappa_dm": k_dm, "kappa_empirical_subset": k_emp_t,
 "phi_dm": phi_dm,
 "n_test_runs": len(usable), "n_draws": len(draws),
 })

 out = {
 "schema_version": "1.0",
 "generated_by": "kappa_rival_dispersion.py",
 "args": vars(args),
 "note": ("Dirichlet-multinomial rival null: each simulated run draws its "
 "own wrong-label preference q* ~ Dirichlet(alpha*q_hat) before "
 "drawing its wrong votes. alpha per cell is moment-matched to "
 "the within-case cross-run variance of the plurality share on "
 "a random FIT half of the eligible cases (>=3 wrong runs); "
 "kappa_dm and phi_dm are then evaluated on the HELD-OUT half, "
 "so the fitted parameter never explains the data it was fitted "
 "on. alpha=inf = multinomial rival."),
 "cells": rows,
 }
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2)
 print(f"wrote {args.output}")


if __name__ == "__main__":
 main(
