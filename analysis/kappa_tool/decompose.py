"""Decomposition orchestration for kappa_tool.

Imports the validated math verbatim from the sibling scripts and re-runs the
exact per-cell sequence of ``kappa_rival_preference.main`` (the canonical
producer of the paper's headline family) followed by the decompose-only extras
of ``kappa_decompose.main`` over aggregated per-run rows. The two committed
JSON generators are kept untouched (they remain the canonical evidence
producers); this module is a library wrapper over the same functions, with a
parity test (``tests/test_parity.py``) asserting the wrapper reproduces the
committed numbers on the Ding data.

Parity contract (per cell, single seeded Generator advanced exactly as in
``kappa_rival_preference.main``):

 BIT-EXACT vs results/kappa_rival_preference.json (same rng sub-stream):
 kappa_empirical, kappa_empirical_ci, kappa_empirical_ci_clustered,
 kappa_empirical_subset, kappa_empirical_subset_ci_clustered,
 kappa_iid_perq, kappa_rival_case, kappa_rival_case_ci,
 kappa_rival_case_ci_clustered, kappa_rival_pool, kappa_rival_pool_ci,
 mechanical_addon_case, shared_residual_case, share_explained_mech,
 share_explained_mech_ci, share_explained_mech_ci_clustered,
 mc_se_rival_case, mc_se_rival_pool, n_test_runs, n_test_with_draw,
 n_test_cases, and the deterministic scalars (p, consensus_acc, C_options,
 K, n_runs, n_wrong, n_cases, E_alpha_given_wrong, wrong_consensus_emp).

 MC-TOLERATED vs results/kappa_decompose.json (independent re-draws, the
 committed decompose file used its own rng sub-stream and n_sim=200000, so
 bit-identity is unreachable; the test asserts |diff| within a documented
 tolerance): kappa_iid_pooled, plurality_share, share_lo, share_hi,
 wrong_consensus_perq, wrong_consensus_pooled.

Two input modes:

 * parity/DB mode (cli --parity): consumes the Ding per-run tables with their
 STORED A/C floats (A = single-sample accuracy, C = plurality share), so
 the rival family is bit-exact. Uses ``student_id`` as the run key.
 * per-sample mode (default): consumes a per-sample long table
 (case_id, run_id, answer, is_correct|ground_truth), re-derives A/C by
 re-counting, and runs the same cell body with a fresh seed. A/C
 re-derivation may differ from Ding's stored floats in the last bit, so the
 rival family is a fresh MC estimate, not bit-locked to the committed JSON
 (documented, not a silent claim).

Headline quantities produced per cell:

 p single-sample accuracy
 consensus_acc mean(majority_is_correct)
 kappa_empirical E[alpha|wrong] / ((1-p)/(C-1))
 kappa_iid_perq per-question difficulty-matched i.i.d.-MC null (PRIMARY)
 kappa_iid_pooled pooled-p i.i.d.-MC null (contrast / failed baseline)
 kappa_rival_case leak-free per-case answer-preference null
 kappa_rival_pool preference pooled over the whole cell (negative control)
 kappa_empirical_subset empirical kappa on the held-out test runs only
 mechanical_addon_case rival_case - iid_perq
 shared_residual_case empirical - rival_case
 plurality_share iid_perq / empirical
 share_explained_mech rival_case / emp_subset
 wrong_consensus_emp / perq / pooled

Bootstrap: run-level percentile (B=10^4) and case-clustered CIs; the perq
plurality-share CI uses a coupled case-clustered bootstrap.

Random-tie-break sensitivity (cfg.tie_break="random") recomputes the per-run
majority from answer_counts with a dedicated tie Generator. NOTE: the
committed kappa_rival_tie_rnd*.json files applied the tie draws over the whole
dist frame BEFORE the cell loop; this wrapper applies them per cell, so random
mode is a documented sensitivity path and NOT bit-locked to those files.
"""
from __future__ import annotations

import ast
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Make the sibling validated scripts importable as modules.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ANALYSIS = os.path.dirname(_HERE)
if _ANALYSIS not in sys.path:
 sys.path.insert(0, _ANALYSIS)

from kappa_decompose import ( # noqa: E402 (validated primary null + helpers)
 iid_mc_plurality,
 iid_mc_plurality_perq_sample,
)
from kappa_rival_preference import ( # noqa: E402 (validated rival null + helpers)
 _C_for,
 _aggregate_labels,
 _bootstrap_stat,
 _clustered_bootstrap,
 _clustered_ratio_bootstrap,
 _shrink,
 _subtract,
 iid_mc_plurality_pref,
)


class Config:
 """Thin args object mirroring the CLI flags of the two validated scripts.

 Defaults match the committed kappa_rival_preference.json invocation
 (seed=0, bootstrap=1e4, n_sim=1e5, min_wrong=1, shrink=1.0, c_gpqa=4,
 aime_c_mode=mean_distinct, tie_break=argmin) plus the committed
 kappa_decompose.json extras (share_n_sim=4000, share_bootstrap=2000).
 """

 def __init__(self, *, seed: int = 0, bootstrap: int = 10000,
 n_sim: int = 100000, share_n_sim: int = 4000,
 share_bootstrap: int = 2000, min_wrong: int = 1,
 shrink: float = 1.0, tie_break: str = "argmin",
 tie_seed: int = 0, c_gpqa: int = 4,
 aime_c_mode: str = "mean_distinct",
 aime_c_fixed: Optional[float] = None,
 c_fixed: Optional[float] = None):
 self.seed = seed
 self.bootstrap = bootstrap
 self.n_sim = n_sim
 self.share_n_sim = share_n_sim
 self.share_bootstrap = share_bootstrap
 self.min_wrong = min_wrong
 self.shrink = shrink
 self.tie_break = tie_break
 self.tie_seed = tie_seed
 self.c_gpqa = c_gpqa
 self.aime_c_mode = aime_c_mode
 self.aime_c_fixed = aime_c_fixed
 self.c_fixed = c_fixed


def _group_cells(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
 """Return {cell_label: subgroup} over the (model, benchmark, prompt) cells.

 A cell label is 'model|benchmark|prompt'; if none of those columns exist,
 the whole frame is a single cell labelled 'single'.
 """
 keys = [c for c in ("model", "benchmark", "prompt") if c in df.columns]
 if not keys:
 return {"single": df}
 out: Dict[str, pd.DataFrame] = {}
 for key, sub in df.groupby(keys, dropna=False):
 if not isinstance(key, tuple):
 key = (key,)
 out["|".join(str(k) for k in key)] = sub
 return out


def _majority_random_sub(frame: pd.DataFrame, tie_rng: np.random.Generator,
 run_key: str) -> pd.DataFrame:
 """Recompute per-run majority identity under random tie-break (sensitivity).

 Mirrors kappa_rival_preference.main's ``_maj_random`` applied to a frame
 carrying ``answer_counts``. NOTE: applied per cell here (see module
 docstring) — a documented sensitivity path, not bit-locked.
 """
 def _maj_random(entry):
 d = ast.literal_eval(entry)
 if not d:
 return None
 maxc = max(d.values()
 top = [k for k, v in d.items( if v == maxc]
 if len(top) > 1:
 return top[int(tie_rng.integers(len(top)))]
 return top[0]

 frame = frame.copy(
 maj = [_maj_random(e) for e in frame["answer_counts"]]
 frame["_correct"] = (pd.Series(maj).astype(str) ==
 frame["ground_truth"].astype(str)).to_numpy(
 return frame


def decompose_cell(cases: pd.DataFrame, dist: pd.DataFrame, cfg: Config,
 label: str, rng: np.random.Generator,
 run_key: str = "run_id") -> Dict[str, object]:
 """Run the full decomposition for a single cell (Ding-like run rows).

 ``cases`` and ``dist`` must already be restricted to this cell.
 ``run_key`` names the per-run id column ('run_id' in per-sample mode,
 'student_id' in parity/DB mode). The Generator is advanced EXACTLY in the
 order of kappa_rival_preference.main for the rival family (bit-exact
 parity), then further for the decompose-only extras (MC-tolerated parity).
 """
 sub = cases.copy(
 sub["majority_is_correct"] = sub["majority_is_correct"].astype(bool)
 sub["A"] = sub["A"].astype(float)
 sub["C"] = sub["C"].astype(float)
 sub["n_distinct_answers"] = sub["n_distinct_answers"].astype(int)

 K = int(sub["K"].iloc[0])
 p = float(sub["A"].mean()
 consensus = float(sub["majority_is_correct"].mean()
 benchmark = str(sub["benchmark"].iloc[0]) if "benchmark" in sub.columns \
 else "aime"
 c = cfg.c_fixed if getattr(cfg, 'c_fixed', None) else _C_for(None, benchmark, sub, cfg)
 if c <= 1:
 c = 2.0
 denom = (1.0 - p) / (c - 1)

 # ---- _correct convention (mirror rival.main; argmin = majority_is_correct)
 dsub = dist.copy(
 if cfg.tie_break == "random":
 tie_rng = np.random.default_rng(cfg.tie_seed)
 dsub = _majority_random_sub(dsub, tie_rng, run_key)
 corr_map = dsub.set_index(run_key)["_correct"].to_dict(
 sub["_correct"] = sub[run_key].map(corr_map)
 else:
 sub["_correct"] = sub["majority_is_correct"]
 dsub["_correct"] = dsub["majority_is_correct"].astype(bool)

 alpha_wrong = sub.loc[~sub["_correct"], "C"].values
 wrong_case_ids = sub.loc[~sub["_correct"], "case_id"].to_numpy(
 n_wrong = int(len(alpha_wrong))
 p_i = sub.groupby("case_id")["A"].mean(.to_numpy(

 row: Dict[str, object] = {
 "label": label, "n_runs": int(len(sub)),
 "n_cases": int(len(p_i)), "n_wrong": n_wrong,
 "K": K, "p": p, "consensus_acc": consensus, "C_options": c,
 "E_alpha_given_wrong": float(np.mean(alpha_wrong)) if n_wrong else np.nan,
 }
 for col in ("model", "benchmark", "prompt"):
 row[col] = sub[col].iloc[0] if col in sub.columns else None

 # ---- degenerate path: no wrong runs, or no usable disagreement ----
 if n_wrong == 0 or denom <= 0:
 row.update({
 "kappa_empirical": np.nan,
 "kappa_empirical_ci": [np.nan, np.nan],
 "kappa_empirical_ci_clustered": [np.nan, np.nan],
 "kappa_empirical_subset": np.nan,
 "kappa_empirical_subset_ci_clustered": [np.nan, np.nan],
 "kappa_iid_perq": np.nan,
 "kappa_iid_pooled": np.nan,
 "kappa_rival_case": np.nan,
 "kappa_rival_case_ci": [np.nan, np.nan],
 "kappa_rival_case_ci_clustered": [np.nan, np.nan],
 "kappa_rival_pool": np.nan,
 "kappa_rival_pool_ci": [np.nan, np.nan],
 "mechanical_addon_case": np.nan,
 "shared_residual_case": np.nan,
 "share_explained_mech": np.nan,
 "share_explained_mech_ci": [np.nan, np.nan],
 "share_explained_mech_ci_clustered": [np.nan, np.nan],
 "mc_se_rival_case": np.nan,
 "mc_se_rival_pool": np.nan,
 "plurality_share": np.nan,
 "share_lo": np.nan, "share_hi": np.nan,
 "wrong_consensus_emp": float((~sub["_correct"]).mean(),
 "wrong_consensus_perq": np.nan,
 "wrong_consensus_pooled": np.nan,
 "n_test_runs": 0, "n_test_with_draw": 0, "n_test_cases": 0,
 "degraded": "no-wrong-runs",
 })
 return row

 k_emp = float(np.mean(alpha_wrong)) / denom

 # ================= RIVAL FAMILY (bit-exact, mirror rival.main) ==========
 # 1) primary null (uniform distractor, difficulty-matched) -- FIRST rng use
 perq = iid_mc_plurality_perq_sample(p_i, int(c), K, cfg.n_sim, rng)
 k_perq = perq["kappa_iid"]

 # 2) per-case rival (leak-free hold-out)
 dsub["_wrong"] = ~dsub["_correct"].astype(bool)
 wrong_per_case = dsub.groupby("case_id")["_wrong"].sum(
 n_runs_per_case = dsub.groupby("case_id").size(
 case_A_sum = dsub.groupby("case_id")["A"].sum(
 usable = []
 for case_id, nw_case in wrong_per_case.items(:
 if nw_case < cfg.min_wrong or n_runs_per_case[case_id] < 2:
 continue
 for sid in dsub.loc[(dsub["case_id"] == case_id) & dsub["_wrong"],
 run_key]:
 usable.append((case_id, sid))

 if not usable:
 kappa_case_arr = np.array([])
 test_alphas = np.array([])
 test_case_ids = np.array([], dtype=str)
 n_test = 0
 k_rival_case = k_rival_pool = np.nan
 k_emp_subset = np.nan
 mech_add = shared_res = share_mech = np.nan
 mc_se_rival = mc_se_pool = np.nan
 share_lo_r = share_hi_r = np.nan
 k_emp_lo = k_emp_hi = np.nan
 k_rival_lo = k_rival_hi = np.nan
 k_pool_lo = k_pool_hi = np.nan
 share_cl_lo = share_cl_hi = np.nan
 cl_emp_lo = cl_emp_hi = np.nan
 cl_sub_lo = cl_sub_hi = np.nan
 cl_riv_lo = cl_riv_hi = np.nan
 else:
 kappa_case_draws: List[float] = []
 kappa_pool_draws: List[float] = []
 draw_case_ids_l: List[str] = []
 mc_var_case: List[float] = []
 mc_var_pool: List[float] = []
 test_alphas_l: List[float] = []
 test_case_ids_l: List[str] = []
 cell_agg = _aggregate_labels(dsub["answer_counts"])
 case_agg: Dict = {}
 for cid, grp in dsub.groupby("case_id"):
 case_agg[cid] = _aggregate_labels(grp["answer_counts"])
 parsed_cache: Dict = {}
 for cid, sid in usable:
 case_rows = dsub[dsub["case_id"] == cid]
 n_case = int(n_runs_per_case[cid])
 gt = str(case_rows.loc[case_rows[run_key] == sid,
 "ground_truth"].iloc[0])
 test = case_rows[case_rows[run_key] == sid]
 A_test = float(test["A"].iloc[0])
 p_lopo = float(np.clip((float(case_A_sum[cid]) - A_test)
 / (n_case - 1), 1e-6, 1.0 - 1e-6))
 test_counts = parsed_cache.get(sid)
 if test_counts is None:
 test_counts = ast.literal_eval(test["answer_counts"].iloc[0])
 parsed_cache[sid] = test_counts
 q_case = _subtract(case_agg[cid], test_counts, gt)
 q_pool = _subtract(cell_agg, test_counts, gt)
 if cfg.shrink < 1.0:
 q_case = _shrink(q_case, cfg.shrink)
 q_pool = _shrink(q_pool, cfg.shrink)
 got_case = bool(q_case)
 if got_case:
 test_alphas_l.append(float(test["C"].iloc[0]))
 test_case_ids_l.append(cid)
 for q, tag in ((q_case, "case"), (q_pool, "pool")):
 if not q:
 continue
 mc = iid_mc_plurality_pref(p_lopo, int(c), K, cfg.n_sim,
 q, gt, rng,
 tie_break=cfg.tie_break)
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

 # 3) run-level bootstrap CIs (order: emp, rival_case, rival_pool)
 _, k_emp_lo, k_emp_hi = _bootstrap_stat(alpha_wrong, np.mean,
 cfg.bootstrap, rng)
 _, k_rival_lo, k_rival_hi = _bootstrap_stat(kappa_case_arr, np.mean,
 cfg.bootstrap, rng)
 _, k_pool_lo, k_pool_hi = _bootstrap_stat(kappa_pool_arr, np.mean,
 cfg.bootstrap, rng)
 k_emp_lo, k_emp_hi = k_emp_lo / denom, k_emp_hi / denom
 k_rival_lo, k_rival_hi = k_rival_lo / denom, k_rival_hi / denom
 k_pool_lo, k_pool_hi = k_pool_lo / denom, k_pool_hi / denom

 # 4) coupled share bootstrap (phi on the test-run subset)
 if (np.isfinite(k_rival_case) and np.isfinite(k_emp_subset)
 and k_emp_subset > 0 and len(test_alphas)):
 emps = rng.choice(test_alphas, size=(cfg.bootstrap, len(test_alphas)),
 replace=True).mean(axis=1) / denom
 rvals = rng.choice(kappa_case_arr,
 size=(cfg.bootstrap, len(kappa_case_arr)),
 replace=True).mean(axis=1) / denom
 share_boot = rvals / np.where(emps > 0, emps, np.nan)
 share_lo_r, share_hi_r = np.percentile(share_boot, [2.5, 97.5])
 else:
 share_lo_r, share_hi_r = np.nan, np.nan

 # 5) case-clustered bootstraps (order: emp, rival, subset, ratio)
 _, cl_emp_lo, cl_emp_hi = _clustered_bootstrap(
 alpha_wrong, wrong_case_ids, cfg.bootstrap, rng)
 _, cl_riv_lo, cl_riv_hi = _clustered_bootstrap(
 kappa_case_arr, draw_case_ids, cfg.bootstrap, rng)
 _, cl_sub_lo, cl_sub_hi = _clustered_bootstrap(
 test_alphas, test_case_ids, cfg.bootstrap, rng)
 _, share_cl_lo, share_cl_hi = _clustered_ratio_bootstrap(
 kappa_case_arr, draw_case_ids,
 test_alphas, test_case_ids, cfg.bootstrap, rng)
 cl_emp_lo, cl_emp_hi = cl_emp_lo / denom, cl_emp_hi / denom
 cl_riv_lo, cl_riv_hi = cl_riv_lo / denom, cl_riv_hi / denom
 cl_sub_lo, cl_sub_hi = cl_sub_lo / denom, cl_sub_hi / denom

 # ================= DECOMPOSE-ONLY EXTRAS (MC-tolerated parity) ==========
 row.update(decompose_extras(sub, cfg))

 row.update({
 "kappa_empirical": k_emp,
 "kappa_empirical_ci": [k_emp_lo, k_emp_hi],
 "kappa_empirical_ci_clustered": [cl_emp_lo, cl_emp_hi],
 "kappa_empirical_subset": k_emp_subset,
 "kappa_empirical_subset_ci_clustered": [cl_sub_lo, cl_sub_hi],
 "kappa_iid_perq": k_perq,
 "kappa_rival_case": k_rival_case,
 "kappa_rival_case_ci": [k_rival_lo, k_rival_hi],
 "kappa_rival_case_ci_clustered": [cl_riv_lo, cl_riv_hi],
 "kappa_rival_pool": k_rival_pool,
 "kappa_rival_pool_ci": [k_pool_lo, k_pool_hi],
 "mechanical_addon_case": mech_add,
 "shared_residual_case": shared_res,
 "share_explained_mech": share_mech,
 "share_explained_mech_ci": [share_lo_r, share_hi_r],
 "share_explained_mech_ci_clustered": [share_cl_lo, share_cl_hi],
 "mc_se_rival_case": mc_se_rival,
 "mc_se_rival_pool": mc_se_pool,
 "wrong_consensus_perq": perq["wrong_consensus_rate"],
 "n_test_runs": n_test,
 "n_test_with_draw": int(len(kappa_case_arr)) if len(kappa_case_arr) else 0,
 "n_test_cases": int(len(np.unique(test_case_ids))) if len(test_case_ids) else 0,
 "degraded": None,
 })
 return row


def decompose_extras(sub: pd.DataFrame, cfg: Config) -> Dict[str, object]:
 """Decompose-only extras (kappa_decompose family) for one cell.

 These draw from a DEDICATED generator (seeded cfg.seed + 1_000_000) so
 they can never perturb the rival-family sub-stream -- the rival family
 stays bit-exact regardless of cell order or usable/degenerate paths.
 Returns {kappa_iid_pooled, plurality_share, share_lo, share_hi,
 wrong_consensus_pooled, wrong_consensus_emp}. Parity vs
 results/kappa_decompose.json is MC-tolerated (the committed file used its
 own rng sub-stream and n_sim=200000; see test_extras_parity.py).
 """
 sub = sub.copy(
 sub["majority_is_correct"] = sub["majority_is_correct"].astype(bool)
 sub["A"] = sub["A"].astype(float)
 sub["C"] = sub["C"].astype(float)
 sub["n_distinct_answers"] = sub["n_distinct_answers"].astype(int)

 K = int(sub["K"].iloc[0])
 p = float(sub["A"].mean()
 benchmark = str(sub["benchmark"].iloc[0]) if "benchmark" in sub.columns \
 else "aime"
 c = cfg.c_fixed if getattr(cfg, 'c_fixed', None) else _C_for(None, benchmark, sub, cfg)
 if c <= 1:
 c = 2.0
 extra_rng = np.random.default_rng(cfg.seed + 1_000_000)

 # 6) contrast null (pooled p)
 pool = iid_mc_plurality(p, int(c), K, cfg.n_sim, extra_rng)
 k_pool = pool["kappa_iid"]

 # 7) coupled bootstrap of the perq plurality share (kappa_decompose path)
 n_boot = cfg.share_bootstrap
 r = extra_rng.integers(0, len(sub), size=(n_boot, len(sub)))
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
 p_i_b, int(c), K, cfg.share_n_sim, extra_rng)
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

 correct_flag = sub["_correct"] if "_correct" in sub.columns \
 else sub["majority_is_correct"]
 return {
 "kappa_iid_pooled": k_pool,
 "plurality_share": share_est,
 "share_lo": share_lo, "share_hi": share_hi,
 "wrong_consensus_pooled": pool["wrong_consensus_rate"],
 "wrong_consensus_emp": float((~correct_flag).mean(),
 }


def decompose_runs(runs: pd.DataFrame, cfg: Optional[Config] = None,
 run_key: str = "run_id") -> Dict[str, object]:
 """Full decomposition over all cells of an aggregated per-run frame.

 ``runs`` is the output of ``load.aggregate_to_runs`` (per-sample mode) or
 a Ding per-run table (parity/DB mode). Returns a {cell_label: row} map in
 the committed-scripts envelope style.
 """
 cfg = cfg or Config(
 rng = np.random.default_rng(cfg.seed)

 cells: Dict[str, object] = {}
 for label, sub in _group_cells(runs).items(:
 from . import load as _load
 cases, dist = _load.split_tables(sub.reset_index(drop=True))
 cells[label] = decompose_cell(cases, dist, cfg, label, rng,
 run_key=run_key)
 return {"schema_version": "1.0", "generated_by": "kappa_tool",
 "config": cfg.__dict__, "cells": cells}


def decompose(df: pd.DataFrame, cfg: Optional[Config] = None,
 per_sample: bool = False,
 run_key: str = "run_id") -> Dict[str, object]:
 """Entry point: decompose a per-sample table or an aggregated-runs frame.

 With ``per_sample=True``, ``df`` is a per-sample long table
 (case_id, run_id, answer, is_correct|ground_truth); A/C are re-derived by
 re-counting (may differ from Ding's stored floats in the last bit -- see
 module docstring). Otherwise ``df`` is an aggregated per-run frame with
 stored A/C.
 """
 if per_sample:
 from . import load as _load
 df = _load.derive_ground_truth(df)
 df = _load.aggregate_to_runs(df)
 return decompose_runs(df, cfg, run_key=run_key)
