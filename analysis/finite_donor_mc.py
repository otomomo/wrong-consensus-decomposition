#!/usr/bin/env python3
"""Finite-donor Monte Carlo calibration of phi = kappa_rival / kappa_emp.

Responds to the reviewer concern that the small Tier-3 overshoot of phi above 1
(max 0.056) may be a finite-sample artifact of (a) estimating the per-case
answer-preference q_hat from only a few donor runs (leave-one-out) and (b) a
finite set of wrong (test) runs, rather than a real effect.

Design (mirrors the EXACT null semantics of kappa_rival_preference.py):
 * Per case, estimate the "true" i.i.d. generative parameters from ALL observed
 votes of that case:
 p_ref = (total gt votes) / (total votes) (= mean run accuracy)
 q_ref = normalized wrong-PARSEABLE vote histogram (drop gt & _UNPARSEABLE_)
 This is the best estimate of the i.i.d. null that generated the case's data.
 * Per replication: for each case, generate n_obs_runs synthetic runs of K i.i.d.
 voters drawn from the null (a voter is gt w.p. p_ref; else a wrong option
 with prob proportional to q_ref). Then run the SAME pipeline as
 kappa_rival_preference.py on the synthetic data:
 - test runs = synthetic WRONG runs (plurality != gt)
 - for each test run: q_hat = leave-one-out wrong-vote histogram over the
 OTHER (n_obs_runs-1) donor runs (drop gt & unparseable, keep positive);
 p_lopo = mean donor accuracy (clipped to [1e-6, 1-1e-6])
 - E_alpha = iid_mc_plurality_pref(p_lopo, n_c, K, n_sim_cal, q_hat) [inner MC]
 - alpha_emp = synthetic plurality share of the test run
 * phi_hat_rep = mean(E_alpha over test runs) / mean(alpha_emp over test runs)
 (identical ratio to share_explained_mech; the (1-p)/(c-1) denom cancels).
 * Under the i.i.d. null phi_true = 1 exactly; the spread of phi_hat around 1 is
 the finite-donor + finite-test plug-in noise. If the observed phi falls inside
 the 95% band, the overshoot is consistent with finite-donor plug-in noise.

Scope honesty: this quantifies plug-in bias under the i.i.d. null. It does NOT
capture inter-voter correlation (a genuine shared-bias mechanism), which would
push phi BELOW 1; that direction is a real signal, not noise, and is not modelled
here.

Usage (Tier-3, the cells with the overshoot):
 python3 analysis/finite_donor_mc.py \
 --tier3 data/sampled/tier3/tier3_*.jsonl \
 --output results/finite_donor_mc.json \
 --reps 300 --n-sim-cal 8000 --seed 0 --workers 56

Optional Ding (GPT-4.1) cells (phi<1; sanity check of the noise band):
 ... --include-ding --ding-cases data/raw/case_results_deid.parquet \
 --ding-dist data/raw/answer_distributions_deid.parquet
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
 sys.path.insert(0, _HERE)
from kappa_rival_preference import iid_mc_plurality_pref # canonical inner null

UNPARSEABLE = "__UNPARSEABLE_"


# ---------------------------------------------------------------------------
# Per-case "true" i.i.d. parameter extraction (from ALL observed votes)
# ---------------------------------------------------------------------------

def _finalize_case(idx, gt, K, n_runs, q: Counter, total_votes: int,
 n_gt: int) -> Dict | None:
 n_wrong = sum(q.values()
 if n_wrong == 0 or n_runs < 2:
 return None
 q_ref = {str(l): c / n_wrong for l, c in q.items(}
 return {"case": str(idx), "gt": str(gt), "K": int(K), "n_runs": int(n_runs),
 "p_ref": float(n_gt / total_votes), "q_ref": q_ref,
 "n_c": len(q) + 1}


def _extract_cases_tier3(jsonl_path: str) -> List[Dict]:
 cases: Dict = {}
 with open(jsonl_path) as f:
 for line in f:
 rec = json.loads(line)
 case = cases.setdefault(rec["idx"], {"gt": rec["gt"], "runs": []})
 case["runs"].append(rec["votes"])
 out = []
 for idx, case in sorted(cases.items():
 gt = case["gt"]
 K = len(case["runs"][0])
 n_runs = len(case["runs"])
 q = Counter(
 n_gt = 0
 for votes in case["runs"]:
 for v in votes:
 if v == gt:
 n_gt += 1
 elif v != UNPARSEABLE:
 q[v] += 1
 row = _finalize_case(idx, gt, K, n_runs, q, n_runs * K, n_gt)
 if row:
 out.append(row)
 return out


def _extract_cases_ding(cases_parquet: str, dist_parquet: str, model: str,
 benchmark: str, prompt: str) -> List[Dict]:
 c = pd.read_parquet(cases_parquet)
 d = pd.read_parquet(dist_parquet)
 mp = c.drop_duplicates(["axis", "condition"])[
 ["axis", "condition", "model", "prompt"]]
 d = d.merge(mp, on=["axis", "condition"], how="left")
 d = d[(d["model"] == model) & (d["benchmark"] == benchmark)
 & (d["prompt"] == prompt)]
 cases: Dict = {}
 for row in d.itertuples(index=False):
 gt = str(row.ground_truth)
 counts = ast.literal_eval(row.answer_counts)
 cases.setdefault(row.case_id, {"gt": gt, "K": int(row.K),
 "runs": []})["runs"].append(counts)
 out = []
 for idx, case in sorted(cases.items():
 gt = case["gt"]
 K = case["K"]
 n_runs = len(case["runs"])
 q = Counter(
 n_gt = 0
 for counts in case["runs"]:
 for label, cnt in counts.items(:
 if label == gt:
 n_gt += int(cnt)
 elif label != UNPARSEABLE:
 q[str(label)] += int(cnt)
 row = _finalize_case(idx, gt, K, n_runs, q, n_runs * K, n_gt)
 if row:
 out.append(row)
 return out


# ---------------------------------------------------------------------------
# One calibration replication over all cases of a cell
# ---------------------------------------------------------------------------

def _precompute_oracle(case_list: List[Dict], n_sim_cal: int,
 seed: int) -> None:
 """Precompute the ORACLE E_alpha|wrong per case: inner MC with the TRUE
 (p_ref, q_ref), no finite-donor estimation. Fixed per case, reused across all
 reps. Under the i.i.d. null this equals the true E[alpha|wrong], so
 phi_oracle ~ 1 isolates inner-MC + finite-test noise (no q_hat bias)."""
 rng = np.random.default_rng(seed * 7919 + 13)
 for case in case_list:
 n_c = case["n_c"]
 q_ref = case["q_ref"]
 mc = iid_mc_plurality_pref(case["p_ref"], n_c, case["K"], n_sim_cal,
 q_ref, "0", rng)
 e_w = mc["E_alpha_given_wrong"]
 case["e_alpha_oracle"] = float(e_w) if (np.isfinite(e_w)
 and mc["n_wrong"] > 0) else float("nan")


def _sim_rep(case_list: List[Dict], n_sim_cal: int,
 rng: np.random.Generator):
 """One calibration replication. Returns (phi_finite, phi_oracle) or (nan, nan).

 phi_finite: inner MC with the leave-one-out estimated q_hat / p_lopo (mirrors
 the exact pipeline; carries the finite-donor bias).
 phi_oracle: inner MC with the TRUE (p_ref, q_ref) [precomputed]; isolates the
 inner-MC + finite-test noise without the q_hat bias.
 Both use the SAME synthetic test runs and alpha_emp denominator, so the
 difference phi_finite - phi_oracle is the finite-donor plug-in bias."""
 e_alpha_fin: List[float] = []
 e_alpha_ora: List[float] = []
 alpha_emp: List[float] = []
 for case in case_list:
 K = case["K"]
 n_runs = case["n_runs"]
 n_c = case["n_c"]
 q_ref = case["q_ref"]
 e_oracle = case.get("e_alpha_oracle", float("nan"))
 wrong_labels = sorted(q_ref.keys()
 w = np.array([q_ref[l] for l in wrong_labels], dtype=float)
 w = w / w.sum(
 prob = np.empty(n_c)
 prob[0] = case["p_ref"]
 prob[1:] = (1.0 - case["p_ref"]) * w
 prob = prob / prob.sum(

 cats = rng.choice(n_c, size=(n_runs, K), p=prob) # (n_runs, K), 0=gt
 counts = np.stack([np.bincount(r, minlength=n_c) for r in cats])
 maj = np.argmax(counts, axis=1)
 alphas = counts.max(axis=1) / K
 wrong_idx = np.where(maj != 0)[0]

 for t in wrong_idx:
 donor = np.delete(cats, t, axis=0)
 if donor.size == 0:
 continue
 dc = np.bincount(donor.reshape(-1), minlength=n_c)
 qhat: Dict[str, float] = {}
 for k in range(1, n_c):
 if dc[k] > 0:
 qhat[str(k)] = float(dc[k])
 if not qhat:
 continue
 p_lopo = float(np.clip((donor == 0).sum( / (donor.shape[0] * K),
 1e-6, 1.0 - 1e-6))
 mc = iid_mc_plurality_pref(p_lopo, n_c, K, n_sim_cal, qhat, "0", rng)
 e_w = mc["E_alpha_given_wrong"]
 if np.isfinite(e_w) and mc["n_wrong"] > 0:
 e_alpha_fin.append(float(e_w))
 if np.isfinite(e_oracle):
 e_alpha_ora.append(e_oracle)
 alpha_emp.append(float(alphas[t]))
 if not e_alpha_fin:
 return float("nan"), float("nan")
 den = float(np.mean(alpha_emp))
 if den <= 0:
 return float("nan"), float("nan")
 phi_fin = float(np.mean(e_alpha_fin)) / den
 phi_ora = (float(np.mean(e_alpha_ora)) / den) if e_alpha_ora else float("nan")
 return phi_fin, phi_ora


def _worker(payload):
 case_list, reps, seed_base, rep_offset, n_sim_cal = payload
 fin, ora = [], []
 for i in range(reps):
 rng = np.random.default_rng(seed_base * 1_000_003 + rep_offset + i)
 pf, po = _sim_rep(case_list, n_sim_cal, rng)
 fin.append(pf)
 ora.append(po)
 return fin, ora


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_observed_phi(path: str, label: str) -> float:
 """Look up the observed share_explained_mech for a cell label (model|bench|prompt).
 Handles both dict-of-cells (tier3_kappa.json) and list-of-cells
 (kappa_rival_preference.json) schemas."""
 try:
 d = json.load(open(path))
 except (FileNotFoundError, json.JSONDecodeError, ValueError):
 return float("nan")
 cells = d.get("cells", {})
 want_m, want_b, want_p = label.split("|")
 if isinstance(cells, dict):
 rows = list(cells.values()
 keys = [k for k in cells.keys(]
 else:
 rows = list(cells)
 keys = [None] * len(rows)
 for key, row in zip(keys, rows):
 m = row.get("model") or (key.split("|")[0] if key else None)
 b = row.get("benchmark") or (key.split("|")[1] if key else None)
 pr = row.get("prompt") or (key.split("|")[2] if key else None)
 if m == want_m and b == want_b and pr == want_p:
 v = row.get("share_explained_mech")
 if v is not None and np.isfinite(v):
 return float(v)
 return float("nan")


def parse_args( -> argparse.Namespace:
 p = argparse.ArgumentParser(description=__doc__,
 formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument("--tier3", nargs="*", default=[],
 help="Tier-3 jsonl paths (the cells with the phi>1 overshoot)")
 p.add_argument("--include-ding", action="store_true",
 help="also calibrate the gpt-4.1 (Ding) cells (phi<1)")
 p.add_argument("--ding-cases", default=None)
 p.add_argument("--ding-dist", default=None)
 p.add_argument("--ding-model", default="gpt-4.1")
 p.add_argument("--output", required=True)
 p.add_argument("--reps", type=int, default=300,
 help="calibration replications per cell")
 p.add_argument("--n-sim-cal", type=int, default=8000,
 help="inner i.i.d.-MC draws per synthetic test run")
 p.add_argument("--seed", type=int, default=0)
 p.add_argument("--workers", type=int, default=1)
 p.add_argument("--obs-tier3-json",
 default=os.path.join(os.path.dirname(_HERE), "results",
 "tier3_kappa.json"))
 p.add_argument("--obs-ding-json",
 default=os.path.join(os.path.dirname(_HERE), "results",
 "kappa_rival_preference.json"))
 return p.parse_args(


def _tier3_label(jsonl_path: str) -> str:
 # tier3_qwen3.8:27b_gpqa.jsonl -> qwen3.8-27b|gpqa_diamond|zero_shot
 base = os.path.basename(jsonl_path)
 stem = base[len("tier3_"):-len(".jsonl")] if base.startswith("tier3_") else base[:-len(".jsonl")]
 if "_gpqa" in stem:
 model = stem[:-len("_gpqa")]
 bench = "gpqa_diamond"
 elif "_aime" in stem:
 model = stem[:-len("_aime")]
 bench = "aime"
 else:
 model, bench = stem, "unknown"
 model = model.replace(":", "-")
 return f"{model}|{bench}|zero_shot"


def main( -> None:
 args = parse_args(
 cells = []
 for jp in args.tier3:
 cells.append({"label": _tier3_label(jp), "kind": "tier3", "path": jp,
 "obs_json": args.obs_tier3_json})
 if args.include_ding and args.ding_cases and args.ding_dist:
 for bench in ["gpqa_diamond", "aime"]:
 for prompt in ["zero_shot"]:
 cells.append({"label": f"{args.ding_model}|{bench}|{prompt}",
 "kind": "ding", "model": args.ding_model,
 "benchmark": bench, "prompt": prompt,
 "obs_json": args.obs_ding_json})
 if not cells:
 raise SystemExit("no cells: pass --tier3 ... and/or --include-ding")

 cell_cases = {}
 for cell in cells:
 if cell["kind"] == "tier3":
 cell_cases[cell["label"]] = _extract_cases_tier3(cell["path"])
 else:
 cell_cases[cell["label"]] = _extract_cases_ding(
 args.ding_cases, args.ding_dist, cell["model"],
 cell["benchmark"], cell["prompt"])
 # precompute the oracle E_alpha|wrong (true p_ref,q_ref) per case
 _precompute_oracle(cell_cases[cell["label"]], args.n_sim_cal, args.seed)
 n_cases = len(cell_cases[cell["label"]])
 print(f"[fdm] {cell['label']}: {n_cases} usable cases", flush=True)

 # parallelize over (cell, rep-batch); fine batches for load balancing
 n_batches = max(8, min(80, 2 * max(args.workers, 1)))
 batch = max(1, int(np.ceil(args.reps / n_batches)))
 payloads = []
 for cell in cells:
 cl = cell_cases[cell["label"]]
 done = 0
 while done < args.reps:
 r = min(batch, args.reps - done)
 payloads.append((cl, r, args.seed, done, args.n_sim_cal))
 done += r

 if args.workers > 1:
 from multiprocessing import Pool
 with Pool(args.workers) as pool:
 results = pool.map(_worker, payloads)
 else:
 results = [_worker(p) for p in payloads]

 def _dist_stats(vals: np.ndarray) -> Dict:
 vals = vals[np.isfinite(vals)]
 if len(vals) == 0:
 return {"n": 0}
 return {"n": int(len(vals)),
 "mean": float(vals.mean(),
 "median": float(np.median(vals)),
 "sd": float(vals.std(ddof=1)),
 "p05": float(np.percentile(vals, 5)),
 "p95": float(np.percentile(vals, 95)),
 "p99": float(np.percentile(vals, 99)),
 "P_gt_1": float((vals > 1.0).mean()}

 out_cells = {}
 idx = 0
 for cell in cells:
 fin, ora = [], []
 done = 0
 while done < args.reps:
 r = min(batch, args.reps - done)
 f_i, o_i = results[idx]
 fin.extend(f_i); ora.extend(o_i)
 idx += 1
 done += r
 fin_arr = np.asarray(fin, dtype=float)
 ora_arr = np.asarray(ora, dtype=float)
 observed = _load_observed_phi(cell["obs_json"], cell["label"])
 finite = _dist_stats(fin_arr)
 oracle = _dist_stats(ora_arr)
 # paired bias = phi_finite - phi_oracle (both on the same reps)
 paired = fin_arr - ora_arr
 paired = paired[np.isfinite(paired)]
 bias = _dist_stats(paired) if len(paired) else {"n": 0}
 stats = {
 "n_cases_calibrated": len(cell_cases[cell["label"]]),
 "observed_phi": observed,
 "phi_finite": finite, # pipeline with leave-one-out q_hat (biased)
 "phi_oracle": oracle, # inner MC with TRUE q_ref (unbiased ~ 1)
 "finite_minus_oracle_bias": bias,
 }
 if finite.get("n"):
 stats["observed_explained_by_finite_bias"] = (
 np.isfinite(observed) and
 bool(oracle["p05"] <= observed <= oracle["p99"] +
 (finite["mean"] - oracle["mean"])))
 stats["observed_within_oracle_plus_bias_band"] = (
 np.isfinite(observed) and
 bool(finite["p05"] <= observed <= finite["p95"]))
 out_cells[cell["label"]] = stats
 print(f"[fdm] {cell['label']}: observed={observed:.4f} "
 f"finite_mean={finite.get('mean', float('nan')):.4f} "
 f"oracle_mean={oracle.get('mean', float('nan')):.4f} "
 f"bias_mean={bias.get('mean', float('nan')):.4f} "
 f"finite_p95={finite.get('p95', float('nan')):.4f}",
 flush=True)

 out = {
 "schema_version": "1.0",
 "generated_by": os.path.basename(__file__),
 "args": {k: v for k, v in vars(args).items(},
 "note": (
 "phi = kappa_rival/kappa_emp (the (1-p)/(c-1) denom cancels) under an "
 "i.i.d. generative null per case. p_ref/q_ref are estimated from ALL "
 "observed votes of each case (best estimate of the i.i.d. null that "
 "generated the data). Per replication, n_obs_runs synthetic runs are "
 "drawn from the null and the SAME leave-one-out pipeline as "
 "kappa_rival_preference.py is run on them. TWO phi are reported per "
 "replication on the same synthetic test runs: "
 " phi_finite = mean(E_alpha|wrong with leave-one-out q_hat,p_lopo) / "
 " mean(alpha_emp) -> mirrors the paper's estimator; "
 " phi_oracle = mean(E_alpha|wrong with the TRUE p_ref,q_ref) / "
 " mean(alpha_emp) -> unbiased (phi_true=1), isolates the "
 " inner-MC + finite-test noise. "
 "finite_minus_oracle_bias = phi_finite - phi_oracle is the finite-donor "
 "plug-in bias. MECHANISM: estimating q_hat from a few donor runs makes "
 "it MORE concentrated than the true q (E||q_hat||^2 > ||q||^2, a "
 "multinomial/Jensen effect); a more concentrated q makes wrong voters "
 "agree more on the same wrong option, raising the simulated "
 "E[alpha|wrong] and hence inflating phi above 1. If phi_oracle ~ 1 and "
 "phi_finite ~ 1+bias with the observed phi landing in the phi_finite "
 "band, the observed overshoot is the finite-donor plug-in bias, not a "
 "real effect. Scope: quantifies plug-in bias under the i.i.d. null only; "
 "does NOT model inter-voter correlation (a real shared-bias mechanism, "
 "which pushes phi BELOW 1 and is a signal, not noise)."),
 "cells": out_cells,
 }
 os.makedirs(os.path.dirname(args.output), exist_ok=True)
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2, default=str)
 print(f"wrote {args.output}")


if __name__ == "__main__":
 main(
