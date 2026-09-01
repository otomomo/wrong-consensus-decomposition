#!/usr/bin/env python3
"""Reproduce the self-consistency "backfire" from Ding 2026 per-run data.

Zero-sampling reproduction (prefers Ding public per-run
data over Ollama sampling). P0 goal is to lock the phenomenon:

 * p single-sample accuracy = mean over runs of A (n_correct/K)
 * consensus_acc fraction of runs where majority_is_correct (deployment label)
 * "backfire" when majority voting LOWERS accuracy on hard/high-agreement
 questions, relative to the single-sample baseline (p).

Two complementary views:
 (A) Global: consensus_acc - p per (model, benchmark, prompt) cell. Positive =
 voting helps; strongly negative on hard cells = backfire.
 (B) Agreement-bucketed (Bahuguna-style): bin runs by self-consistency alpha
 (= Ding's C = n_majority/K) and report consensus_acc within each bin.
 If the highest-agreement bin does NOT approach 1.0 (Bahuguna: ~half),
 that is the backfire / ceiling signature.

Conventions:
 * p = single-sample accuracy, NOT consensus accuracy
 * alpha = self-consistency = Ding 'C' = n_majority/K
 * consensus_acc = mean(majority_is_correct)
 * bootstrap CIs on headline numbers; coupled bootstrap for the
 backfire gap (consensus_acc - p).

Output: results/backfire_repro.json (canonical committed evidence).

Usage:
 python analysis/reproduce_backfire.py \
 --input data/raw/case_results_deid.parquet \
 --output results/backfire_repro.json \
 --seed 0 --bootstrap 10000 --n-bins 5
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_ci(x: np.ndarray, B: int, rng: np.random.Generator) -> tuple[float, float, float]:
 """Percentile bootstrap CI (B resamples) on the mean of x. Returns est, lo, hi."""
 est = float(np.mean(x))
 r = rng.integers(0, len(x), size=(B, len(x)))
 means = np.mean(x[r], axis=1)
 lo, hi = np.percentile(means, [2.5, 97.5])
 return est, float(lo), float(hi)


def _coupled_gap_ci(p_series: np.ndarray, m_series: np.ndarray, B: int,
 rng: np.random.Generator) -> tuple[float, float, float]:
 """Coupled bootstrap CI on gap = consensus_acc - p (rule 10: resample
 (p, m) pairs together; both derived from the same runs)."""
 x = np.column_stack([p_series, m_series])
 r = rng.integers(0, len(x), size=(B, len(x)))
 xb = x[r]
 gaps = xb[:, :, 1].mean(axis=1) - xb[:, :, 0].mean(axis=1)
 est = float(np.mean(m_series) - np.mean(p_series))
 lo, hi = np.percentile(gaps, [2.5, 97.5])
 return est, float(lo), float(hi)


def _equidistant_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
 """Bin labels by quantiles (even counts) of x. Returns array of bin ids."""
 if n_bins <= 1:
 return np.zeros(len(x), dtype=int)
 qs = np.linspace(0, 100, n_bins + 1)[1:-1]
 edges = np.percentile(x, qs)
 return np.digitize(x, edges)


def _wilson(ok: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
 """Wilson score CI for a proportion (ok/n). Returns est, lo, hi."""
 if n == 0:
 return float("nan"), float("nan"), float("nan")
 phat = ok / n
 denom = 1 + z * z / n
 center = (phat + z * z / (2 * n)) / denom
 half = (z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
 return float(phat), float(center - half), float(center + half)


def main( -> None:
 args = _parse_args(
 rng = np.random.default_rng(args.seed)

 df = pd.read_parquet(args.input)
 df["alpha"] = df["C"] # self-consistency = n_majority/K
 df["p_run"] = df["A"] # single-sample accuracy per run
 df["m"] = df["majority_is_correct"].astype(float)

 cells: dict = {}
 groups = df.groupby(list(args.group_cols), dropna=False)
 for label, sub in groups:
 label = "|".join(str(x) for x in label)
 p_series = sub["p_run"].values
 m_series = sub["m"].values
 alpha = sub["alpha"].values
 n = int(len(sub))

 p = float(np.mean(p_series))
 cons = float(np.mean(m_series))
 gap = cons - p
 gap_est, gap_lo, gap_hi = _coupled_gap_ci(p_series, m_series, args.bootstrap, rng)

 # (B) agreement-bucketed consensus accuracy
 bins = _equidistant_bins(alpha, args.n_bins)
 bins_acc = []
 for b in range(args.n_bins):
 sel = bins == b
 ok = int(m_series[sel].sum()
 nb = int(sel.sum()
 est, lo, hi = _wilson(ok, nb)
 bins_acc.append({
 "bin": b,
 "alpha_range": [float(alpha[sel].min() if nb else None,
 float(alpha[sel].max() if nb else None],
 "alpha_mid": float(np.mean(alpha[sel])) if nb else None,
 "n_runs": nb,
 "consensus_acc": est, "ci_lo": lo, "ci_hi": hi,
 })

 # highest-agreement bin: the "ceiling" signature (Bahuguna ~half)
 hb = max(bins_acc, key=lambda r: r.get("alpha_mid") or -1) if bins_acc else None

 # per-case consensus gain: majority correct vs single-sample correct,
 # bucketed by per-case p (difficulty) to show backfire on hard questions.
 pcase = sub.groupby("case_id")["p_run"].mean(
 mcase = sub.groupby("case_id")["m"].mean(
 pcases = pcase.values
 mcases = mcase.values
 n_cases = int(len(pcase))
 case_bins = _equidistant_bins(pcases, args.n_bins)
 difficulty = []
 for b in range(args.n_bins):
 sel = case_bins == b
 if sel.sum( == 0:
 continue
 n_ok = int(mcases[sel].sum()
 n_tot = int(sel.sum()
 c_est, c_lo, c_hi = _wilson(n_ok, n_tot)
 p_est, p_lo, p_hi = _wilson(int(np.round(pcases[sel].sum()),
 n_tot)
 # coupled case-level bootstrap CI for the gap (resample cases
 # with replacement; m and p of a sampled case move together)
 m_sel = mcases[sel]
 p_sel = pcases[sel]
 idx = rng.integers(0, len(m_sel), size=(args.bootstrap, len(m_sel)))
 gaps_boot = m_sel[idx].mean(axis=1) - p_sel[idx].mean(axis=1)
 gap_b_lo, gap_b_hi = np.percentile(gaps_boot, [2.5, 97.5])
 difficulty.append({
 "bin": b,
 "case_p_range": [float(pcases[sel].min(), float(pcases[sel].max()],
 "case_p_mid": float(np.mean(pcases[sel])),
 "n_cases": int(sel.sum(),
 "case_consensus_acc": float(np.mean(mcases[sel])),
 "case_consensus_ci": [c_lo, c_hi],
 "case_single_acc": float(np.mean(pcases[sel])),
 "case_gap": float(np.mean(mcases[sel]) - np.mean(pcases[sel])),
 "case_gap_ci": [float(c_lo - p_hi), float(c_hi - p_lo)],
 "case_gap_ci_coupled": [float(gap_b_lo), float(gap_b_hi)],
 })

 cells[label] = {
 "n_runs": n, "n_cases": n_cases,
 "p_single": p, "consensus_acc": cons,
 "backfire_gap": gap_est, "gap_lo": gap_lo, "gap_hi": gap_hi,
 "highest_agreement_bin": hb,
 "agreement_bins": bins_acc,
 "difficulty_bins": difficulty,
 }

 out = {
 "schema_version": "1.0",
 "generated_by": os.path.basename(__file__),
 "args": vars(args),
 "group_cols": args.group_cols,
 "note": "p = single-sample accuracy (Ding 'A'); consensus_acc = "
 "mean(majority_is_correct); backfire_gap = consensus - p (coupled "
 "bootstrap). agreement_bins: runs binned by self-consistency alpha "
 "(Ding 'C'); difficulty_bins: cases binned by per-case p.",
 "cells": cells,
 }
 os.makedirs(os.path.dirname(args.output), exist_ok=True)
 with open(args.output, "w") as f:
 json.dump(out, f, indent=2, sort_keys=False)
 print(f"wrote {args.output}")


def _parse_args( -> argparse.Namespace:
 p = argparse.ArgumentParser(description=__doc__)
 p.add_argument("--input", required=True, help="Path to case_results parquet/csv")
 p.add_argument("--output", required=True, help="Path to write evidence JSON")
 p.add_argument("--seed", type=int, default=0)
 p.add_argument("--bootstrap", type=int, default=10000)
 p.add_argument("--n-bins", type=int, default=5,
 help="Number of agreement / difficulty bins")
 p.add_argument("--group-cols", nargs="+",
 default=["model", "benchmark", "prompt"],
 help="Columns that define a cell")
 return p.parse_args(


if __name__ == "__main__":
 main(
