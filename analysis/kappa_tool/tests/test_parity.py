"""Parity test: kappa_tool reproduces the committed results/*.json numbers.

Contract (decompose.py docstring is normative):

 * rival family vs results/kappa_rival_preference.json -> BIT-EXACT (==)
 * deterministic scalars vs both committed files -> BIT-EXACT (==)
 * decompose-only MC family vs results/kappa_decompose.json -> MC-TOLERATED

The two committed files draw their MC nulls from incompatible independent rng
sub-streams (each script seeds its own default_rng(0) and advances it through
different call sequences), so bit-identity is reachable only for the rival
family; the decompose-only family is asserted within a documented absolute
tolerance calibrated against observed cross-stream MC noise at n_sim=1e5
(max |delta| ~ 0.02 for kappa; safety factor >= 2).

Run: cd analysis && python -m kappa_tool.tests.test_parity
Exit code 0 = all cells pass.

Optional: python -m kappa_tool.tests.test_parity --n-sim 2000 (fast smoke run:
checks structure + the deterministic DECOMP_EXACT fields only, since a
different n_sim draws a different rng sub-stream).

EXPECTED RUNTIME at the committed n_sim=1e5: several hours on this box.
The bottleneck is iid_mc_plurality_pref's O(n*K*n_c) counts loop with
per-case/pooled preferences of up to ~250-400 distinct labels (AIME open
answers); this is the same cost the committed evidence paid when
kappa_rival_preference.py originally ran. The kernels are frozen (committed
evidence producers) and reused verbatim, so the runtime is inherent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve(.parent # .../analysis/kappa_tool/tests
_TOOL = _HERE.parent # .../analysis/kappa_tool
_ANALYSIS = _TOOL.parent # .../analysis
for _p in (_ANALYSIS, _TOOL):
 if str(_p) not in sys.path:
 sys.path.insert(0, str(_p))

from kappa_tool.decompose import Config, decompose_cell # noqa: E402

ROOT = _ANALYSIS.parent
DATA = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

# fields bit-exact vs kappa_rival_preference.json (same rng sub-stream)
RIVAL_EXACT = [
 "kappa_empirical",
 "kappa_empirical_ci",
 "kappa_empirical_ci_clustered",
 "kappa_empirical_subset",
 "kappa_empirical_subset_ci_clustered",
 "kappa_iid_perq",
 "kappa_rival_case",
 "kappa_rival_case_ci",
 "kappa_rival_case_ci_clustered",
 "kappa_rival_pool",
 "kappa_rival_pool_ci",
 "mechanical_addon_case",
 "shared_residual_case",
 "share_explained_mech",
 "share_explained_mech_ci",
 "share_explained_mech_ci_clustered",
 "mc_se_rival_case",
 "mc_se_rival_pool",
 "n_test_runs",
 "n_test_with_draw",
 "n_test_cases",
]

# fields bit-exact vs kappa_decompose.json (deterministic; no rng involved)
DECOMP_EXACT = [
 "p", "consensus_acc", "C_options", "K", "n_runs", "n_wrong", "n_cases",
 "E_alpha_given_wrong", "kappa_empirical", "wrong_consensus_emp",
]

# fields MC-tolerated vs kappa_decompose.json (independent re-draws).
# Tolerances calibrated against measured cross-stream MC noise (20 independent
# seeds): sd(kappa_iid_perq) ~ 0.006 at n=2e5, sd(kappa_iid_pooled) ~ 0.027 at
# n=2e5 (~0.038 at n=1e5); 0.15 ~ 4x the n=1e5 pooled sd (false-fail < 0.1%).
DECOMP_TOL = {
 "kappa_iid_perq": 0.05,
 "kappa_iid_pooled": 0.15,
 "plurality_share": 0.02,
 "share_lo": 0.02,
 "share_hi": 0.02,
 "wrong_consensus_perq": 0.01,
 "wrong_consensus_pooled": 0.01,
}
# k_emp bootstrap CI in kappa_decompose.json == wrapper's kappa_empirical_ci
# (same statistic, different rng stream -> MC-tolerated elementwise)
K_EMP_CI_TOL = 0.05


def _eq(a, b) -> bool:
 """Exact equality with NaN == NaN."""
 if isinstance(a, list) and isinstance(b, list):
 return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
 try:
 return bool(a == b) or (bool(np.isnan(a)) and bool(np.isnan(b)))
 except TypeError:
 return bool(a == b)


def _tol_ok(a, b, tol: float) -> bool:
 try:
 if bool(np.isnan(a)) and bool(np.isnan(b)):
 return True
 return abs(float(a) - float(b)) <= tol
 except (TypeError, ValueError):
 return False


def load_parity_frames(:
 """Mirror kappa_rival_preference.main's data prep (argmin tie-break)."""
 df = pd.read_parquet(DATA / "case_results_deid.parquet")
 df["majority_is_correct"] = df["majority_is_correct"].astype(bool)
 df["A"] = df["A"].astype(float)
 df["C"] = df["C"].astype(float)
 df["n_distinct_answers"] = df["n_distinct_answers"].astype(int)
 mp = df.drop_duplicates(["axis", "condition", "model", "prompt"])[
 ["axis", "condition", "model", "prompt"]]
 dist = pd.read_parquet(DATA / "answer_distributions_deid.parquet")
 dist = dist.merge(mp, on=["axis", "condition"], how="left")
 df["_correct"] = df["majority_is_correct"]
 dist["_correct"] = dist["majority_is_correct"].astype(bool)
 return df, dist


def main( -> int:
 ap = argparse.ArgumentParser(
 ap.add_argument("--n-sim", type=int, default=100000,
 help="MC draws per null; MUST be 100000 for the committed "
 "bit-exact parity (lower = smoke run, MC checks skipped)")
 args = ap.parse_args(

 ref_riv = json.load(open(RESULTS / "kappa_rival_preference.json"))
 ref_dec = json.load(open(RESULTS / "kappa_decompose.json"))
 riv_by_key = {(c["model"], c["benchmark"], c["prompt"]): c
 for c in ref_riv["cells"]}
 dec_by_label = ref_dec["cells"] # dict keyed by label

 df, dist = load_parity_frames(
 # committed kappa_rival_preference.json args (seed=0, bootstrap=1e4,
 # n_sim=1e5, min_wrong=1, shrink=1.0, c_gpqa=4, aime mean_distinct, argmin)
 cfg = Config(seed=0, bootstrap=10000, n_sim=args.n_sim, min_wrong=1,
 shrink=1.0, c_gpqa=4, aime_c_mode="mean_distinct",
 tie_break="argmin")
 full_mc = args.n_sim == 100000
 rng = np.random.default_rng(cfg.seed)

 failures = []
 n_cells = 0
 for (model, benchmark, prompt), sub in df.groupby(
 ["model", "benchmark", "prompt"], dropna=False):
 label = "|".join([model, benchmark, prompt])
 t0 = time.time(
 dsub = dist[(dist["model"] == model) & (dist["benchmark"] == benchmark) &
 (dist["prompt"] == prompt)].copy(
 row = decompose_cell(sub, dsub, cfg, label, rng, run_key="student_id")
 n_cells += 1

 if full_mc:
 r = riv_by_key[(model, benchmark, prompt)]
 for f in RIVAL_EXACT:
 if f not in r:
 continue
 if not _eq(row[f], r[f]):
 failures.append(
 f"{label} [{f}] wrapper={row[f]!r} ref_rival={r[f]!r}")

 d = dec_by_label[label]
 for f in DECOMP_EXACT:
 if f not in d:
 continue
 if not _eq(row[f], d[f]):
 failures.append(
 f"{label} [{f}] wrapper={row[f]!r} ref_decomp={d[f]!r}")
 if full_mc:
 for f, tol in DECOMP_TOL.items(:
 if f not in d:
 continue
 if not _tol_ok(row[f], d[f], tol):
 failures.append(
 f"{label} [{f}] wrapper={row[f]!r} ref_decomp={d[f]!r} "
 f"tol={tol}")
 if "k_emp_lo" in d and "k_emp_hi" in d:
 for i, f in enumerate(("k_emp_lo", "k_emp_hi")):
 if not _tol_ok(row["kappa_empirical_ci"][i], d[f],
 K_EMP_CI_TOL):
 failures.append(
 f"{label} [{f}] wrapper_ci={row['kappa_empirical_ci'][i]!r} "
 f"ref_decomp={d[f]!r} tol={K_EMP_CI_TOL}")
 print(f"[{time.time( - t0:6.1f}s] {label} done "
 f"(failures so far: {len(failures)})", flush=True)

 print(f"checked {n_cells} cells; failures: {len(failures)}")
 for f in failures:
 print(" FAIL", f)
 return 1 if failures else 0


if __name__ == "__main__":
 raise SystemExit(main()
