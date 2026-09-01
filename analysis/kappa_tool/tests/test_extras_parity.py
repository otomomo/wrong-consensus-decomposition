"""Decompose-only extras parity vs results/kappa_decompose.json (full fidelity).

The committed kappa_decompose.json drew its nulls from its own rng sub-stream
(n_sim=200000), so bit-identity is unreachable; the contract is an MC
tolerance (see test_parity.py). This test checks ONLY the decompose-only
extras (no rival pref loop), so it runs in minutes at full fidelity.

Run: cd analysis && python -m kappa_tool.tests.test_extras_parity
Exit code 0 = all cells within tolerance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve(.parent
_TOOL = _HERE.parent
_ANALYSIS = _TOOL.parent
for _p in (_ANALYSIS, _TOOL):
 if str(_p) not in sys.path:
 sys.path.insert(0, str(_p))

from kappa_tool.decompose import Config, decompose_extras # noqa: E402
from kappa_tool.tests.test_parity import (_eq, _tol_ok, load_parity_frames) # noqa: E402

ROOT = _ANALYSIS.parent
RESULTS = ROOT / "results"

EXACT = ["wrong_consensus_emp"]
# tolerances calibrated against measured cross-stream MC noise (20 independent
# seeds): sd(kappa_iid_pooled) ~ 0.027 at n=2e5; 0.10 ~ 3.7 sd (false-fail < 0.1%)
TOL = {
 "kappa_iid_pooled": 0.10,
 "plurality_share": 0.02,
 "share_lo": 0.02,
 "share_hi": 0.02,
 "wrong_consensus_pooled": 0.01,
}


def main( -> int:
 ref = json.load(open(RESULTS / "kappa_decompose.json"))["cells"]
 df, _ = load_parity_frames(
 # committed kappa_decompose.json args: n_sim=200000, share_n_sim=4000,
 # share_bootstrap=2000, seed=0 (extras rng seeded seed+1e6 internally)
 cfg = Config(seed=0, n_sim=200000, share_n_sim=4000, share_bootstrap=2000,
 c_gpqa=4, aime_c_mode="mean_distinct")

 failures = []
 n_cells = 0
 for (model, benchmark, prompt), sub in df.groupby(
 ["model", "benchmark", "prompt"], dropna=False):
 label = "|".join([model, benchmark, prompt])
 sub = sub.copy(
 sub["_correct"] = sub["majority_is_correct"]
 extras = decompose_extras(sub, cfg)
 n_cells += 1
 d = ref[label]
 for f in EXACT:
 if f not in d:
 continue
 if not _eq(extras[f], d[f]):
 failures.append(f"{label} [{f}] wrapper={extras[f]!r} "
 f"ref={d[f]!r}")
 for f, tol in TOL.items(:
 if f not in d:
 continue
 if not _tol_ok(extras[f], d[f], tol):
 failures.append(f"{label} [{f}] wrapper={extras[f]!r} "
 f"ref={d[f]!r} tol={tol}")
 print(f"{label}: ok so far, failures={len(failures)}", flush=True)

 print(f"checked {n_cells} cells; failures: {len(failures)}")
 for f in failures:
 print(" FAIL", f)
 return 1 if failures else 0


if __name__ == "__main__":
 raise SystemExit(main()
