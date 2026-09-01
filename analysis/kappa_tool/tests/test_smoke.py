"""End-to-end smoke test of the per-sample path (fast, synthetic data).

Builds a tiny per-sample long table, runs kappa_tool.decompose in per-sample
mode with reduced MC sizes, and asserts the output envelope: all headline
fields present, kappa_empirical finite and deterministic across the two entry
points (per-sample vs pre-aggregated runs), plurality_share well-defined.

Run: cd analysis && python -m kappa_tool.tests.test_smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve(.parent
_TOOL = _HERE.parent
_ANALYSIS = _TOOL.parent
for _p in (_ANALYSIS, _TOOL):
 if str(_p) not in sys.path:
 sys.path.insert(0, str(_p))

from kappa_tool import load # noqa: E402
from kappa_tool.decompose import Config, decompose # noqa: E402

REQUIRED = [
 "p", "consensus_acc", "C_options", "K", "n_runs", "n_wrong", "n_cases",
 "E_alpha_given_wrong", "kappa_empirical", "kappa_iid_perq",
 "kappa_iid_pooled", "kappa_rival_case", "kappa_rival_pool",
 "kappa_empirical_subset", "mechanical_addon_case", "shared_residual_case",
 "plurality_share", "share_explained_mech",
 "wrong_consensus_emp", "wrong_consensus_perq", "wrong_consensus_pooled",
 "n_test_runs", "n_test_with_draw", "n_test_cases",
]


def _synthetic( -> pd.DataFrame:
 rng = np.random.default_rng(7)
 rows = []
 n_cases, n_runs, K, C = 6, 8, 12, 4
 for ci in range(n_cases):
 gt = str(rng.integers(0, C))
 for ri in range(n_runs):
 p_correct = 0.25 + 0.5 * (ci % 3) / 3
 ans = []
 ok = []
 for _ in range(K):
 if rng.random( < p_correct:
 ans.append(gt)
 ok.append(True)
 else:
 ans.append(str(rng.integers(0, C)))
 ok.append(False)
 rows.extend(
 {"case_id": f"c{ci}", "run_id": f"c{ci}_r{ri}",
 "answer": a, "is_correct": c, "ground_truth": gt,
 "model": "m0", "benchmark": "synthetic", "prompt": "zero_shot"}
 for a, c in zip(ans, ok))
 return pd.DataFrame(rows)


def main( -> int:
 df = _synthetic(
 cfg = Config(seed=0, n_sim=500, bootstrap=100, share_bootstrap=50,
 min_wrong=1)
 out = decompose(df, cfg, per_sample=True)
 cells = out["cells"]
 assert len(cells) == 1, cells.keys(
 label, row = next(iter(cells.items())
 assert label == "m0|synthetic|zero_shot", label

 for f in REQUIRED:
 assert f in row, f"missing field {f}"
 assert np.isfinite(row["kappa_empirical"])
 assert np.isfinite(row["kappa_iid_perq"])
 assert np.isfinite(row["kappa_rival_case"])
 assert 0.0 < row["plurality_share"] < 10.0
 assert row["n_wrong"] > 0 and row["n_test_runs"] > 0
 print("per-sample path OK:", {k: round(v, 4) if isinstance(v, float) else v
 for k, v in row.items(
 if k in ("p", "C_options", "kappa_empirical",
 "kappa_iid_perq", "kappa_rival_case",
 "plurality_share", "n_test_runs")})

 # same numbers via the pre-aggregated entry point
 runs = load.derive_ground_truth(df)
 runs = load.aggregate_to_runs(runs)
 out2 = decompose(runs, cfg, per_sample=False)
 row2 = next(iter(out2["cells"].values())
 for f in ("p", "C_options", "K", "n_runs", "n_wrong", "n_cases",
 "E_alpha_given_wrong", "kappa_empirical",
 "wrong_consensus_emp"):
 assert row[f] == row2[f], (f, row[f], row2[f])
 print("pre-aggregated path agrees on deterministic fields: OK")
 return 0


if __name__ == "__main__":
 raise SystemExit(main()
