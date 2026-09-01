"""Sequencing check: wrapper decompose_cell ≡ kappa_rival_preference.main body.

Runs BOTH the wrapper and the original validated script at a REDUCED n_sim
(--n-sim, default 2000) with the same seed and asserts bit-exact equality of
the rival family per cell. The original script's per-cell rng-call ORDER is
n_sim-independent, so a pass here proves the wrapper feeds the same kernels in
the same order with the same inputs -- which, together with the full
n_sim=1e5 parity check in test_parity.py, closes the bit-exactness contract
against results/kappa_rival_preference.json without re-running the ~4h full
null simulation every time.

The reference is produced by the ORIGINAL script as a subprocess (not by the
wrapper's own imports), so the check is non-circular.

Run: cd analysis && python -m kappa_tool.tests.test_sequencing [--n-sim N]
Exit code 0 = all cells bit-exact.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve(.parent
_TOOL = _HERE.parent
_ANALYSIS = _TOOL.parent
for _p in (_ANALYSIS, _TOOL):
 if str(_p) not in sys.path:
 sys.path.insert(0, str(_p))

from kappa_tool.decompose import Config, decompose_cell # noqa: E402
from kappa_tool.tests.test_parity import RIVAL_EXACT, _eq, load_parity_frames # noqa: E402

ROOT = _ANALYSIS.parent
DATA = ROOT / "data" / "raw"


def main( -> int:
 ap = argparse.ArgumentParser(
 ap.add_argument("--n-sim", type=int, default=2000)
 args = ap.parse_args(

 df, dist = load_parity_frames(
 cfg = Config(seed=0, bootstrap=10000, n_sim=args.n_sim, min_wrong=1,
 shrink=1.0, c_gpqa=4, aime_c_mode="mean_distinct",
 tie_break="argmin")

 # reference: the original validated script, same seed / reduced n_sim
 with tempfile.TemporaryDirectory( as tmp:
 ref_path = Path(tmp) / "rival_ref.json"
 cmd = [
 sys.executable, str(_ANALYSIS / "kappa_rival_preference.py"),
 "--input-cases", str(DATA / "case_results_deid.parquet"),
 "--input-dist", str(DATA / "answer_distributions_deid.parquet"),
 "--output", str(ref_path),
 "--seed", "0", "--bootstrap", "10000",
 "--n-sim", str(args.n_sim), "--min-wrong", "1",
 "--shrink", "1.0", "--tie-break", "argmin",
 ]
 subprocess.run(cmd, check=True)
 ref = json.load(open(ref_path))
 ref_by_key = {(c["model"], c["benchmark"], c["prompt"]): c
 for c in ref["cells"]}

 rng = np.random.default_rng(cfg.seed)
 failures = []
 n_cells = 0
 for (model, benchmark, prompt), sub in df.groupby(
 ["model", "benchmark", "prompt"], dropna=False):
 label = "|".join([model, benchmark, prompt])
 dsub = dist[(dist["model"] == model) & (dist["benchmark"] == benchmark) &
 (dist["prompt"] == prompt)].copy(
 row = decompose_cell(sub, dsub, cfg, label, rng, run_key="student_id")
 n_cells += 1
 r = ref_by_key[(model, benchmark, prompt)]
 for f in RIVAL_EXACT:
 if f not in r:
 continue
 if not _eq(row[f], r[f]):
 failures.append(
 f"{label} [{f}] wrapper={row[f]!r} ref={r[f]!r}")
 print(f"{label}: {len(failures)} failures so far", flush=True)

 print(f"checked {n_cells} cells at n_sim={args.n_sim}; "
 f"failures: {len(failures)}")
 for f in failures:
 print(" FAIL", f)
 return 1 if failures else 0


if __name__ == "__main__":
 raise SystemExit(main()
