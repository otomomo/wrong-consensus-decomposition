#!/usr/bin/env python3
"""Reproducibility check: re-run the canonical Tier-3 decomposition for every
committed cell straight from the in-repo raw votes (data/sampled/tier3/*.jsonl)
and diff the headline quantities against results/tier3_kappa.json.

This is the permanent replication test for the Tier-3 table. It uses ONLY
committed code (tier3_to_samples.py + kappa_tool) and committed raw data. It
does not touch /tmp/ac_work (the legacy transient CSVs that tier3_kappa.py
pointed at); it regenerates the per-sample CSV in a temp dir from the JSONL,
exactly as tier3_kappa_append.py does.

Canonical precision (must match the committed evidence):
 seed=0, n_sim=1e5, bootstrap=1e4, c_fixed=4 (gpqa) / None->mean_distinct (aime),
 c_gpqa=4, min_wrong=1, shrink=1.0, tie_break=argmin.

Verdict rule: a cell MATCHES if p, kappa_empirical, kappa_iid_perq,
kappa_rival_case, share_explained_mech agree to 4 decimals AND the
share_explained_mech CI agrees to 3 decimals. Any other difference -> MISMATCH.
Exit code 0 iff all cells match.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
 sys.path.insert(0, _HERE)
ROOT = os.path.dirname(_HERE)
from kappa_tool import load
from kappa_tool.decompose import Config, decompose


def discover_cells( -> list:
 """(jsonl_path, model_label, benchmark, c_fixed) for every tier-3 jsonl."""
 cells = []
 for p in sorted(glob.glob(os.path.join(ROOT, "data", "sampled", "tier3", "tier3_*.jsonl"))):
 stem = os.path.basename(p)
 stem = stem[len("tier3_"):-len(".jsonl")]
 bench = "aime" if stem.endswith("_aime") else ("gpqa" if stem.endswith("_gpqa") else None)
 if bench is None:
 continue
 model_file = stem[: -(len(bench) + 1)]
 model_label = model_file.replace(":", "-") # tier3 label uses dashes
 bench_label = "gpqa_diamond" if bench == "gpqa" else "aime"
 c_fixed = 4 if bench == "gpqa" else None
 cells.append((p, model_label, bench_label, c_fixed))
 return cells


def to_samples_csv(jsonl_path: str, out_csv: str, model_label: str, bench_label: str) -> None:
 import subprocess
 cmd = [sys.executable, os.path.join(_HERE, "tier3_to_samples.py"),
 "--input", jsonl_path, "--output", out_csv,
 "--model", model_label, "--benchmark", bench_label]
 r = subprocess.run(cmd, capture_output=True, text=True)
 if r.returncode != 0:
 raise RuntimeError(f"tier3_to_samples failed for {jsonl_path}:\n{r.stderr}")


def close(a, b, tol):
 try:
 return abs(float(a) - float(b)) <= tol
 except (TypeError, ValueError):
 return float(a) == float(b) # both nan


def main( -> int:
 canon = json.load(open(os.path.join(ROOT, "results", "tier3_kappa.json")))["cells"]
 cells = discover_cells(
 print(f"reproducing {len(cells)} cells from in-repo JSONL "
 f"(canonical evidence holds {len(canon)} cells)\n")
 tmp = tempfile.mkdtemp(prefix="t3repro_")
 n_match = 0
 rows = []
 for jsonl_path, model, bench, cfix in cells:
 label = f"{model}|{bench}|zero_shot"
 out_csv = os.path.join(tmp, f"t3_{model}_{bench}.csv")
 to_samples_csv(jsonl_path, out_csv, model, bench)
 cfg = Config(seed=0, n_sim=100000, bootstrap=10000, c_fixed=cfix, c_gpqa=4)
 d = decompose(load.load_samples(out_csv), cfg, per_sample=True)
 row = next(iter(d["cells"].values())

 ref = canon.get(label)
 if ref is None:
 rows.append((label, "MISSING_FROM_EVIDENCE", None, None, None))
 continue
 checks = [
 ("p", row["p"], ref["p"], 1e-4),
 ("kappa_emp", row["kappa_empirical"], ref["kappa_empirical"], 1e-4),
 ("kappa_iid", row["kappa_iid_perq"], ref["kappa_iid_perq"], 1e-4),
 ("kappa_rival", row["kappa_rival_case"], ref["kappa_rival_case"], 1e-4),
 ("phi", row["share_explained_mech"], ref["share_explained_mech"], 1e-4),
 ]
 ci_ok = (close(row["share_explained_mech_ci"][0], ref["share_explained_mech_ci"][0], 1e-3)
 and close(row["share_explained_mech_ci"][1], ref["share_explained_mech_ci"][1], 1e-3))
 all_ok = all(close(g, r, t) for _, g, r, t in checks) and ci_ok
 if all_ok:
 n_match += 1
 detail = (f"phi={row['share_explained_mech']:.4f} vs {ref['share_explained_mech']:.4f} "
 f"p={row['p']:.4f} vs {ref['p']:.4f} ci={[round(x,3) for x in row['share_explained_mech_ci']]} "
 f"vs {[round(x,3) for x in ref['share_explained_mech_ci']]}")
 rows.append((label, "MATCH" if all_ok else "MISMATCH", detail, row, ref))
 print(f"[{('OK ' if all_ok else 'BAD')}] {label}\n {detail}")

 shutil.rmtree(tmp, ignore_errors=True)

 print(f"\n=== VERDICT: {n_match}/{len(cells)} cells reproduced ===")
 bad = [r for r in rows if r[1] not in ("MATCH",)]
 if bad:
 print("FAILURES:")
 for label, status, detail, *_ in bad:
 print(f" {status}: {label} {detail}")
 return 1
 print("ALL CELLS REPRODUCE FROM IN-REPO RAW DATA WITH COMMITTED CODE.")
 return 0


if __name__ == "__main__":
 sys.exit(main()
