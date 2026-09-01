#!/usr/bin/env python3
"""Join the observed phi (from the committed evidence JSONs) into the
finite-donor calibration result.

The calibration (finite_donor_mc.py) is run on a compute box that may not have
the committed results/ JSONs, so it emits observed_phi=nan. This script re-attaches
the observed share_explained_mech from results/tier3_kappa.json (dict-of-cells)
and results/kappa_rival_preference.json (list-of-cells), keyed by
model|benchmark|prompt, and recomputes the two derived boolean fields.

Usage:
 python3 analysis/attach_observed_phi.py \
 --fdm results/finite_donor_mc.json \
 --obs-tier3 results/tier3_kappa.json \
 --obs-ding results/kappa_rival_preference.json \
 [--in-place]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _load(path: str):
 try:
 return json.load(open(path))
 except (FileNotFoundError, json.JSONDecodeError, ValueError):
 return None


def _obs_index(path: str) -> dict:
 """Return {model|benchmark|prompt: share_explained_mech} from an evidence file."""
 d = _load(path)
 idx = {}
 if d is None:
 return idx
 cells = d.get("cells", {})
 if isinstance(cells, dict):
 for key, row in cells.items(:
 m = row.get("model") or key.split("|")[0]
 b = row.get("benchmark") or key.split("|")[1]
 pr = row.get("prompt") or (key.split("|")[2] if key.count("|") >= 2 else "zero_shot")
 v = row.get("share_explained_mech")
 if v is not None and np.isfinite(v):
 idx[f"{m}|{b}|{pr}"] = float(v)
 else:
 for row in cells:
 m, b, pr = row.get("model"), row.get("benchmark"), row.get("prompt")
 v = row.get("share_explained_mech")
 if None not in (m, b, pr) and v is not None and np.isfinite(v):
 idx[f"{m}|{b}|{pr}"] = float(v)
 return idx


def main( -> None:
 ap = argparse.ArgumentParser(description=__doc__,
 formatter_class=argparse.RawDescriptionHelpFormatter)
 ap.add_argument("--fdm", required=True, help="finite_donor_mc.json to patch")
 ap.add_argument("--obs-tier3", default="results/tier3_kappa.json")
 ap.add_argument("--obs-ding", default="results/kappa_rival_preference.json")
 ap.add_argument("--out", default=None, help="output path (default: --fdm in place)")
 args = ap.parse_args(

 obs = {}
 obs.update(_obs_index(args.obs_tier3))
 obs.update(_obs_index(args.obs_ding))

 d = json.load(open(args.fdm))
 n_filled = 0
 for label, cell in d.get("cells", {}).items(:
 if label in obs:
 cell["observed_phi"] = obs[label]
 fin, ora = cell.get("phi_finite", {}), cell.get("phi_oracle", {})
 observed = obs[label]
 if fin.get("n") and ora.get("n"):
 bias_mean = fin.get("mean", np.nan) - ora.get("mean", np.nan)
 cell["observed_within_oracle_plus_bias_band"] = bool(
 ora.get("p05", np.nan) <= observed <=
 ora.get("p99", np.nan) + bias_mean)
 cell["observed_within_finite_band"] = bool(
 fin.get("p05", np.nan) <= observed <= fin.get("p95", np.nan))
 n_filled += 1
 print(f"[attach] {label}: observed={observed:.4f}")
 else:
 print(f"[attach] {label}: NO observed phi found (left as-is)")

 out = args.out or args.fdm
 with open(out, "w") as f:
 json.dump(d, f, indent=2, default=str)
 print(f"filled {n_filled}/{len(d.get('cells', {}))} cells; wrote {out}")


if __name__ == "__main__":
 main(
