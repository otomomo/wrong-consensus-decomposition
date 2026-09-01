#!/usr/bin/env bash
# Launch the finite-donor MC calibration on the V100 (56 cores).
# Idempotent: safe to re-run; overwrites results/finite_donor_mc.json.
set -euo pipefail
cd ~/agreement-ceiling
PY=${PY:-$HOME/miniconda3/bin/python}
REPS=${REPS:-300}
NSIM=${NSIM:-8000}
WORKERS=${WORKERS:-56}
OUT=${OUT:-results/finite_donor_mc.json}
LOG=${LOG:-/tmp/fdm_run.log}

echo "=== finite_donor_mc: reps=$REPS n_sim_cal=$NSIM workers=$WORKERS -> $OUT ==="
$PY analysis/finite_donor_mc.py \
  --tier3 data/sampled/tier3/tier3_*.jsonl \
  --include-ding \
  --ding-cases data/raw/case_results_deid.parquet \
  --ding-dist  data/raw/answer_distributions_deid.parquet \
  --ding-model gpt-4.1 \
  --output "$OUT" \
  --reps "$REPS" --n-sim-cal "$NSIM" --seed 0 --workers "$WORKERS"
echo "=== DONE ==="
