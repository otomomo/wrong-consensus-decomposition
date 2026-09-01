"""Per-sample loading and aggregation for kappa_tool.

The public input contract is a bare per-sample table with at least:

 case_id question / problem id (a run-cluster unit)
 run_id one self-consistency sampling run of the case
 answer the token/option the sample produced
 is_correct bool (can be omitted if ``ground_truth`` is present)

Optional columns used for grouping / for the leak-free rival null:

 ground_truth the case's correct answer (if absent, derived from the
 set of answers observed as correct within the case)
 model, benchmark, prompt grouping columns that define a "cell"
 (default single-cell when absent)

``aggregate_to_runs`` collapses samples to per-(case_id, run_id) rows in the
same shape the Ding 2026 tables carry (A = single-sample accuracy, C = alpha =
self-consistency, majority label/flag, n_distinct_answers, answer_counts as a
JSON string) so that decompose.py can reuse the validated math 1:1.

Tie-break for the majority label : the label with the
maximum vote count, ties broken toward the smallest label (lexicographic order
over the sorted distinct labels) — the string analogue of ``np.argmax``.

Ground truth derivation: if a ``ground_truth`` column is present it is used
verbatim (per case, must be constant). Otherwise it is inferred as the answer
value observed with is_correct=True inside the case; a case with no verifiably
correct sample has unknown gt. Rows without a usable gt are still valid for the
mechanical decomposition (which needs only A and majority_is_correct) but are
excluded from the answer-preference (rival) null, which needs gt to define the
correct option. This boundary is documented in README.md.
"""
from __future__ import annotations

import ast
import json
from collections import Counter
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLS = ("case_id", "run_id", "answer")
OPTIONAL_COLS = ("is_correct", "ground_truth", "model", "benchmark", "prompt")


def _unique_or_raise(values: Iterable, what: str) -> Optional[str]:
 vals = [str(v) for v in dict.fromkeys(values) if pd.notna(v)]
 uniq = set(vals)
 if len(uniq) == 0:
 return None
 if len(uniq) > 1:
 raise ValueError(
 f"column '{what}' is not constant within a case: {sorted(uniq)}")
 return vals[0]


def derive_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
 """Attach per-case ground_truth, derived from correct samples if absent.

 Returns a new DataFrame with a ``ground_truth`` column (object, may be
 NaN). If the input already has a usable ground_truth, it is trusted.
 """
 out = df.copy(
 if "ground_truth" in out.columns and out["ground_truth"].notna(.any(:
 gt = out.groupby("case_id")["ground_truth"].transform(
 lambda s: _unique_or_raise(s, "ground_truth"))
 out["ground_truth"] = gt
 return out

 if "is_correct" not in out.columns:
 raise ValueError(
 "cannot derive ground_truth: need either a 'ground_truth' column "
 "or an 'is_correct' column from which to infer the correct answer")

 def _infer(s):
 correct = s.loc[s["is_correct"].astype(bool), "answer"]
 if len(correct) == 0:
 return np.nan
 return _unique_or_raise(correct, "answer (among correct samples)")

 out["ground_truth"] = out.groupby("case_id", group_keys=False).apply(_infer)
 return out


def _majority_label(counts: Counter, order: Dict[str, int]) -> str:
 """Plurality label with rule-19 tie-break: smallest option id.

 Option ids are first-occurrence positions within the run (the string
 analogue of np.argmax over integer class ids). Ties break toward the
 option that appeared first in the samples, matching the validated
 scripts' integer argmin convention.
 """
 best = max(counts.values()
 top = [k for k, v in counts.items( if v == best]
 return min(top, key=lambda k: order.get(k, 1 << 30))


def aggregate_to_runs(df: pd.DataFrame) -> pd.DataFrame:
 """Collapse per-sample rows to per-(case_id, run_id) rows.

 Each output row carries (mirroring the Ding tables so the validated math
 applies unchanged): case_id, run_id, K, A (= n_correct/K), C (= alpha =
 n_majority/K), ground_truth, majority_label, majority_is_correct,
 n_distinct_answers, and answer_counts (JSON string of the label histogram,
 unparseable/misc labels kept verbatim as voters for the majority).
 """
 if any(c not in df.columns for c in REQUIRED_COLS):
 raise ValueError(f"missing required columns; need at least {REQUIRED_COLS}")

 if "is_correct" in df.columns:
 correct = df["is_correct"].astype(bool)
 else:
 if "ground_truth" not in df.columns:
 raise ValueError("need an 'is_correct' or 'ground_truth' column")
 gts = df.set_index(df.index, drop=False)["ground_truth"]
 correct = df["answer"].astype(str) == gts.astype(str)

 # per-(case_id, run_id) aggregation
 def _agg(g):
 K = int(len(g))
 n_correct = int(correct.loc[g.index].sum()
 counts = Counter(g["answer"].astype(str).tolist()
 if "option_id" in g.columns:
 # explicit option ids (rule 19: smallest id wins ties)
 order = {str(a): int(o) for a, o in
 zip(g["answer"].astype(str), g["option_id"])}
 else:
 # no ids recorded: first-occurrence order within the run
 order = {}
 for a in g["answer"].astype(str):
 if a not in order:
 order[a] = len(order)
 maj = _majority_label(counts, order)
 n_maj = counts[maj]
 gt = g["ground_truth"].iloc[0] if "ground_truth" in g.columns else np.nan
 maj_correct = bool(pd.notna(gt) and maj == str(gt))
 row = {
 "case_id": g["case_id"].iloc[0],
 "run_id": str(g["run_id"].iloc[0]),
 "K": K,
 "A": float(n_correct) / K,
 "C": float(n_maj) / K,
 "ground_truth": gt,
 "majority_label": maj,
 "majority_is_correct": maj_correct,
 "n_distinct_answers": len(counts),
 "answer_counts": json.dumps({k: v for k, v in
 sorted(counts.items(, key=lambda kv: kv[0])},
 sort_keys=True),
 }
 for col in ("model", "benchmark", "prompt"):
 if col in df.columns and df[col].notna(.any(:
 row[col] = g[col].iloc[0]
 return pd.Series(row)

 runs = (df.groupby(["case_id", "run_id"], dropna=False, group_keys=False)
 .apply(_agg)
 .reset_index(drop=True))
 return runs


def split_tables(runs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
 """Split aggregated runs into the two Ding-schema frames.

 Returns (case_results_like, answer_distributions_like): the former is the
 per-run frame with A/C/ground_truth/majority_is_correct/n_distinct_answers;
 the latter keeps answer_counts plus the same aligned key/scalar columns so
 decompose.py's rival pref needs (case,run)->counts and gt over case/run.
 """
 keys = [c for c in ("model", "benchmark", "prompt") if c in runs.columns]
 cols = keys + ["case_id", "run_id", "K", "A", "C", "ground_truth",
 "majority_is_correct", "n_distinct_answers"]
 cases = runs[cols].copy(
 dist_cols = keys + ["case_id", "run_id", "K", "A", "C", "ground_truth",
 "majority_is_correct", "answer_counts"]
 dist = runs[dist_cols].copy(
 return cases, dist


def load_samples(path: str, **read_kwargs) -> pd.DataFrame:
 """Load a per-sample table from .parquet / .csv / .json / .jsonl.

 For CSV, the string label columns (answer, ground_truth) are read
 with dtype=str: pandas' default dtype inference coerces an
 all-numeric answer column to float64, which rewrites "704" to
 "704.0" and silently breaks string equality against ground truth
 (caught by the 2026-08-22 experiment audit on two all-numeric AIME
 cells). The parquet (Ding) path is untouched and remains bit-exact.
 """
 low = path.lower(
 if low.endswith(".parquet"):
 df = pd.read_parquet(path, **read_kwargs)
 elif low.endswith(".csv"):
 cols = pd.read_csv(path, nrows=0, **read_kwargs).columns
 dtypes = {c: str for c in ("answer", "ground_truth") if c in cols}
 kwargs = dict(read_kwargs)
 if "dtype" in kwargs:
 if isinstance(kwargs["dtype"], dict):
 kwargs["dtype"] = {**kwargs["dtype"], **dtypes}
 else:
 kwargs["dtype"] = dtypes
 df = pd.read_csv(path, **kwargs)
 elif low.endswith(".jsonl"):
 df = pd.read_json(path, lines=True, **read_kwargs)
 elif low.endswith(".json"):
 df = pd.read_json(path, **read_kwargs)
 else:
 # fall back to parquet / csv detection
 return pd.read_csv(path) if low.endswith(".csv") else pd.read_parquet(path)
 if any(c not in df.columns for c in REQUIRED_COLS):
 raise ValueError(
 f"missing required columns {REQUIRED_COLS}; got columns"
 f"{list(df.columns)}")
 return df.reset_index(drop=True)