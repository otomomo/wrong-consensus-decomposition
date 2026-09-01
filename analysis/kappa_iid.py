"""i.i.d.-MC null models for the LLM self-consistency kappa decomposition.

numpy-only helpers shared by analysis/llm_selfconsistency.py. Two nulls:

 * simulate_iid_kappa(p, K, C) pooled-p null: every voter correct
 w.p. p and, when wrong, uniform over the C-1 wrong classes.
 * simulate_iid_kappa_perq(p_i, K, C) per-question difficulty-matched null:
 each simulated question draws a per-question accuracy from the empirical
 distribution {p_i}, then K independent voters at that accuracy. This
 isolates the plurality mechanics under the observed difficulty spread;
 the residual versus empirical kappa is the within-question choice
 correlation (shared bias).

Tie-break: np.argmax(counts) (smallest index), matching the project
convention and the i.i.d.-MC used in kappa_decompose.py.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def simulate_iid_kappa(p: float, K: int, C: int, N: int = 100000,
 seed: int = 42) -> Dict[str, float]:
 """i.i.d.-MC null with a single pooled accuracy p for all K voters.

 Option 0 is the correct class w.l.o.g.; a wrong voter answers uniformly
 over classes 1..C-1. Returns E[alpha|correct], E[alpha|wrong] and the
 implied kappa = E[alpha|wrong]*(C-1)/(1-p).
 """
 rng = np.random.RandomState(seed)
 correct = rng.random((N, K)) < p
 preds = np.where(correct, 0, rng.randint(1, C, size=(N, K)))
 counts = np.zeros((N, C))
 for k in range(K):
 counts[np.arange(N), preds[:, k]] += 1
 cons = np.argmax(counts, axis=1)
 cons_correct = (cons == 0)
 alpha = counts[np.arange(N), cons] / K
 e_a_c = float(alpha[cons_correct].mean() if cons_correct.sum( else float("nan")
 e_a_w = float(alpha[~cons_correct].mean() if (~cons_correct).sum( else float("nan")
 p_mean = float(p)
 kappa = e_a_w * (C - 1) / (1 - p_mean) if (1 - p_mean) > 0 else float("inf")
 return {"E_alpha_correct_iid": e_a_c, "E_alpha_wrong_iid": e_a_w,
 "kappa_iid": kappa,
 "consensus_accuracy_iid": float(cons_correct.mean(),
 "wrong_consensus_rate_iid": float((~cons_correct).mean()}


def simulate_iid_kappa_perq(p_list: Sequence[float], K: int, C: int,
 M: int = 200000, seed: int = 42) -> Dict[str, float]:
 """i.i.d.-MC null given question difficulty.

 Each simulated question draws a per-question accuracy p from the empirical
 distribution {p_i}, then K independent voters at that accuracy. The
 residual between the resulting kappa and the empirical kappa is the
 within-question choice correlation (shared bias).
 """
 rng = np.random.RandomState(seed)
 p_list = np.asarray(p_list, dtype=float)
 p = rng.choice(p_list, size=M)
 correct = rng.random((M, K)) < p[:, None]
 preds = np.where(correct, 0, rng.randint(1, C, size=(M, K)))
 counts = np.zeros((M, C))
 for k in range(K):
 counts[np.arange(M), preds[:, k]] += 1
 cons = np.argmax(counts, axis=1)
 cons_correct = (cons == 0)
 alpha = counts[np.arange(M), cons] / K
 e_a_w = float(alpha[~cons_correct].mean() if (~cons_correct).sum( else float("nan")
 p_mean = float(p.mean()
 kappa = e_a_w * (C - 1) / (1 - p_mean) if (1 - p_mean) > 0 else float("inf")
 return {"E_alpha_wrong_iid": e_a_w, "kappa_iid": kappa,
 "consensus_accuracy_iid": float(cons_correct.mean(),
 "wrong_consensus_rate_iid": float((~cons_correct).mean()}
