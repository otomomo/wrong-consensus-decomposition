# LLM self-consistency kappa decomposition (Qwen3.5-9B)

**Data + analysis for the LLM self-consistency arm of the kappa-plurality
decomposition.** Samples exchangeable rollouts of one model per question and
treats them as "voters"; the plurality answer is the consensus; agreement is
the fraction of samples voting the consensus. This is the same protocol as the
vision ensembles and the GPT-4.1 per-run cells, applied to a second model
(Qwen3.5-9B) on multiple-choice benchmarks with fixed class count C.

The numbers below are produced from the committed JSONL in this directory by
`analysis/llm_selfconsistency.py` (numpy-only). Do not hand-edit any value;
the canonical evidence file is
`results/anchoring_llm_selfconsistency_report.{json,txt}`.

## Data (`data/sampled/per_llm/`)

| File | Dataset | Questions | C | K |
|---|---|---|---|---|
| `llm_sc_qwen3.5-9b-ctx4k_mmlu_16.jsonl` | MMLU test (500 shuffled, seed 42) | 500 | 4 | 16 |
| `llm_sc_qwen3.5-9b-ctx4k_mmlu_pro_32.jsonl` | MMLU-Pro validation | 70 | 10 | 32 |

Schema per line: `{idx, gt, C, question, options, votes, n_valid}` where
`votes` is a length-K list of option indices (`-1` = unparseable), `n_valid` =
count of parseable votes. 570 records total.

## Key numbers (from the canonical report)

| | MMLU-Pro val | MMLU test |
|---|---|---|
| Questions / C / K | 70 / 10 / 32 | 500 / 4 / 16 |
| Sample accuracy p | 0.561 | 0.744 |
| Consensus accuracy | 0.600 | 0.760 |
| Wrong-consensus rate: empirical | **40.0%** | **24.0%** |
| — pooled-p i.i.d. null | 0.0% (Condorcet) | 0.04% |
| — per-question difficulty-matched null | 34.1% | 21.0% |
| kappa empirical | 14.16 | 9.18 |
| kappa i.i.d. per-question | 4.15 | 5.10 |
| **Plurality share (kappa_iid_perq / kappa_emp)** | **29%** | **56%** |
| Highest agreement bin accuracy | 85.7% (n=28) | 90.8% (n=314) |

**Reading.** The pooled-p i.i.d. null gives a ~0 wrong-consensus rate (Condorcet
— a single biased-but-accurate voter is enough to win plurality), yet real
data shows 24-40% wrong consensus. The per-question difficulty-matched null
(accounts for easy/hard question spread) explains ~85% of that gap. The
conditional kappa decomposition yields share 29% on the hard high-C set
(shared-bias dominated) vs 56% on the easy set (plurality dominated) — the
difficulty-graded analogue of the family-conditional split in the vision data.

**Honest contrast with Bahuguna 2026.** The highest-agreement bins remain
86-91% accurate here; we do not observe the near-chance top bin reported for
small models on GPQA. So the "confidence can backfire" failure is
regime-dependent: it appears where within-question choice correlation is
strong relative to question difficulty (low share), not where plurality
mechanics dominate (high share).

## Protocol (for replication)

- Model: Qwen3.5-9B (`qwen3.5-9b` ctx4k variant, 4-bit), thinking disabled
  (`think: false`), T=0.7, `num_predict=64`, OLLAMA_NUM_PARALLEL=4.
- Parsing: "Answer/option/choice" prefix pattern, fallback to last standalone
  option letter; measured parse rate **99.8%**.
- Nulls: pooled-p via `simulate_iid_kappa(p, K, C, seed 42, N=100000)`;
  per-question via `simulate_iid_kappa_perq(p_i, K, C, M=200000, seed 42)`
  (in `analysis/kappa_iid.py`); ties to lowest index via argmax on counts
  (matches the i.i.d.-MC convention).
- Note on sampling: pooled accuracy p used in the null is the mean over
  questions of the within-question sample accuracy (the single-sample
  accuracy convention used throughout this repository).

## Reproduce / extend

```bash
# Re-run the analysis on the shipped data (numpy only):
python3 analysis/llm_selfconsistency.py

# Extend to a new model (Ollama endpoint, resumable):
python3 analysis/extract_llm_selfconsistency.py \
    --datasets mmlu_pro,mmlu --n_samples 32 --model qwen3.5-9b \
    --ollama_url http://127.0.0.1:11434
```

## Notes on sampling

1. A 256k-context variant with verbose thinking gave empty responses
   (`done_reason=length`) and 300 s timeouts; the ctx4k variant with
   `think: false` + `num_predict=64` + a tightened prompt fixed this
   (~10x speedup, ~204 s to ~20 s per question).
2. `OLLAMA_NUM_PARALLEL=4` is applied via a systemd drop-in; the benchmark
   confirms 4x concurrent throughput even though the server process shows
   `-np 1`.
