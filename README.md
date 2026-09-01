# Decomposing Wrong-Consensus Agreement in LLM Self-Consistency

Data, code, and evidence for the paper of the same name. Preprint: [arXiv:2608.18795](https://arxiv.org/abs/2608.18795).

## What the paper studies

When a language model is sampled repeatedly on the same question, the majority
answer and its share of the votes are read as a confidence signal. This is the
self-consistency heuristic: higher agreement, higher reliability.

The signal degrades on hard questions. Wrong answers can agree as strongly as
right ones, so the assumption breaks down exactly where confidence is most
needed. This repository accompanies a quantitative account of what that
wrong-consensus agreement contains.

## The decomposition

We normalize a wrong run's expected agreement with the consensus by the
reference scale

```
d = (1 - p) / (C - 1)
```

where `p` is single-sample accuracy and `C` the number of answer options. The
resulting agreement index is split into two parts:

- a **mechanical** component, the agreement a plurality vote would produce from
  the per-case answer preference alone; and
- a **preference-unexplained residual**, the remainder.

The mechanical reference is leak-free. Each case's preference and accuracy are
estimated from that case's other runs only, so the samples need not be
independent. The central quantity is the coverage

```
phi = kappa_rival / kappa_emp
```

the fraction of the observed wrong-consensus agreement that the mechanical
account explains.

## Main results

- On public GPT-4.1 per-run data, `phi` is 0.81–0.93 on multiple-choice
  GPQA-Diamond and 0.59–0.78 on open-ended AIME, where a residual of 1.54–2.80
  index units remains.
- Under one fixed protocol (4 runs, 32 votes, temperature 0.7), all ten cells
  across five open-weights checkpoints sit at `phi` near 1, and the pattern
  holds for a two-run design.
- Agreement is treated as graded evidence of correctness, not a certificate.
  The paper proposes no new voting method.

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | Manuscript (Elsevier `cas-sc` template) and three figures; build with `./compile.sh` |
| `analysis/` | One script per reported table and appendix |
| `analysis/kappa_tool/` | Reusable decomposition package: loader, decomposer, CLI, parity tests |
| `data/raw/` | De-identified per-run tables, 5,300 runs |
| `data/sampled/` | Our sampling on open-weights checkpoints |
| `results/` | Committed evidence as JSON, one file per reported number |

## Requirements

```
pip install -r requirements.txt
```

The dependencies are `numpy`, `pandas`, `pyarrow`, and `scipy`. A LaTeX engine
(`latexmk`, `pdflatex`, or `tectonic`) builds the paper.

## Reproduction

Every number in the manuscript is produced by a committed script from committed
data with a fixed seed (0); a run reproduces the committed `results/*.json` bit
for bit. Inputs:

- `data/raw/case_results_deid.parquet` — per-run rows (5,300)
- `data/raw/answer_distributions_deid.parquet` — per-run answer-count distributions
- `data/sampled/per_llm/llm_sc_qwen3.5-9b-*.jsonl` — Qwen appendix arm
- `data/sampled/tier3/tier3_*.jsonl` — ten-cell controlled replication

Main decomposition (Table 1; about four hours):

```bash
python3 analysis/kappa_rival_preference.py \
  --input-cases data/raw/case_results_deid.parquet \
  --input-dist  data/raw/answer_distributions_deid.parquet \
  --output results/kappa_rival_preference.json \
  --seed 0 --bootstrap 10000 --n-sim 100000 --min-wrong 1 \
  --c-gpqa 4 --aime-c-mode mean_distinct
```

Appendix B (shrinkage, lambda in {1, 0.5, 0}):

```bash
for LAM in 1.0 0.5 0.0; do
python3 analysis/kappa_rival_preference.py \
  --input-cases data/raw/case_results_deid.parquet \
  --input-dist  data/raw/answer_distributions_deid.parquet \
  --output results/kappa_rival_shrink_lam$LAM.json \
  --seed 0 --bootstrap 2000 --n-sim 20000 --min-wrong 1 --shrink $LAM \
  --c-gpqa 4 --aime-c-mode mean_distinct
done
```

Appendix C (C-invariance) and Appendix D (run-level dispersion):

```bash
for CVAL in 9 20; do
python3 analysis/kappa_rival_preference.py \
  --input-cases data/raw/case_results_deid.parquet \
  --input-dist  data/raw/answer_distributions_deid.parquet \
  --output results/kappa_rival_csens_C$CVAL.json \
  --seed 0 --bootstrap 100 --n-sim 2000 --min-wrong 1 \
  --c-gpqa 4 --aime-c-mode fixed --aime-c-fixed $CVAL
done
PYTHONPATH=analysis python3 analysis/kappa_rival_dispersion.py \
  --input-cases data/raw/case_results_deid.parquet \
  --input-dist  data/raw/answer_distributions_deid.parquet \
  --output results/kappa_rival_dispersion.json --seed 0 --n-sim 2000 --min-wrong 1
```

Remaining tables:

```bash
python3 analysis/reproduce_backfire.py --input data/raw/case_results_deid.parquet \
  --output results/backfire_repro.json --seed 0 --bootstrap 10000 --n-bins 5
python3 analysis/champion_fragility.py --input data/raw/case_results_deid.parquet \
  --output results/champion_fragility.json --seed 0 --bootstrap 10000
python3 analysis/jensen_gap.py --input data/raw/answer_distributions_deid.parquet \
  --input-cases data/raw/case_results_deid.parquet --output results/jensen_gap.json
python3 analysis/kappa_support_audit.py --input-cases data/raw/case_results_deid.parquet \
  --input-dist data/raw/answer_distributions_deid.parquet --output results/kappa_support_audit.json
python3 analysis/phi_split_test.py --input results/kappa_rival_preference.json \
  --output results/phi_split_test.json
python3 analysis/make_figures.py --rival results/kappa_rival_preference.json \
  --backfire results/backfire_repro.json --outdir paper
```

Tie-break sensitivity is a flag on the main script (`--tie-break random
--tie-seed S`, against the default argmin tie-break).

Ten-cell controlled replication (Table 3 / Fig. 3; self-contained, no GPU):

```bash
python3 analysis/tier3_kappa.py          # -> results/tier3_kappa.json
python3 analysis/tier3_kappa.py --check  # re-verify the ten cells, write nothing
python3 analysis/tier3_data_audit.py     # must report 10/10 OK before citing
python3 analysis/tier3_kappa_r2.py       # 2-run design -> results/tier3_kappa_r2.json
python3 analysis/fig_phi_vs_p.py         # -> paper/fig3_phi_vs_p.pdf
```

`kappa_emp` is deterministic and reproduces at any `n_sim`; `phi` and the
confidence intervals fall within the canonical Monte-Carlo error at lower
`n_sim`. A full regeneration takes about four hours; bit-for-bit CI reproduction
needs the same `n_sim = 1e5` and seed 0.

## Data

- **GPT-4.1 per-run data.** De-identified rows from Ding (2026), released under
  CC BY 4.0, committed in `data/raw/` (per-run results and per-run answer-count
  distributions).
- **Open-weights sampling.** Our Ollama runs under the fixed protocol, committed
  in `data/sampled/`.

All inputs are committed. No figure or table depends on a gated or external
download.

## Citation

```bibtex
@misc{zhang2026decomposing,
  title         = {Decomposing Wrong-Consensus Agreement in Large Language
                   Model Self-Consistency},
  author        = {Zhang, Lizhuo and Tang, Mengmeng and Long, Chenfeng and
                   Tang, Xiaoyong and Luo, Xiang},
  year          = {2026},
  eprint        = {2608.18795},
  archivePrefix = {arXiv}
}
```
