# GPT-4.1 controlled-sampling protocol

Protocol used for the additional GPT-4.1 cells. The main GPT-4.1 rows come from
the Ding 2026 release under `data/raw/`; `analysis/extract_gpt41_api.py`
implements this protocol for any further GPT-4.1 cells.

## Setup

1. Python 3 (>= 3.8) plus `pip install requests huggingface_hub pyarrow`.
2. Set the API key in the environment only: `export OPENAI_API_KEY=sk-...`
   (the key is never written to any file or log).
3. Question files (pick one per benchmark):
   - GPQA-Diamond requires accepting the GPQA terms on your own Hugging Face
     account (`export HF_TOKEN=...`; the script downloads it), or drop
     `gpqa_diamond.csv` into `data/tier3_static/`.
   - AIME uses the public `di-zhang-fdu/AIME_1983_2024` release, placed the same
     way or auto-downloaded.

## Run (one command per cell, backgrounded)

```
nohup python3 extract_gpt41_api.py --bench gpqa --model gpt-4.1-mini > run_gpqa.log 2>&1 &
nohup python3 extract_gpt41_api.py --bench aime --model gpt-4.1-mini > run_aime.log 2>&1 &
# optionally: --model gpt-4.1 for a second round
```

## Protocol (fixed in the script; no edits needed)

4 runs x 32 votes x temperature 0.7, zero-shot. Resumable: the script skips
completed (question, run) pairs, so re-running the same command after an
interruption continues where it left off.

## Cost

Each cell is ~200 questions x 4 runs x 32 votes x ~600 tokens (~15M tokens);
at GPT-4.1-mini pricing (~$0.40/M input, $1.60/M output) that is roughly
$6-10 per cell, billed to the account that supplies the key.
