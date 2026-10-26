# Toxic or Just Technical? Toxicity Detection in Code Reviews

Course project for TDT13 (NTNU, autumn 2026).

This project re-evaluates binary toxicity detection in code review comments using the ToxiCR
dataset (Sarker et al., ACM TOSEM 2023). It compares a TF-IDF + logistic regression baseline,
two off-the-shelf Detoxify models, fine-tuned encoder transformers (classic and modern), and
prompted large language models, all on the same deduplicated 5-fold cross-validation splits.
It also measures how much duplicate comments inflate scores under the original evaluation
protocol.

The ToxiCR data is GPL-3.0 licensed and is downloaded by the preparation script, not included
in this repository.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # create .venv and install dependencies
cp .env.example .env       # then paste your NTNU API token into .env
uv run pytest              # check that the setup works
```

## How to reproduce

*To be filled in as the project progresses.*

## Repository layout

```
data/raw/            downloaded ToxiCR xlsx (not committed)
data/                clean_folds.csv, leaky_folds.csv, LLM evaluation ids, few-shot pool
preds/               one predictions CSV per system x setting
llm_raw/             raw LLM API responses (JSONL)
prompts/             versioned prompt files
results/             metrics tables, bootstrap results, figures
analysis/            error analysis spreadsheets
kaggle/              notebooks for GPU fine-tuning on Kaggle
src/                 Python scripts, one per step, plus shared helpers
tests/               pytest tests
docs/decisions.md    log of design decisions and why they were made
report/              figures and tables exported for the report
```
