# Toxicity detection in code reviews

Course project for TDT13 at NTNU (autumn 2026). The project re-evaluates binary toxicity detection on
the ToxiCR code review dataset (Sarker et al., ACM TOSEM 2023) with these systems:

- a TF-IDF and logistic regression baseline
- two off-the-shelf Detoxify models, with the default and a calibrated threshold
- ten fine-tuned encoders, from BERT-base (2018) to ModernBERT, Ettin and NeoBERT (2024 and 2025)
- two prompted decoder LLMs (gpt-oss-120b and Kimi-K2.6) through NTNU's LLM API, with three prompts

All systems use the same 5-fold cross-validation splits on a deduplicated version of the data. The
project also measures how much the duplicate comments in the original data inflate the scores.

The research questions are:

1. How much do duplicates inflate scores under the original protocol?
2. How well do general-purpose toxicity detectors work here, and how much of their error comes from
   technical vocabulary?
3. Do modern encoders (2024 and 2025) beat classic ones, and does code in pretraining help?
4. How close do prompted decoder LLMs come to fine-tuned encoders, and how much does the dataset's
   rubric help them?

## Main results

F1 for the toxic class with 95% bootstrap intervals, from `results/RESULTS_SUMMARY.md`.

| System | All 12,722 clean comments | 2,000-comment LLM subset |
|---|---|---|
| TF-IDF + logistic regression | 0.761 [0.748, 0.774] | 0.777 [0.744, 0.806] |
| Detoxify original, calibrated threshold | 0.581 [0.565, 0.596] | 0.608 [0.569, 0.646] |
| DeBERTa-v3-base (best encoder on all clean comments) | 0.844 [0.834, 0.855] | 0.837 [0.807, 0.864] |
| gpt-oss-120b, rubric and examples (best LLM setup) | | 0.735 [0.700, 0.770] |
| Kimi-K2.6, rubric and examples | | 0.722 [0.682, 0.758] |

On the original, non-deduplicated data the baseline reaches 0.860 instead of 0.761, and BERT-base
0.893 instead of 0.834. More than half of the test comments there (54.5%) have a copy in the
training folds. The fine-tuned encoders all score between 0.818 and 0.844 on the clean data. Giving
the LLMs the dataset's annotation rubric instead of a general definition of toxicity raised their F1
by 0.319 (gpt-oss-120b) and 0.455 (Kimi-K2.6). All tables, figures and the full summary are in
[`results/`](results/README.md).

## Setup

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # create .venv and install the pinned dependencies from uv.lock
cp .env.example .env       # only needed for the LLM scripts: paste your NTNU API token into .env
uv run pytest              # 121 tests, under a minute
```

`.env` is git-ignored. The scripts read the token from it and never print it.

## Reproducing the results

### From the committed predictions

The predictions of every system are committed in `preds/`, so all tables and figures can be rebuilt
without training anything:

```bash
uv run python src/make_results.py
```

This takes about half a minute and reproduces the files in `results/tables/`, `results/figures/` and
`results/RESULTS_SUMMARY.md` exactly. To score single predictions files on the command line:

```bash
uv run python src/evaluate.py --preds preds/tfidf_lr_clean.csv preds/deberta_v3_base_clean.csv
uv run python src/evaluate.py --preds "preds/*_v1_clean.csv" --subset data/llm_eval_ids.csv
```

### Running the systems again

These are the steps in the order they were run. All scripts are in `src/`. Each one says in its
docstring what it reads and writes, and prints a short summary at the end.

| Step | Command | Where | Writes |
|---|---|---|---|
| 1. Data | `uv run python src/prepare_data.py` | local | `data/clean_folds.csv`, `data/leaky_folds.csv` |
| 2. Baseline | `uv run python src/baseline_tfidf.py` | local | `preds/tfidf_lr_{clean,leaky}.csv` |
| 3. Detoxify | `uv run python src/detoxify_eval.py` | local | `preds/detoxify_*_clean.csv` |
| 4. Encoders | `kaggle/train_encoder.ipynb`, once per model | Kaggle T4 GPU | `preds/<model>_clean.csv`, run logs |
| 5. LLM data | `uv run python src/make_llm_subset.py`, `find_fewshot_candidates.py`, `make_llm_pilot_ids.py` | local | `data/llm_eval_ids.csv`, `data/fewshot_pool.csv`, `data/llm_pilot_ids.csv` |
| 6. LLM checks | `uv run python src/llm_smoke_test.py`, then the pilot (below) | NTNU network | `results/llm_smoke_test.txt`, `results/llm_pilot/` |
| 7. LLM runs | `uv run python src/llm_classify.py ...` (below), then `llm_run_stats.py` | NTNU network | `llm_raw/*.jsonl`, `preds/*_v1_clean.csv` |
| 8. Error analysis | `uv run python src/error_analysis.py sample`, later `recode` | local, then by hand | `analysis/*.xlsx` |
| 9. Results | `uv run python src/make_results.py` | local | `results/` |

Some notes on the steps:

- `prepare_data.py` and `make_llm_subset.py` create the frozen data files and refuse to overwrite
  them. Every result depends on these exact fold assignments and subsets, so do not run them with
  `--force` in this repository. `make_llm_pilot_ids.py` and `find_fewshot_candidates.py` overwrite
  their output, and running them again reproduces the committed files. The data is described in
  [`docs/data.md`](docs/data.md).
- Step 4 is described in [`kaggle/README_kaggle.md`](kaggle/README_kaggle.md), including the settings
  of all 11 runs. GPU training is not bit-wise deterministic, so a new run gives slightly different
  numbers.
- In step 5, the eight few-shot examples in `prompts/fewshot_v1.txt` were chosen by hand from the
  candidates in `prompts/fewshot_candidates.csv`. See [`prompts/README.md`](prompts/README.md).
- In step 8, the categories are coded by hand in the spreadsheet. The committed workbooks hold these
  codes, and the script never overwrites an existing workbook.

The LLM scripts need the NTNU network or VPN and the token in `.env`. The pilot and one full
run look like this:

```bash
uv run python src/llm_classify.py --model openai/gpt-oss-120b --condition B --prompt-version v1 \
    --ids data/llm_pilot_ids.csv --pilot
uv run python src/llm_pilot_review.py --model gptoss120b

uv run python src/llm_classify.py --model openai/gpt-oss-120b --condition B --prompt-version v1 \
    --ids data/llm_eval_ids.csv
```

The full runs repeat the last command for conditions A, B and C and for both models
(`openai/gpt-oss-120b` and `moonshotai/Kimi-K2.6`), six runs in total. A run that stops can be
started again with the same command, and it skips the comments that already have an answer.

## Documentation

| File | Contents |
|---|---|
| [`docs/data.md`](docs/data.md) | Data source and version, deduplication, folds, LLM subset, every file in `data/` |
| [`docs/methods.md`](docs/methods.md) | Settings of every system and of the evaluation, problems found during the runs, software versions |
| [`results/README.md`](results/README.md) | What every table, figure and log in `results/` contains |
| [`prompts/README.md`](prompts/README.md) | The three prompts, how the messages are built, prompt versioning |
| [`kaggle/README_kaggle.md`](kaggle/README_kaggle.md) | Step-by-step guide to the encoder runs on Kaggle |

## Repository layout

```
src/            one script per step, plus config.py, evaluate.py and llm_client.py as shared helpers
tests/          pytest tests for the scripts in src/
data/           frozen fold files, LLM subset, few-shot pool, pilot ids (data/raw/ is git-ignored)
kaggle/         encoder fine-tuning notebook and its guide
prompts/        versioned prompt files and the few-shot candidates
preds/          one predictions file per system and setting
llm_raw/        every LLM response as JSONL, pilot runs in llm_raw/pilot/
results/        tables, figures, summary and run logs, built by src/make_results.py
analysis/       hand-coded error analysis workbooks
```

Every predictions file has the columns `id, fold, label, prob_toxic, pred, system, setting`. Labels
are 1 for toxic and 0 for non-toxic, `prob_toxic` is empty for the LLMs, and `setting` is `clean` or
`leaky`.

## Data and license

The ToxiCR dataset is by Sarker, Turzo, Dong and Bosu
([repository](https://github.com/WSU-SEAL/ToxiCR), [paper](https://doi.org/10.1145/3583562)) and is
licensed under GPL-3.0. The raw file is downloaded by `src/prepare_data.py` from a pinned commit and
is not committed, but the fold files in `data/` contain its comment texts and labels, and the LLM
reasoning traces in `llm_raw/` often quote them.

The code and data in this repository are licensed under the GNU General Public License v3.0 (see
[`LICENSE`](LICENSE)). The changes made to the ToxiCR data are described in
[`docs/data.md`](docs/data.md).
