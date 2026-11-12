# Results

Everything in `tables/`, `figures/` and `RESULTS_SUMMARY.md` is built from the predictions in `preds/`
by one command:

```bash
uv run python src/make_results.py
```

It takes about half a minute and is deterministic: running it on the committed predictions reproduces
the committed files exactly. All metrics are for the toxic class, and [a, b] is a 95% bootstrap
interval. How the metrics are computed is described in [`docs/methods.md`](../docs/methods.md).

The `.csv` files hold the full numbers. The `.tex` files hold only the `tabular` environment and are
meant to be wrapped in a `table` environment.

## Tables

| File | Contents |
|---|---|
| `T1_data` | Dataset statistics: raw file, clean and leaky data, rows and toxic comments per fold |
| `T2_main_clean` | The non-LLM systems on all 12,722 clean comments: precision, recall and F1 (mean and SD over folds), pooled F1 with interval, AP |
| `T2_per_fold.csv` | Precision, recall and F1 of the same systems in each fold |
| `T3_llm_subset` | All 21 systems on the 2,000-comment subset |
| `T4_leakage` | TF-IDF and BERT-base on the clean and leaky folds, with the leaky test comments split into seen and unseen |
| `T5_paired` | The 23 paired bootstrap comparisons (difference in pooled F1 with interval) |
| `T6_jargon` | False-positive rates on non-toxic comments with a technical word, against all non-toxic comments |
| `T7_encoder_runs` | Learning-rate search (validation F1 on fold 0 for every value), chosen learning rate, best epoch per fold, batch and run time per encoder |
| `T8_llm_runs` | LLM runs: answer counts, latency, output tokens and wall-clock time |
| `T9_confusion` | Confusion matrices on the subset for the best system of each family |
| `T10_error_categories` | Error categories per system from the hand coding, next to ToxiCR's own shares |
| `T11_error_agreement.csv` | Agreement between the first coding and the recoding of 30 errors |

## Figures

| File | Contents |
|---|---|
| `F1_subset.pdf/.png` | F1 with 95% interval for all systems on the 2,000-comment subset |
| `F2_leakage.pdf/.png` | Clean against leaky F1, and seen against unseen, for TF-IDF and BERT-base |

## Run logs and other files

| File | Contents | Written by |
|---|---|---|
| `RESULTS_SUMMARY.md` | The key numbers of every table in plain sentences | `make_results.py` |
| `data_stats.csv` | Row and toxic counts per dataset and fold, source commit, checksum, seen share | `prepare_data.py` |
| `tfidf_lr_log.csv` | Per fold: chosen C, inner cross-validation F1 for every C, vocabulary size, run time | `baseline_tfidf.py` |
| `detoxify_thresholds.csv` | Calibrated threshold per Detoxify model and fold | `detoxify_eval.py` |
| `kaggle_logs/*.json` | One log per encoder run: settings, library versions, validation scores per epoch, chosen learning rate, timing | `kaggle/train_encoder.ipynb` |
| `llm_run_stats.csv` | Per LLM system: parsed, unparsed and refused answers, retries, latency, tokens, start time, wall-clock | `llm_run_stats.py` |
| `llm_smoke_test.txt` | The API checks made before any LLM run (model ids, cache, temperature, reasoning effort) | `llm_smoke_test.py` |
| `llm_pilot/` | Pilot predictions and side-by-side review sheets (50 comments). Never used in the tables | `llm_classify.py --pilot`, `llm_pilot_review.py` |
