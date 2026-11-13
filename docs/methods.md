# Methods and settings

This file gives the full settings of every system and of the evaluation, and the problems that came up
during the runs. Numbers about runs come from the logs in `results/` (see
[`results/README.md`](../results/README.md)).

All randomness uses seed 42. Every system writes one predictions file to `preds/` with the columns
`id, fold, label, prob_toxic, pred, system, setting`, and no model script computes metrics itself.

## Evaluation

`src/evaluate.py` holds the metric functions and `src/make_results.py` builds all tables and figures
with them. Before scoring, every predictions file is checked against the fold file: the ids, folds and
labels must match exactly.

- All metrics are for the toxic class (label 1). Precision, recall and F1 are computed per fold
  (reported as mean and standard deviation over the five folds) and on all out-of-fold predictions
  pooled. Average precision (AP) is computed when `prob_toxic` is filled in, which excludes the LLMs.
- Confidence intervals for F1 use 2,000 bootstrap resamples of the comments (with replacement,
  seed 42) and the 2.5th and 97.5th percentiles.
- The paired bootstrap draws the same resampled comments for both systems in each of the 2,000
  resamples and records the difference in F1. A difference counts as reliable when its 95% interval
  excludes 0. An interval end within 0.002 of 0 is marked as borderline. The 23 comparisons in T5 are
  not adjusted for multiple comparisons.
- The best system of a family (for the error analysis, T9 and some comparisons) is the one with the
  highest pooled F1 on all 12,722 clean comments. For the LLMs, which only cover the subset, it is
  chosen on the 2,000-comment subset.
- Seen and unseen (leaky setting only): a test comment is seen if its normalized key also appears in
  the four training folds of its split.
- Technical-jargon slice (T6): non-toxic comments that contain one of these words as a whole word, in
  any capitalization. The words are the software engineering terms that the two ToxiCR papers name as
  misread by toxicity detectors (Sarker et al. 2020, Sarker et al. 2023 Section 5.6), each with its
  regular inflections: kill, die, dead, execute, garbage, dummy, junk, dirty, trash, daemon, naked, dump
  and bug. The 2020 paper's self-deprecation words (stupid, dumb, idiot, fool, ignorant) are left out,
  because they are not technical terms. A rate based on fewer than 100 comments is flagged as only
  indicative.

## TF-IDF and logistic regression (`src/baseline_tfidf.py`)

For each fold, the vectorizer and the classifier are fitted on the other four folds only.

- `TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)`
- `LogisticRegression(class_weight="balanced", max_iter=2000, tol=1e-6)` with the L2 penalty and the
  L-BFGS solver. The tolerance is lower than scikit-learn's default because the default stopped
  before convergence.
- C is chosen per fold from {0.25, 0.5, 1, 2, 4, 8} by `GridSearchCV` with stratified, shuffled 3-fold
  cross-validation on the training folds (scoring: toxic-class F1), then refitted on all training
  folds.
- A comment is predicted toxic when its probability is at least 0.5.

The chosen C, the inner cross-validation F1 for every C, the vocabulary size and the run time per
fold are in `results/tfidf_lr_log.csv`. On the clean folds the chosen C was 4 or 8 and the vocabulary
had about 22,000 terms. On the leaky folds it was about 55,000 terms.

## Detoxify (`src/detoxify_eval.py`)

Detoxify 0.5.4 with two of its models, used off the shelf without any training on this data:
`original` (BERT-base trained on the Jigsaw Wikipedia comments) and `unbiased` (RoBERTa-base trained
on the Jigsaw Civil Comments). The score is the `toxicity` output, and comments are truncated to 2,000
characters before scoring. The scores are cached in `data/detoxify_scores.csv` because scoring all
comments on a CPU takes 30 to 60 minutes.

Each model gives two systems. The plain one predicts toxic when the score is at least 0.5. The
calibrated one (`_cal`) uses, for each fold, the threshold that maximizes F1 on the other four folds.
The thresholds are in `results/detoxify_thresholds.csv`: 0.058 to 0.061 for `original` and 0.212 to
0.384 for `unbiased`.

## Fine-tuned encoders (`kaggle/train_encoder.ipynb`)

All encoders were fine-tuned on one Kaggle T4 GPU with the same recipe. How to run the notebook is
described in [`kaggle/README_kaggle.md`](../kaggle/README_kaggle.md).

- Sequence classification head with two labels, maximum length 256 tokens.
- AdamW, weight decay 0.01, linear decay with 10% warmup, fp16 mixed precision.
- Up to 4 epochs. A stratified 10% of the training folds is the validation split, and the epoch with
  the best validation F1 is kept.
- The learning rate is chosen once per model: one model per value in the grid is trained on fold 0,
  and the value with the best validation F1 is used for all five folds. The test fold is never used.
  Fold 0 reuses the model trained during the search.
- `prob_toxic` is the softmax probability of the toxic class, and a comment is predicted toxic when
  it is at least 0.5.

| Model | Hugging Face id | LR grid | Chosen LR | Batch x accumulation | Minutes per training run |
|---|---|---|---|---|---|
| BERT-base | `google-bert/bert-base-uncased` | 2e-5, 3e-5, 5e-5 | 2e-5 | 32 x 1 | 4.0 |
| RoBERTa-base | `FacebookAI/roberta-base` | 2e-5, 3e-5, 5e-5 | 5e-5 | 32 x 1 | 4.6 |
| CodeBERT-base | `microsoft/codebert-base` | 2e-5, 3e-5, 5e-5 | 5e-5 | 32 x 1 | 4.3 |
| DeBERTa-v3-base | `microsoft/deberta-v3-base` | 1e-5, 2e-5, 3e-5 | 3e-5 | 32 x 1 | 7.3 |
| DeBERTa-v3-large | `microsoft/deberta-v3-large` | 5e-6, 1e-5, 2e-5 | 2e-5 | 8 x 4 | 18.1 |
| ModernBERT-base | `answerdotai/ModernBERT-base` | 2e-5, 3e-5, 5e-5 | 2e-5 | 32 x 1 | 7.0 |
| ModernBERT-large | `answerdotai/ModernBERT-large` | 1e-5, 2e-5, 3e-5 | 2e-5 | 16 x 2 | 16.1 |
| Ettin-150m | `jhu-clsp/ettin-encoder-150m` | 2e-5, 3e-5, 5e-5 | 2e-5 | 32 x 1 | 7.1 |
| Ettin-400m | `jhu-clsp/ettin-encoder-400m` | 1e-5, 2e-5, 3e-5 | 3e-5 | 16 x 2 | 15.1 |
| NeoBERT | `chandar-lab/NeoBERT` | 2e-5, 3e-5, 5e-5 | 5e-5 | 32 x 1 | 10.1 |

Each clean run trains seven models (three for the search, then folds 1 to 4). BERT-base on the leaky
folds reuses the learning rate from its clean run and trains five models (6.5 minutes each).
DeBERTa-v3-large uses batch 8 with 4 accumulation steps because 16 with 2 ran out of memory. In 9 of
the 10 searches the chosen value is at the edge of its grid. On a validation split with about 196
toxic comments, the differences between learning rates are about as large as the run-to-run noise
described below, so I did not extend the grids. The validation F1 for every learning rate, the best
epoch per fold and the run times are in `results/tables/T7_encoder_runs.csv`, and the full logs are in
`results/kaggle_logs/`.

### Problems found during the runs

All of these were fixed in the notebook. Every kept run was either made with the fix or was not
affected by the problem, as stated for each one.

- Best-epoch restore. In Transformers 5 the Trainer reloads the best checkpoint without a strict name
  check. BERT checkpoints store their LayerNorm weights under the old names (`gamma`, `beta`), so the
  reload silently skipped all 50 LayerNorm tensors, and they kept their last-epoch values. The
  notebook now saves no checkpoints. A callback keeps the best epoch's weights in memory, restores them
  with a strict load, and checks that the validation F1 and loss of that epoch are reproduced. BERT-base
  was run again with the fix. RoBERTa-base is the only kept run made before the fix. It is not affected,
  because its weight names match (checked by saving and reloading the model).
- fp16 weights. DeBERTa-v3 is stored in fp16, and Transformers 5 loads a model in its stored precision,
  so mixed-precision training failed. The notebook converts all weights to fp32 after loading. All
  other models are stored in fp32, so the conversion changes nothing for them.
- Gradient accumulation. With accumulation steps above 1, Transformers 5 assumed that these models
  normalize the loss over all accumulated batches and skipped its own division, so the gradients were
  2 or 4 times too large. Adam's updates are invariant to a constant factor on the gradient apart from
  its small epsilon term, so the main effect was a stricter effective gradient clipping. The notebook
  now sets `trainer.model_accepts_loss_kwargs = False`, so the Trainer divides the loss itself. With
  this fix, a batch of 8 with 4 accumulation steps and a batch of 32 give identical gradients. The
  three affected runs (ModernBERT-large, Ettin-400m, DeBERTa-v3-large) were repeated.
- Attention on the T4. FlashAttention does not run on the T4, so ModernBERT and Ettin are loaded with
  `attn_implementation="sdpa"` (PyTorch's built-in attention).
- NeoBERT runs its authors' own model code, pinned to revision
  `5424c8efeea6491b151d62dee55a752165407430`. It needed two fixes under Transformers 5. Its
  feed-forward layer imports the `xformers` package, which is not installed on Kaggle, so the notebook
  provides a plain PyTorch SwiGLU layer with the same parameters and computation. Its rotary position
  table is not stored in the checkpoint and was left at zero by the library, so the notebook recomputes
  it with NeoBERT's own code after loading and checks that it is no longer zero. With both fixes, the
  pretrained model fills in masked words correctly.

### Run-to-run variation

GPU training is not bit-wise deterministic, so running the notebook again with the same settings and
seed gives slightly different numbers. The first BERT-base run (discarded because of the restore
problem, which did not affect the learning-rate search) gave validation F1 0.829 / 0.839 / 0.850 for
the three learning rates. The kept run gave 0.840 / 0.836 / 0.838. The local scripts and
`make_results.py` are deterministic.

## Decoder LLMs (`src/llm_classify.py`, `src/llm_client.py`)

The LLMs run on NTNU's LLM API, an OpenAI-compatible endpoint
(`https://llm.hpc.ntnu.no/v1/chat/completions`) served with vLLM 0.22.0 according to NTNU HPC's
documentation (checked 26 October 2026). The two models are `openai/gpt-oss-120b` and
`moonshotai/Kimi-K2.6`. The provider can update the served models, and all full runs were made on
1 November 2026.

- Request settings: temperature 0, seed 42, no `max_tokens` limit, and `reasoning_effort` not set. The
  smoke test (`results/llm_smoke_test.txt`) showed that not setting it gives the same output as
  "medium". Both models are reasoning models, and a small token limit can leave the final answer empty.
- The API's proxy caches identical requests. Every request bypasses this cache, so every answer is
  computed fresh and the latencies are real.
- The answer is read from the final message (`content`), not from the reasoning trace. NON-TOXIC is
  checked before TOXIC, because the first contains the second. An answer with neither word, with both,
  or with a refusal is asked once more. In the predictions file, answers that are still unparsed count
  as non-toxic.
- HTTP errors 429 and 5xx, timeouts and connection errors are retried up to 5 times with exponential
  backoff. Each request has a 300-second timeout.
- Every response is appended to `llm_raw/<system>.jsonl` straight away, with the comment id, the
  prompt's SHA-256, the parameters, the answer and the reasoning text. The comment itself is not
  stored, but the reasoning text often quotes it. Ids that already have a final answer are skipped, so
  a crashed run continues where it stopped.

The prompts are described in [`prompts/README.md`](../prompts/README.md). Before the full runs, I ran
both models with all three prompts on the 50 pilot comments. All 300 answers were clean single words,
and the `v1` prompts were frozen unchanged afterwards. The pilot outputs are in `llm_raw/pilot/` and
`results/llm_pilot/` and are never used in the result tables.

In the full runs (2,000 comments x 2 models x 3 prompts = 12,000 requests), every answer was parsed on
the first attempt, with no refusals and no failed requests (`results/llm_run_stats.csv`).

| System | Mean latency (s) | Mean completion tokens | HTTP retries | Wall-clock (h) |
|---|---|---|---|---|
| gpt-oss-120b A | 2.6 | 82 | 0 | 0.36 |
| gpt-oss-120b B | 2.3 | 98 | 1 | 0.32 |
| gpt-oss-120b C | 3.0 | 107 | 0 | 0.42 |
| Kimi-K2.6 A | 9.3 | 239 | 12 | 1.32 |
| Kimi-K2.6 B | 20.5 | 431 | 12 | 2.89 |
| Kimi-K2.6 C | 34.4 | 682 | 7 | 4.93 |

## Error analysis (`src/error_analysis.py`)

The errors come from the 2,000-comment subset, so the three systems are judged on the same comments:
50 false positives and 50 false negatives of the best encoder (DeBERTa-v3-base), and 25 of each from the
better Detoxify model (Detoxify original, calibrated) and from the best LLM setup (gpt-oss-120b with
prompt C). A comment that is an error of more than one of these systems appears once, which gives
175 rows for the 200 errors. The coding sheet does not show which system made the error. That
information is in a hidden sheet.

Each error got one of seven categories. The first five are the categories of ToxiCR's own error
analysis: general error, SE domain word, self-deprecation, bad acronym and confounding context. Two are
new in this project: definition mismatch and questionable label. At least a day after the first pass,
30 rows were coded a second time without looking at the first codes. The two passes agree on 77% of
them (Cohen's kappa 0.72, table T11). The coded sheets are `analysis/error_analysis.xlsx` and
`analysis/error_recode.xlsx`. They hold hand-made codes, so they are never regenerated.

## Software and hardware

| | Local scripts | Encoder training (Kaggle) |
|---|---|---|
| Hardware | CPU only | one NVIDIA Tesla T4 |
| Python | 3.12.10 | 3.13.15 |
| PyTorch | 2.14.1 (CPU) | 2.11.0 (CUDA 12.8) |
| Transformers | 5.20.2 | 5.16.1 |
| scikit-learn | 1.9.1 | 1.6.1 |
| pandas | 3.1.0 | 2.3.3 |
| NumPy | 2.5.4 | 2.1.3 |

The local versions are pinned in `uv.lock`. The Kaggle versions are recorded in every run log in
`results/kaggle_logs/` and were the same in all runs.
