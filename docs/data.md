# Data

All data comes from the ToxiCR code review dataset (Sarker et al., ACM TOSEM 2023,
[doi:10.1145/3583562](https://doi.org/10.1145/3583562)). This file describes the version used, how
`src/prepare_data.py` turns it into the two fold files, and what every file in `data/` contains.

## Source and version

| | |
|---|---|
| Repository | <https://github.com/WSU-SEAL/ToxiCR> |
| Commit | `e71370aa175aa057d505229e4f25fcc687cf4a2b` (master on 26 Oct 2026) |
| File | `models/code-review-dataset-full.xlsx` (last changed in commit `4fb4f4d`, 19 April 2023) |
| SHA-256 | `51517e250312eadd50c157ef9efc269161938aa864bee97d60c587162b0f03f1` |
| License | GPL-3.0 |

`prepare_data.py` downloads the file from this commit into `data/raw/` (git-ignored) and records the
commit and the checksum in `results/data_stats.csv`. The file has the columns `message` and
`is_toxic` (1 = toxic, 0 = non-toxic).

## From the raw file to the fold files

The raw file has 19,651 rows, 3,757 of them toxic. Four rows have an empty message and are dropped,
which leaves 19,647.

Each message gets a normalized key: lowercase, with all whitespace collapsed to single spaces
(`" ".join(text.lower().split())`). The 19,647 messages have 12,782 distinct keys. For 60 keys the
copies carry both labels (187 rows in total), and all rows with these keys are dropped. The clean data
keeps one row per remaining key, the first one in the file: 12,722 comments, 2,451 of them toxic
(19.3%).

The leaky data keeps all 19,647 non-empty rows without any deduplication, as the original evaluation
did. Both files are split with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on
`is_toxic`, after sorting by `id`.

| | Rows | Toxic | Rows per fold | Toxic per fold |
|---|---|---|---|---|
| Clean | 12,722 | 2,451 | 2,544 to 2,545 | 490 to 491 |
| Leaky | 19,647 | 3,757 | 3,929 to 3,930 | 751 to 752 |

In the leaky folds, 54.5% of the test comments (53.3% to 55.5% per fold) have a key that also
appears in the four training folds. These are the "seen" comments in table T4.

The fold files are frozen. `prepare_data.py` refuses to overwrite them unless it is run with
`--force`, and every system in this project uses exactly these fold assignments.

## Files in `data/`

| File | Rows | Columns | Made by |
|---|---|---|---|
| `clean_folds.csv` | 12,722 | `id, message, is_toxic, fold, key` | `prepare_data.py` |
| `leaky_folds.csv` | 19,647 | `id, message, is_toxic, fold, key` | `prepare_data.py` |
| `llm_eval_ids.csv` | 2,000 | `id, fold, is_toxic` | `make_llm_subset.py` |
| `fewshot_pool.csv` | 10,722 | same as `clean_folds.csv` | `make_llm_subset.py` |
| `llm_pilot_ids.csv` | 50 | `id, fold, is_toxic` | `make_llm_pilot_ids.py` |
| `detoxify_scores.csv` | 12,722 | `id, original, unbiased` | `detoxify_eval.py` (a cache) |

`id` is the row position in the original xlsx file (0-based). It is the same in both fold files,
so a comment can be followed from the raw file through every predictions file.

The LLM evaluation subset is a stratified random sample of 2,000 clean comments (seed 42), with 385
toxic comments. The LLMs were too slow to run on all 12,722 comments, so every other system is also
scored on these 2,000 ids, using its out-of-fold predictions. The other 10,722 clean comments form the
few-shot pool. The eight few-shot examples of prompt C and the 50 pilot comments (a simple random
sample, 12 of them toxic, without the eight examples) come from the pool, so nothing used to design the
prompts is in the evaluation subset.

## License

The fold files contain the ToxiCR comment texts and labels, which are licensed under GPL-3.0 by the
ToxiCR authors. `clean_folds.csv`, `leaky_folds.csv` and `fewshot_pool.csv` are modified
versions of the ToxiCR file, created on 26 October 2026 by `src/prepare_data.py` and
`src/make_llm_subset.py`. Empty rows were removed and the columns `id`, `key` and `fold` were added. In
`clean_folds.csv`, comments with conflicting labels were removed and duplicates dropped, and
`fewshot_pool.csv` is the part of `clean_folds.csv` outside the LLM evaluation subset. They are distributed under the same license, GPL-3.0 (see
[`LICENSE`](../LICENSE)).
