# Running the encoder fine-tuning on Kaggle

`train_encoder.ipynb` fine-tunes one encoder model with 5-fold cross-validation on a Kaggle GPU and
writes one predictions file plus one log. You run it once per model (see the run table below).
Everything you change is in the **configuration cell** at the top of the notebook.

What the notebook does in one run:
1. Picks the learning rate on fold 0: it trains one model per value in `LR_GRID` and keeps the value
   with the best F1 on a validation part (10% of fold 0's training folds). The test fold is not used.
2. Trains all 5 folds with that learning rate, keeps the best of up to 4 epochs (by validation F1),
   and predicts each held-out fold. Fold 0 reuses the model already trained during step 1.
3. Writes `preds/<RUN_NAME>_<setting>.csv` after every fold and `logs/<RUN_NAME>_<setting>.json`.
   Model weights are never saved to disk (the best epoch is kept in memory); only these two files are
   kept as output.

---

## A. One-time setup (about 10 minutes)

### A0. Verify your phone number

Kaggle only allows GPU and internet in notebooks for phone-verified accounts. Click your profile picture
(top right) → **Settings** → **Phone verification**, and follow the steps. If it is already verified, skip this.

### A1. Upload the fold files as a private dataset

1. Go to <https://www.kaggle.com> and sign in.
2. In the left menu, click **Datasets**, then **+ New Dataset** (top right).
3. Drag in the two files from the repository's `data/` folder:
   - `clean_folds.csv`
   - `leaky_folds.csv`

   Upload nothing else (no raw data, no other files).
4. Title: `toxicr-folds` (any name works; the notebook searches for the files by name).
5. Make sure visibility is **Private**.
6. Click **Create**. Wait until the dataset page shows both files.

Never edit or re-upload these files later: every run must use exactly these folds.

### A2. Import the notebook

1. In the left menu, click **Code**, then **+ New Notebook**. An editor opens.
2. In the editor's top menu: **File → Import Notebook**.
3. Choose **Upload** (or "Browse files") and select `kaggle/train_encoder.ipynb` from the repository.
   Confirm if Kaggle asks whether to replace the current notebook.
4. Rename the notebook (click the title at the top left), for example `train-encoder`.

### A3. Attach the dataset

1. In the right-hand panel, find **Input** and click **+ Add Input**.
2. Switch to **Datasets** and the **Your Work** (or "Your Datasets") filter.
3. Click the **+** next to `toxicr-folds`. It now appears under Input.

### A4. Settings (right-hand panel, "Session options" / "Settings")

- **Accelerator: GPU T4 x2** (or "GPU T4" if offered). The notebook uses only one of the two GPUs on purpose.
  Do not choose P100 or TPU.
- **Internet: On** (needed to download the models from HuggingFace).
- Leave everything else (environment, persistence) as it is.

These settings are saved with the notebook and used for every run.

---

## B. Smoke test (do this first, about 5-10 minutes per model)

The smoke test trains on 500 comments for 1 epoch, fold 0 only, with the first learning rate. It
checks that everything works before you spend GPU hours.

1. In the configuration cell, set:
   ```python
   MODEL_ID = "FacebookAI/roberta-base"
   RUN_NAME = "roberta_base"
   FOLDS_FILE = "clean"
   SMOKE_TEST = True
   ```
2. Click **Run All** (top toolbar) and watch the cells run. The first run downloads the model.
3. Check the output:
   - The first lines show `GPU: Tesla T4 (...)` and the library versions.
   - `Folds file: /kaggle/input/.../clean_folds.csv (12,722 rows, 2,451 toxic)`.
   - A line like `fold 0, lr 2e-05: best epoch 1 (validation F1 0.xx), ... min`.
     The F1 does not matter here: with only 500 training comments and one epoch it is often close to 0,
     or exactly 0 with the note `This is normal in the smoke test`. The smoke test only checks that
     everything runs.
   - The last cell prints `Predictions: /kaggle/working/preds/roberta_base_SMOKE_clean.csv (2,545 rows, folds [0])`
     and `SMOKE TEST finished`.
   - No red error box anywhere.
4. Write down how many minutes the training line took (for planning the real runs).
5. Repeat steps 1-4 with ModernBERT, the model most likely to cause problems:
   ```python
   MODEL_ID = "answerdotai/ModernBERT-base"
   RUN_NAME = "modernbert_base"
   ```
   Its output should also say `Architecture: modernbert | extra loading options: {'attn_implementation': 'sdpa', ...}`.
6. When both work, **stop the session** (power icon / "Stop Session" at the top) so it does not use
   GPU quota while idle.

Smoke-test files are named `..._SMOKE_...`. You do not need to download them; never put them in `preds/`.

---

## C. A real run ("commit": runs in the background, you can close the browser)

1. In the configuration cell, set the values for the run from the run table below, and
   **`SMOKE_TEST = False`**.
2. Click **Save Version** (top right).
3. Choose **Save & Run All (Commit)**. Version name: the run name, e.g. `roberta_base clean`.
4. Click **Save**. Kaggle starts a fresh copy of the notebook on a GPU and runs every cell.
   You can now close the browser and turn off your computer.
5. To check progress: open the notebook from **Code → Your Work**, click the version number next to
   the title (or the "Versions" list), and select the running version. Its log shows the printed lines.
   A commit can run for up to 12 hours.
6. Only one real run per version. For the next model, change the configuration cell and save a new
   version. Each version keeps its own outputs. Kaggle may limit how many GPU runs happen at the same
   time; if a new version waits in a queue, that is why.

### Download the outputs

1. When the version shows as complete (a green check), open it and go to the **Output** tab/section.
2. Download the two files (or "Download All" as a zip):
   - `preds/<RUN_NAME>_<setting>.csv` → put it in the repository's `preds/` folder.
   - `logs/<RUN_NAME>_<setting>.json` → put it in the repository's `results/kaggle_logs/` folder.
3. Then compute the metrics locally:
   ```
   uv run python src/evaluate.py --preds "preds/*.csv" --out results/
   ```
   `evaluate.py` checks the file against the fold file (all ids, folds and labels) before scoring it.

Command-line alternative (Kaggle CLI already set up):
`kaggle kernels output <your-username>/<notebook-slug> -p <download-folder>` downloads the latest
version's output.

---

## D. Run table (order: the most important runs first)

| Run | MODEL_ID | RUN_NAME | FOLDS_FILE | LR_GRID | BATCH_SIZE / GRAD_ACCUM |
|---|---|---|---|---|---|
| 1 | `FacebookAI/roberta-base` | `roberta_base` | clean | `[2e-5, 3e-5, 5e-5]` | 32 / 1 |
| 2 | `google-bert/bert-base-uncased` | `bert_base` | clean | `[2e-5, 3e-5, 5e-5]` | 32 / 1 |
| 3 | `google-bert/bert-base-uncased` | `bert_base` | **leaky** | `[<chosen_lr from run 2>]` (one value) | 32 / 1 |
| 4 | `microsoft/codebert-base` | `codebert_base` | clean | `[2e-5, 3e-5, 5e-5]` | 32 / 1 |
| 5 | `answerdotai/ModernBERT-base` | `modernbert_base` | clean | `[2e-5, 3e-5, 5e-5]` | 32 / 1 |
| 6 | `answerdotai/ModernBERT-large` | `modernbert_large` | clean | `[1e-5, 2e-5, 3e-5]` | 16 / 2 |
| 7 | `jhu-clsp/ettin-encoder-150m` | `ettin_150m` | clean | `[2e-5, 3e-5, 5e-5]` | 32 / 1 |
| 8 | `jhu-clsp/ettin-encoder-400m` | `ettin_400m` | clean | `[1e-5, 2e-5, 3e-5]` | 16 / 2 |
| 9 | `microsoft/deberta-v3-base` | `deberta_v3_base` | clean | `[1e-5, 2e-5, 3e-5]` | 32 / 1 |
| 10 (optional) | `microsoft/deberta-v3-large` | `deberta_v3_large` | clean | to be decided after run 9 | 16 / 2 |
| 11 (optional) | `chandar-lab/NeoBERT` | `neobert` | clean | `[2e-5, 3e-5, 5e-5]` | 32 / 1 |

Leave `MAX_LEN = 256`, `MAX_EPOCHS = 4`, `FP16 = True` and `FOLDS_TO_RUN = [0, 1, 2, 3, 4]` unless a
run fails (see below). For run 3, open `bert_base_clean.json` from run 2 and copy its `"chosen_lr"`
value into `LR_GRID`, e.g. `LR_GRID = [3e-5]`. For run 9 (DeBERTa-v3), do a smoke test first: it is known to
sometimes produce NaN losses with fp16, which the notebook detects (then set `FP16 = False`, about twice as slow).
Run 10 only if run 9 clearly beats the other encoders. For run 11, set `TRUST_REMOTE_CODE = True`; NeoBERT may
need extra packages, so try it only after the other runs.

Rough run times (estimates; check the first real run): one run trains 7 models (3 for the learning-rate
search on fold 0, then folds 1-4), so the whole run takes about 7 times the minutes of the first
`fold 0, lr ...` line in the log. Measured: BERT/RoBERTa/CodeBERT about 4-4.5 minutes per training run
(about 32 minutes per run), ModernBERT-base about 7 minutes (about 50 minutes per run). Large models should take
roughly 2.5-3.5 hours; all runs together about 10-13 GPU hours. The smoke-test time is not a good guide (it includes
the model download and uses only 500 comments). The weekly GPU quota (at least 30 hours) resets on Saturdays.

---

## E. If something goes wrong

| What you see | What to do |
|---|---|
| `No GPU found` | Settings: Accelerator → GPU T4. Then run again. |
| Run All stops after a cell without an error (the next cell never starts) | A quirk of the interactive editor. Click the next cell and choose **Run → Run after** (or run the remaining cells one by one); finished cells do not need to run again. Commits are not affected. |
| `clean_folds.csv not found` | The dataset is not attached: right panel → Input → + Add Input (step A3). |
| `transformers ... is too old` | In the second code cell, remove the `#` before the `!pip install` line, run that cell, then **Restart Session** (Run menu) and Run All again. |
| Errors while downloading the model (`Connection`, `HTTPError`, `401`) | Internet is off: Settings → Internet → On. Check the model id for typos. |
| `Training loss became NaN/inf` | Set `FP16 = False` and run again (slower, but stable). |
| `SentencePieceExtractor requires the protobuf library` or `tiktoken is required` (DeBERTa-v3) | Add a new code cell at the top with `!pip install -q protobuf sentencepiece`, run it, then **Restart Session** and Run All again. |
| `CUDA out of memory` | Halve `BATCH_SIZE` and double `GRAD_ACCUM` (e.g. 32/1 → 16/2), so the effective batch stays the same. |
| `validation F1 is 0 in every epoch` (real run, not smoke test) | The model learned nothing. Do not use the run; tell Claude Code and include the log. |
| A run stopped half-way (timeout, crash) | The predictions file in the version's Output already contains the finished folds, and the log contains `"chosen_lr"`. Download both. Start a new version with `LR_GRID = [<chosen_lr>]` and `FOLDS_TO_RUN` = only the missing folds; then ask Claude Code to combine the two predictions files. |

Any other error: copy the complete error message (the whole red box) and give it to Claude Code.
