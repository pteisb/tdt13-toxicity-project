"""Error analysis: draw the errors to code by hand, and later the rows to code a second time.

The errors come from the 2,000-comment LLM subset, so all three systems are judged on the same comments:
50 false positives + 50 false negatives of the best fine-tuned encoder, 25 + 25 of the better Detoxify model
and 25 + 25 of the best LLM condition. These are the "best" systems that make_results.py picks. A comment
that is an error of several of these systems gets one row, and its code counts for each of them. The coding
sheet does not say which system made the error (blind coding); that is in the hidden sheet "key".

Reads:  preds/<system>_clean.csv, data/clean_folds.csv, data/llm_eval_ids.csv
        (recode: analysis/error_analysis.xlsx)
Writes: sample: analysis/error_analysis.xlsx  sheets "coding", "key" (hidden) and "categories"
        recode: analysis/error_recode.xlsx    30 coded rows again, with empty category and note
Neither command overwrites an existing file. The tables (T10, T11) are built by src/make_results.py.

Run:  uv run python src/error_analysis.py sample
      uv run python src/error_analysis.py recode    (after all rows are coded, at least a day later)
"""

import argparse
import sys

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from config import ERROR_CATEGORIES, ERROR_CODING, ERROR_RECODE, SEED
from evaluate import classification_scores, load_folds
from make_results import NON_LLM, SYSTEMS, comments, error_analysis_systems, read_coding

# (false positives, false negatives) per system, in the order of error_analysis_systems():
# best encoder, better Detoxify model, best LLM condition
SAMPLE_SIZES = [(50, 50), (25, 25), (25, 25)]
N_RECODE = 30

CODING_COLUMNS = ["row", "id", "message", "gold", "pred", "error_type", "category", "note"]
COLUMN_WIDTHS = {"row": 6, "id": 8, "message": 80, "gold": 6, "pred": 6, "error_type": 10, "category": 22,
                 "note": 50, "system": 24, "definition": 70, "example": 55}

# Definitions with one short example each (no real profanity). The examples for general error, SE domain word,
# self-deprecation, bad acronym and confounding context are taken or adapted from ToxiCR's paper.
CATEGORY_GUIDE = [
    ("General error",
     ("The system missed what the comment means in context: an ordinary phrase that often appears in toxic "
     "comments was read as toxic, or a rude remark without obviously rude words was missed."),
     "Done. I think they came from a messed up rebase. (non-toxic, predicted toxic)"),
    ("SE domain word",
     ("A technical word with a violent or negative everyday meaning (kill, die, dead, abort, fatal, garbage, junk) "
     "was read as toxic, or a harsh use of such a word was missed."),
     "You probably wanted `die` here; this error is not fatal. (non-toxic, predicted toxic)"),
    ("Self-deprecation",
     ("Demeaning words aimed at oneself, which the rubric counts as non-toxic (rule 8), or self-deprecating "
     "phrasing that hides an aggressive tone."),
     "Missing entry in the docs above... (stupid me) (non-toxic, predicted toxic)"),
    ("Bad acronym",
     ("A harmless acronym that also has a profane expansion, such as WebKit's WTF (Web Template Framework) "
     "namespace."),
     "Use WTF::Vector instead of std::vector here. (non-toxic, predicted toxic)"),
    ("Confounding context",
     "Mixed signals that would be hard even for a human annotator to label.",
     "This is a bit ugly, but it is what was asked for, so I added a null check."),
    ("Definition mismatch",
     ("The prediction fits everyday ideas of toxicity but not the rubric: for example profanity used in a "
     "positive way (the rubric counts all profanity as toxic), or blunt but civil criticism of code."),
     "[swear word], this is fast! Nice work. (toxic by the rubric, predicted non-toxic)"),
    ("Questionable label",
     ("You think the gold label is wrong under the rubric. Use it only when you are fairly sure, and write in "
     "the note which rule the comment does or does not fit."),
     "A polite technical question labeled toxic that fits none of the rubric's toxic rules."),
]
CODING_NOTES = [
    ("How to code",
     ("Read the comment and decide why the system got it wrong. Choose ONE category from the dropdown (the one "
     "that explains the error best) and write a one-line note. gold and pred: 1 = toxic, 0 = non-toxic. "
     "FP = predicted toxic but labeled non-toxic; FN = labeled toxic but predicted non-toxic. Do not open the "
     "hidden sheet 'key' while coding: it says which system made each error."), ""),
]


# ---------- Sampling ----------

def chosen_systems():
    """The systems make_results.py picks for the error analysis (only pooled F1 is needed to choose them)."""
    def pooled_f1(systems, comment_set):
        rows = []
        for system in systems:
            preds = comments(system.key, comment_set)
            rows.append({"system": system.key, "f1_pooled": classification_scores(preds["label"], preds["pred"])["f1"]})
        return pd.DataFrame(rows)

    return error_analysis_systems(pooled_f1(NON_LLM, "clean"), pooled_f1(SYSTEMS, "subset"))


def errors(preds, error_type):
    """The false positives ("FP") or false negatives ("FN") in a predictions table."""
    gold = 0 if error_type == "FP" else 1
    return preds[(preds["label"] == gold) & (preds["pred"] != gold)]


def draw_sample(preds_by_system, sizes, seed=SEED):
    """Draw the errors of each system. `sizes` maps a system to (number of FP, number of FN).
    Returns (sample, key): sample has one row per comment (row, id, gold, pred, error_type) in random order;
    key has one line per comment and system that made the error (row, id, error_type, system)."""
    rng = np.random.default_rng(seed)
    picks = []
    for system, (n_fp, n_fn) in sizes.items():
        for error_type, n in [("FP", n_fp), ("FN", n_fn)]:
            pool = errors(preds_by_system[system], error_type).sort_values("id")
            if len(pool) < n:
                raise ValueError(f"{system} has only {len(pool)} {error_type}s, {n} needed")
            picks.append(pool.sample(n, random_state=rng)[["id", "label", "pred"]]
                         .assign(error_type=error_type, system=system))
    picked = pd.concat(picks, ignore_index=True)
    sample = (picked.drop_duplicates("id")[["id", "label", "pred", "error_type"]]
              .sample(frac=1, random_state=rng).rename(columns={"label": "gold"}).reset_index(drop=True))
    sample.insert(0, "row", range(1, len(sample) + 1))
    key = sample[["row", "id"]].merge(picked[["id", "error_type", "system"]], on="id")
    return sample, key.sort_values(["row", "system"]).reset_index(drop=True)


def coding_table(sample, messages):
    """The coding sheet: the sample with each comment's text and empty category and note columns."""
    table = sample.assign(message=sample["id"].map(messages), category=None, note=None)
    return table[CODING_COLUMNS]


def draw_recode(coding, n=N_RECODE, seed=SEED):
    """n coded rows to code a second time, with category and note emptied."""
    if coding["category"].isna().any():
        raise ValueError(f"{int(coding['category'].isna().sum())} of {len(coding)} rows are not coded yet")
    return coding.sample(n, random_state=seed).assign(category=None, note=None)[CODING_COLUMNS]


def category_guide():
    return pd.DataFrame(CATEGORY_GUIDE + [("", "", "")] + CODING_NOTES, columns=["category", "definition", "example"])


# ---------- Excel ----------

def cell_value(value):
    """A value openpyxl can store: None for missing, plain Python numbers, text without control characters."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, str):
        return ILLEGAL_CHARACTERS_RE.sub("", value)
    return value


def write_workbook(path, sheets, hidden=("key",)):
    """Write each table to its own sheet: bold frozen header, wrapped text, a category dropdown on sheets that
    have both a category and a note column, and the sheets in `hidden` hidden. Never overwrites a file."""
    if path.exists():
        raise FileExistsError(f"{path} already exists (it may hold your codes); delete or rename it first")
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, table in sheets.items():
        sheet = workbook.create_sheet(name)
        sheet.append(list(table.columns))
        for values in table.itertuples(index=False):
            sheet.append([cell_value(v) for v in values])
        for cells in sheet.iter_rows():
            for cell in cells:
                if isinstance(cell.value, str):
                    cell.data_type = "s"    # text, also when it starts with "=" (Excel would read a formula)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        sheet.freeze_panes = "A2"
        for i, column in enumerate(table.columns, start=1):
            sheet.column_dimensions[get_column_letter(i)].width = COLUMN_WIDTHS.get(column, 12)
        if {"category", "note"} <= set(table.columns) and len(table):
            dropdown = DataValidation(type="list", formula1='"' + ",".join(ERROR_CATEGORIES) + '"', allow_blank=True,
                                      showErrorMessage=True, error="Choose one of the categories in the list")
            sheet.add_data_validation(dropdown)
            letter = get_column_letter(list(table.columns).index("category") + 1)
            dropdown.add(f"{letter}2:{letter}{len(table) + 1}")
        if name in hidden:
            sheet.sheet_state = "hidden"
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


# ---------- Commands ----------

def sample_command():
    if ERROR_CODING.exists():
        sys.exit(f"{ERROR_CODING} already exists (it may hold your codes); nothing written.")
    systems = chosen_systems()
    sizes = dict(zip(systems, SAMPLE_SIZES))
    sample, key = draw_sample({system: comments(system, "subset") for system in systems}, sizes)
    messages = load_folds("clean").set_index("id")["message"]
    write_workbook(ERROR_CODING, {"coding": coding_table(sample, messages), "key": key,
                                  "categories": category_guide()})

    print("Errors drawn from the 2,000-comment subset (seed 42):")
    for system, (n_fp, n_fn) in sizes.items():
        print(f"  {system}: {n_fp} FP + {n_fn} FN")
    shared = key.groupby("row").size()
    print(f"{len(key)} errors -> {len(sample)} rows to code ({int((shared > 1).sum())} comments are errors of "
          f"more than one system); {int((sample['error_type'] == 'FP').sum())} FP rows, "
          f"{int((sample['error_type'] == 'FN').sum())} FN rows")
    print(f"Wrote {ERROR_CODING} (sheets: coding, key [hidden], categories)")


def recode_command():
    if ERROR_RECODE.exists():
        sys.exit(f"{ERROR_RECODE} already exists (it may hold your codes); nothing written.")
    coding = read_coding(ERROR_CODING)
    if coding is None:
        sys.exit(f"{ERROR_CODING} not found: run the sample command first.")
    try:
        recode = draw_recode(coding)
    except ValueError as error:
        sys.exit(f"Finish the coding first: {error}.")
    write_workbook(ERROR_RECODE, {"coding": recode, "categories": category_guide()})
    print(f"Drew {len(recode)} of {len(coding)} coded rows (seed 42), category and note emptied.")
    print(f"Wrote {ERROR_RECODE}. Code it without looking at error_analysis.xlsx, then run src/make_results.py.")


def main():
    parser = argparse.ArgumentParser(description="Draw the error-analysis sample or the recode sample.")
    parser.add_argument("command", choices=["sample", "recode"],
                        help="sample: write the coding workbook; recode: draw rows to code a second time")
    args = parser.parse_args()
    sample_command() if args.command == "sample" else recode_command()


if __name__ == "__main__":
    main()
