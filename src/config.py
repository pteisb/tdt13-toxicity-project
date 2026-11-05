"""Shared paths and constants for all scripts.

Reads: nothing. Writes: nothing.
Usage: scripts in src/ do `from config import CLEAN_FOLDS, SEED, ...`
(running `python src/<script>.py` puts src/ on the import path automatically).
"""

from pathlib import Path

# Project root = the folder that contains src/
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PREDS_DIR = ROOT / "preds"
LLM_RAW_DIR = ROOT / "llm_raw"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"
ANALYSIS_DIR = ROOT / "analysis"

# Frozen data files: never change once created, every model uses these exact folds
CLEAN_FOLDS = DATA_DIR / "clean_folds.csv"
LEAKY_FOLDS = DATA_DIR / "leaky_folds.csv"
LLM_EVAL_IDS = DATA_DIR / "llm_eval_ids.csv"
FEWSHOT_POOL = DATA_DIR / "fewshot_pool.csv"

SEED = 42
N_FOLDS = 5

# Labels: 1 = toxic, 0 = non-toxic. "Positive class" means toxic.
TOXIC = 1
NON_TOXIC = 0

# Required columns in every predictions file in preds/.
# prob_toxic may be empty for LLMs; setting is "clean" or "leaky".
PRED_COLUMNS = ["id", "fold", "label", "prob_toxic", "pred", "system", "setting"]
SETTINGS = ["clean", "leaky"]

# Error analysis: the hand-coding workbook, the recode workbook, and the categories an error can get.
# The first five are the categories of ToxiCR's own error analysis; the last two are new in this project.
ERROR_CODING = ANALYSIS_DIR / "error_analysis.xlsx"
ERROR_RECODE = ANALYSIS_DIR / "error_recode.xlsx"
ERROR_CATEGORIES = ["General error", "SE domain word", "Self-deprecation", "Bad acronym", "Confounding context",
                    "Definition mismatch", "Questionable label"]
