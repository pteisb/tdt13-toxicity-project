"""Setup check: the shared config matches the project conventions.

Run with: uv run pytest
"""

import config


def test_prediction_columns():
    assert config.PRED_COLUMNS == ["id", "fold", "label", "prob_toxic", "pred", "system", "setting"]


def test_labels_and_seed():
    assert config.TOXIC == 1
    assert config.NON_TOXIC == 0
    assert config.SEED == 42


def test_project_folders_exist():
    for folder in [config.DATA_DIR, config.PREDS_DIR, config.LLM_RAW_DIR, config.PROMPTS_DIR,
                   config.RESULTS_DIR, config.ANALYSIS_DIR]:
        assert folder.is_dir(), f"missing folder: {folder}"
