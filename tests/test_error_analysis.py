"""Tests for the error analysis: sampling and workbooks (src/error_analysis.py) and the tables T10/T11
(src/make_results.py), on small made-up predictions and codings.

Run with: uv run pytest
"""

import openpyxl
import pandas as pd
import pytest

import error_analysis as ea
import make_results
from config import ERROR_CATEGORIES


def fake_preds(n=400, seed=0):
    """Predictions of one made-up system: about a quarter of the comments wrong, toxic and non-toxic alike."""
    preds = pd.DataFrame({"id": range(n), "label": [i % 2 for i in range(n)]})
    wrong = preds["id"].sample(frac=0.25, random_state=seed)
    preds["pred"] = preds["label"].where(~preds["id"].isin(wrong), 1 - preds["label"])
    return preds


# real system keys (the tables look up display names), made-up predictions
ENC, DETOX, LLM = "deberta_v3_base", "detoxify_original_cal", "gptoss120b_C_v1"
SYSTEMS = {ENC: fake_preds(seed=1), DETOX: fake_preds(seed=2), LLM: fake_preds(seed=3)}
SIZES = {ENC: (20, 20), DETOX: (10, 10), LLM: (10, 10)}


def test_sample_sizes_per_system_and_error_type():
    _, key = ea.draw_sample(SYSTEMS, SIZES)
    counts = key.groupby(["system", "error_type"]).size()
    for system, (n_fp, n_fn) in SIZES.items():
        assert counts[(system, "FP")] == n_fp and counts[(system, "FN")] == n_fn


def test_one_row_per_comment_and_key_matches():
    sample, key = ea.draw_sample(SYSTEMS, SIZES)
    assert sample["id"].is_unique
    assert list(sample["row"]) == list(range(1, len(sample) + 1))
    assert set(key["id"]) == set(sample["id"])
    assert len(sample) < len(key)    # the fake systems share some errors, which get one row
    merged = key.merge(sample, on=["row", "id"], suffixes=("", "_sample"))
    assert (merged["error_type"] == merged["error_type_sample"]).all()


def test_sampled_rows_are_real_errors():
    sample, key = ea.draw_sample(SYSTEMS, SIZES)
    assert ((sample["error_type"] == "FP") == (sample["gold"] == 0)).all()
    assert (sample["gold"] != sample["pred"]).all()
    for _, line in key.iterrows():
        preds = SYSTEMS[line["system"]].set_index("id").loc[line["id"]]
        assert preds["label"] != preds["pred"]


def test_coding_sheet_is_blind():
    sample, _ = ea.draw_sample(SYSTEMS, SIZES)
    coding = ea.coding_table(sample, pd.Series({i: f"comment {i}" for i in range(400)}))
    assert list(coding.columns) == ea.CODING_COLUMNS
    assert "system" not in coding.columns
    assert coding["category"].isna().all() and coding["message"].notna().all()


def test_sample_is_reproducible():
    first, key_first = ea.draw_sample(SYSTEMS, SIZES, seed=42)
    second, key_second = ea.draw_sample(SYSTEMS, SIZES, seed=42)
    other, _ = ea.draw_sample(SYSTEMS, SIZES, seed=7)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(key_first, key_second)
    assert list(first["id"]) != list(other["id"])


def test_too_few_errors_is_an_error():
    with pytest.raises(ValueError, match="only"):
        ea.draw_sample(SYSTEMS, {ENC: (1000, 1)})


def test_category_guide_matches_the_categories():
    assert [name for name, _, _ in ea.CATEGORY_GUIDE] == ERROR_CATEGORIES


def test_workbook_round_trip(tmp_path):
    sample, key = ea.draw_sample(SYSTEMS, SIZES)
    messages = pd.Series({i: f"comment {i}" for i in range(400)})
    messages[sample["id"].iloc[0]] = "= this starts like a formula"
    path = tmp_path / "coding.xlsx"
    ea.write_workbook(path, {"coding": ea.coding_table(sample, messages), "key": key,
                             "categories": ea.category_guide()})

    sheets = pd.read_excel(path, sheet_name=None)
    assert list(sheets) == ["coding", "key", "categories"]
    assert sheets["coding"]["message"].iloc[0] == "= this starts like a formula"
    assert sheets["coding"]["id"].tolist() == sample["id"].tolist()
    workbook = openpyxl.load_workbook(path)
    assert workbook["key"].sheet_state == "hidden"
    assert workbook["coding"].freeze_panes == "A2"
    dropdowns = workbook["coding"].data_validations.dataValidation
    assert len(dropdowns) == 1 and "Definition mismatch" in dropdowns[0].formula1

    with pytest.raises(FileExistsError):
        ea.write_workbook(path, {"coding": sheets["coding"]})


def coded(sample, categories):
    """The coding sheet filled in with the given categories, in row order."""
    return ea.coding_table(sample, pd.Series(dtype=str)).assign(category=categories)


def test_recode_draws_coded_rows_only_when_complete():
    sample, _ = ea.draw_sample(SYSTEMS, SIZES)
    with pytest.raises(ValueError, match="not coded"):
        ea.draw_recode(coded(sample, [None] + ["General error"] * (len(sample) - 1)))
    recode = ea.draw_recode(coded(sample, ["General error"] * len(sample)), n=10)
    assert len(recode) == 10 and recode["row"].is_unique and recode["category"].isna().all()
    assert set(recode["row"]) <= set(sample["row"])


def write_coding(tmp_path, sample, key, categories):
    path = tmp_path / "error_analysis.xlsx"
    ea.write_workbook(path, {"coding": coded(sample, categories), "key": key})
    return path


def test_error_tables_skip_until_coding_is_complete(tmp_path):
    t10, t11, notes = make_results.error_analysis_tables(list(SIZES), tmp_path / "none.xlsx", tmp_path / "none.xlsx")
    assert t10 is None and t11 is None and "not sampled" in notes[0]
    sample, key = ea.draw_sample(SYSTEMS, SIZES)
    path = write_coding(tmp_path, sample, key, ["SE domain word"] * (len(sample) - 3) + [None] * 3)
    t10, t11, notes = make_results.error_analysis_tables(list(SIZES), path, tmp_path / "none.xlsx")
    assert t10 is None and "3 of" in notes[0]


def test_error_category_table_counts_and_shares(tmp_path):
    sample, key = ea.draw_sample(SYSTEMS, SIZES)
    categories = [ERROR_CATEGORIES[i % len(ERROR_CATEGORIES)] for i in range(len(sample))]
    path = write_coding(tmp_path, sample, key, categories)
    t10, t11, notes = make_results.error_analysis_tables(list(SIZES), path, tmp_path / "none.xlsx")
    assert t11 is None and "recode" in notes[0]
    assert len(t10) == len(SIZES) * 2 * len(ERROR_CATEGORIES)
    for (system, error_type), part in t10.groupby(["system", "error_type"]):
        assert part["count"].sum() == part["n"].iloc[0] == dict(zip(["FP", "FN"], SIZES[system]))[error_type]
        assert part["share"].sum() == pytest.approx(1)
    # each count is the number of that system's sampled errors coded with that category
    by_row = dict(zip(sample["row"], categories))
    enc_fp = key[(key["system"] == ENC) & (key["error_type"] == "FP")]["row"].map(by_row)
    row = t10[(t10["system"] == ENC) & (t10["error_type"] == "FP") & (t10["category"] == "General error")]
    assert row["count"].iloc[0] == (enc_fp == "General error").sum()
    toxicr = t10[(t10["category"] == "General error") & (t10["error_type"] == "FN")]["toxicr_bert_share"]
    assert (toxicr == 0.46).all()
    assert t10[t10["category"] == "Questionable label"]["toxicr_bert_share"].isna().all()


def test_agreement_table(tmp_path):
    sample, key = ea.draw_sample(SYSTEMS, SIZES)
    categories = ["General error" if i % 2 else "SE domain word" for i in range(len(sample))]
    path = write_coding(tmp_path, sample, key, categories)
    recode = ea.draw_recode(coded(sample, categories), n=10)
    first = recode["row"].map(dict(zip(sample["row"], categories)))
    recode["category"] = first.tolist()[:-2] + ["Bad acronym", "Bad acronym"]    # two of ten coded differently
    recode_path = tmp_path / "error_recode.xlsx"
    ea.write_workbook(recode_path, {"coding": recode})
    _, t11, notes = make_results.error_analysis_tables(list(SIZES), path, recode_path)
    assert notes == [] and t11["n"].iloc[0] == 10
    assert t11["agreement"].iloc[0] == pytest.approx(0.8)
    assert 0 < t11["cohen_kappa"].iloc[0] < 0.8


def test_unknown_category_is_rejected(tmp_path):
    sample, key = ea.draw_sample(SYSTEMS, SIZES)
    path = write_coding(tmp_path, sample, key, ["Typo category"] * len(sample))
    with pytest.raises(ValueError, match="unknown categories"):
        make_results.error_analysis_tables(list(SIZES), path, tmp_path / "none.xlsx")


def test_error_systems_on_real_results():
    """The systems drawn for the error analysis are the same "best" ones the results tables use."""
    assert ea.chosen_systems() == ["deberta_v3_base", "detoxify_original_cal", "gptoss120b_C_v1"]


def test_percent_text():
    assert make_results.percent(0.24) == r"24\%"
    assert make_results.percent(0.0026) == r"0.3\%"
    assert make_results.percent(0) == r"0\%"
    assert make_results.percent(None) == "--"
