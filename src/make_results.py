"""Build the final results package: all tables, figures and a plain-language summary.

All metrics come from the functions in src/evaluate.py, applied to the prediction files in preds/.
The systems are listed explicitly below, because preds/ holds files of two sizes: the non-LLM systems
cover all 12,722 clean comments, the LLMs only the 2,000-comment LLM subset. Pilot files (in
results/llm_pilot/) are never read.

Reads:  preds/<system>_<setting>.csv, data/clean_folds.csv, data/leaky_folds.csv, data/llm_eval_ids.csv,
        results/data_stats.csv, results/detoxify_thresholds.csv, results/kaggle_logs/*.json,
        results/llm_run_stats.csv
Writes: results/tables/T1_data.csv/.tex           dataset statistics, raw vs clean
        results/tables/T2_main_clean.csv/.tex     non-LLM systems on all 12,722 clean comments
        results/tables/T2_per_fold.csv            their precision, recall and F1 in each fold
        results/tables/T3_llm_subset.csv/.tex     all systems on the 2,000-comment subset
        results/tables/T4_leakage.csv/.tex        RQ1: clean vs leaky, seen vs unseen
        results/tables/T5_paired.csv/.tex         paired bootstrap differences (unadjusted)
        results/tables/T6_jargon.csv/.tex         false-positive rates on technical-jargon comments
        results/tables/T7_encoder_runs.csv/.tex   learning-rate search and training details (appendix)
        results/tables/T8_llm_runs.csv/.tex       LLM run statistics (appendix)
        results/tables/T9_confusion.csv/.tex      confusion matrices, best system per family (appendix)
        results/figures/F1_subset.pdf/.png        F1 with 95% CI, all systems on the subset
        results/figures/F2_leakage.pdf/.png       RQ1 bar chart
        results/RESULTS_SUMMARY.md                the key numbers in plain sentences
The .tex files hold only the tabular; wrap them in a table environment in the report.

Run:  uv run python src/make_results.py
"""

import json
import re
from collections import namedtuple
from functools import cache

import matplotlib

matplotlib.use("Agg")    # write files only, no window
import matplotlib.pyplot as plt
import pandas as pd

from config import LLM_EVAL_IDS, PREDS_DIR, RESULTS_DIR
from evaluate import (
    N_RESAMPLES,
    bootstrap_f1_ci,
    classification_scores,
    confusion_counts,
    false_positive_rate,
    filter_to_ids,
    load_folds,
    load_predictions,
    paired_bootstrap,
    per_fold_scores,
    seen_unseen_scores,
    summarize,
)

TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

System = namedtuple("System", "key name family year params")
BASELINE, OFF_THE_SHELF, CLASSIC, MODERN, LLM = (
    "Baseline", "Off-the-shelf", "Classic encoders", "Modern encoders", "Prompted LLMs")
FAMILIES = [BASELINE, OFF_THE_SHELF, CLASSIC, MODERN, LLM]
# Okabe-Ito colors, in an order checked for color-vision deficiency (adjacent families stay distinct).
FAMILY_COLORS = {BASELINE: "#D55E00", OFF_THE_SHELF: "#E69F00", CLASSIC: "#009E73", MODERN: "#0072B2",
                 LLM: "#CC79A7"}

# Parameter counts are the nominal sizes given by the model authors.
SYSTEMS = [
    System("tfidf_lr", "TF-IDF + LR", BASELINE, None, None),
    System("detoxify_original", "Detoxify original", OFF_THE_SHELF, 2020, "110M"),
    System("detoxify_original_cal", "Detoxify original, cal.", OFF_THE_SHELF, 2020, "110M"),
    System("detoxify_unbiased", "Detoxify unbiased", OFF_THE_SHELF, 2021, "125M"),
    System("detoxify_unbiased_cal", "Detoxify unbiased, cal.", OFF_THE_SHELF, 2021, "125M"),
    System("bert_base", "BERT-base", CLASSIC, 2018, "110M"),
    System("roberta_base", "RoBERTa-base", CLASSIC, 2019, "125M"),
    System("codebert_base", "CodeBERT-base", CLASSIC, 2020, "125M"),
    System("deberta_v3_base", "DeBERTa-v3-base", CLASSIC, 2021, "184M"),
    System("deberta_v3_large", "DeBERTa-v3-large", CLASSIC, 2021, "435M"),
    System("modernbert_base", "ModernBERT-base", MODERN, 2024, "149M"),
    System("modernbert_large", "ModernBERT-large", MODERN, 2024, "395M"),
    System("ettin_150m", "Ettin-150m", MODERN, 2025, "150M"),
    System("ettin_400m", "Ettin-400m", MODERN, 2025, "400M"),
    System("neobert", "NeoBERT", MODERN, 2025, "250M"),
] + [System(f"{model}_{condition}_v1", f"{name} {condition}", LLM, year, params)
     for model, name, year, params in [("gptoss120b", "gpt-oss-120b", 2025, "117B"),
                                       ("kimik26", "Kimi-K2.6", 2026, "~1T")]
     for condition in "ABC"]
BY_KEY = {system.key: system for system in SYSTEMS}
NON_LLM = [system for system in SYSTEMS if system.family != LLM]

# Section 5.8 of the project plan: non-toxic comments with these words test technical jargon.
JARGON_WORDS = ["kill", "killed", "kills", "die", "dies", "dead", "abort", "fatal", "crash", "destroy",
                "nuke", "bomb", "attack", "hack", "execute", "garbage", "junk"]
JARGON = re.compile(r"\b(?:" + "|".join(JARGON_WORDS) + r")\b", re.IGNORECASE)
FEW_JARGON = 100    # below this many comments a jargon rate is only indicative
BORDERLINE = 0.002  # an interval end this close to 0 is reported as borderline


def has_jargon(text):
    """True if the text contains one of the jargon words as a whole word (any capitalization)."""
    return bool(JARGON.search(str(text)))


# ---------- Loading ----------

@cache
def predictions(key, setting="clean"):
    return load_predictions(PREDS_DIR / f"{key}_{setting}.csv")


@cache
def subset_ids():
    return tuple(pd.read_csv(LLM_EVAL_IDS)["id"])


def comments(key, comment_set):
    """Predictions of one system on "clean" (all 12,722), "subset" (the 2,000) or "leaky"."""
    if comment_set == "subset":
        return filter_to_ids(predictions(key), subset_ids())
    return predictions(key, comment_set)


# ---------- Tables ----------

def results_table(systems, comment_set):
    """T2 / T3: summarize() for each system, with its name, family, year and size."""
    rows = [{**BY_KEY[s.key]._asdict(), **summarize(comments(s.key, comment_set))} for s in systems]
    table = pd.DataFrame(rows).drop(columns="system").rename(columns={"key": "system"})
    table["family_order"] = table["family"].map(FAMILIES.index)
    table = table.sort_values(["family_order", "f1_pooled"], ascending=[True, False])
    return table.drop(columns="family_order").reset_index(drop=True)


def best(table, keys):
    """The key with the highest pooled F1 among the given keys."""
    rows = table[table["system"].isin(keys)]
    return rows.loc[rows["f1_pooled"].idxmax(), "system"]


def keys_in(family):
    return [s.key for s in SYSTEMS if s.family == family]


def best_of_family(family, t2, t3):
    """Best system of a family: chosen on all 12,722 clean comments where possible (the larger sample, and
    not the comments the subset comparisons use), on the subset only for the LLMs."""
    return best(t3 if family == LLM else t2, keys_in(family))


def leakage_table():
    """T4: TF-IDF and BERT-base on clean data, and on leaky data split into seen and unseen comments."""
    rows = []
    for key in ["tfidf_lr", "bert_base"]:
        clean = predictions(key)
        scores = classification_scores(clean["label"], clean["pred"])
        low, high = bootstrap_f1_ci(clean["label"], clean["pred"])
        rows.append({"system": key, "setting": "clean", "group": "all", "n": len(clean),
                     "toxic": int(clean["label"].sum()), "precision": scores["precision"],
                     "recall": scores["recall"], "f1": scores["f1"], "f1_ci_low": low, "f1_ci_high": high})
        rows += seen_unseen_scores(predictions(key, "leaky"), load_folds("leaky"))
    table = pd.DataFrame(rows)
    table.insert(1, "name", table["system"].map(lambda key: BY_KEY[key].name))
    return table


def comparisons(t2, t3):
    """The paired comparisons of Section 5.8 of the project plan: (research question, A, B, comment set).
    Each row reports F1 of A minus F1 of B."""
    detoxify_original = best(t2, ["detoxify_original", "detoxify_original_cal"])
    detoxify_unbiased = best(t2, ["detoxify_unbiased", "detoxify_unbiased_cal"])
    detoxify = best(t2, keys_in(OFF_THE_SHELF))
    encoder = best(t2, keys_in(CLASSIC) + keys_in(MODERN))
    llm = best_of_family(LLM, t2, t3)
    pairs = [
        ("RQ1", "bert_base", "tfidf_lr", "clean"),
        ("RQ1", "bert_base", "tfidf_lr", "leaky"),
        ("RQ2", "bert_base", detoxify_original, "clean"),
        ("RQ2", "roberta_base", detoxify_unbiased, "clean"),
        ("RQ2", "tfidf_lr", detoxify, "clean"),
        ("RQ3", encoder, "tfidf_lr", "clean"),
        ("RQ3", "roberta_base", "codebert_base", "clean"),
        ("RQ3", "deberta_v3_base", "neobert", "clean"),
        ("RQ3", "deberta_v3_base", "modernbert_base", "clean"),
        ("RQ3", "deberta_v3_base", "ettin_150m", "clean"),
        ("RQ3", "modernbert_large", "modernbert_base", "clean"),
        ("RQ3", "ettin_400m", "ettin_150m", "clean"),
        ("RQ3", "deberta_v3_large", "deberta_v3_base", "clean"),
        ("RQ3", best_of_family(CLASSIC, t2, t3), best_of_family(MODERN, t2, t3), "clean"),
    ]
    for model in ["gptoss120b", "kimik26"]:
        pairs += [("RQ4", f"{model}_B_v1", f"{model}_A_v1", "subset"),
                  ("RQ4", f"{model}_C_v1", f"{model}_B_v1", "subset")]
    pairs += [("RQ4", f"gptoss120b_{c}_v1", f"kimik26_{c}_v1", "subset") for c in "ABC"]
    pairs += [("RQ4", encoder, llm, "subset"), ("RQ4", "tfidf_lr", llm, "subset")]
    return pairs


def paired_table(t2, t3):
    """T5: F1 differences with 95% paired bootstrap intervals."""
    rows = []
    for question, a, b, comment_set in comparisons(t2, t3):
        preds_a, preds_b = comments(a, comment_set), comments(b, comment_set)
        result = paired_bootstrap(preds_a, preds_b)
        rows.append({"rq": question, "a": a, "b": b, "name_a": BY_KEY[a].name, "name_b": BY_KEY[b].name,
                     "comment_set": comment_set, "n": result["n"],
                     "f1_a": classification_scores(preds_a["label"], preds_a["pred"])["f1"],
                     "f1_b": classification_scores(preds_b["label"], preds_b["pred"])["f1"],
                     "f1_diff": result["f1_diff"], "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                     "excludes_zero": bool(result["ci_low"] > 0 or result["ci_high"] < 0),
                     "borderline": bool(min(abs(result["ci_low"]), abs(result["ci_high"])) < BORDERLINE)})
    return pd.DataFrame(rows)


def jargon_table():
    """T6: false-positive rate on non-toxic comments with jargon words vs on all non-toxic comments."""
    folds = load_folds("clean")
    jargon_ids = set(folds.loc[folds["message"].map(has_jargon), "id"])
    rows = []
    for comment_set, systems in [("clean", NON_LLM), ("subset", SYSTEMS)]:
        for system in systems:
            preds = comments(system.key, comment_set)
            non_toxic = preds[preds["label"] == 0]
            jargon = non_toxic[non_toxic["id"].isin(jargon_ids)]
            rows.append({"system": system.key, "name": system.name, "family": system.family,
                         "comment_set": comment_set, "non_toxic": len(non_toxic), "jargon_non_toxic": len(jargon),
                         "fpr_all": false_positive_rate(non_toxic["label"], non_toxic["pred"]),
                         "fpr_jargon": false_positive_rate(jargon["label"], jargon["pred"])})
    return pd.DataFrame(rows)


def lr_text(lr):
    """2e-05 -> 2e-5"""
    return f"{lr:.0e}".replace("e-0", "e-")


def encoder_runs_table():
    """T7: one row per Kaggle run, from its JSON log."""
    rows = []
    for path in sorted((RESULTS_DIR / "kaggle_logs").glob("*.json")):
        log = json.loads(path.read_text(encoding="utf-8"))
        runs = list(log["lr_search"].values()) + [f for k, f in log["folds"].items()
                                                  if not (k == "0" and log["lr_search"])]
        config = log["config"]
        rows.append({
            "system": log["run"], "name": BY_KEY[log["run"]].name, "setting": log["folds_file"],
            "lr_grid": ", ".join(lr_text(lr) for lr in config["lr_grid"]),
            "val_f1_per_lr": " / ".join(f"{r['best_val_f1']:.3f}" for r in log["lr_search"].values()) or "--",
            "chosen_lr": lr_text(log["chosen_lr"]),
            "chosen_lr_at_grid_edge": len(config["lr_grid"]) > 1 and log["chosen_lr"] in (
                min(config["lr_grid"]), max(config["lr_grid"])),
            "best_epochs": ", ".join(str(log["folds"][str(f)]["best_epoch"]) for f in range(5)),
            "batch": f"{config['batch_size']} x {config['grad_accum']}",
            "fp16": config["fp16"], "training_runs": len(runs),
            "minutes_per_run": sum(r["train_seconds"] for r in runs) / len(runs) / 60,
            "transformers": log["versions"]["transformers"],
        })
    table = pd.DataFrame(rows)
    order = {s.key: i for i, s in enumerate(SYSTEMS)}
    return table.sort_values(["setting", "system"], key=lambda col: col.map(order) if col.name == "system" else col)


def llm_runs_table():
    """T8: the LLM run statistics with display names."""
    table = pd.read_csv(RESULTS_DIR / "llm_run_stats.csv")
    table.insert(1, "name", table["system"].map(lambda key: BY_KEY[key].name))
    return table


def confusion_table(t2, t3):
    """T9: confusion counts on the subset for the best system of each family."""
    rows = []
    for family in FAMILIES:
        key = best_of_family(family, t2, t3)
        preds = comments(key, "subset")
        rows.append({"family": family, "system": key, "name": BY_KEY[key].name,
                     **confusion_counts(preds["label"], preds["pred"])})
    return pd.DataFrame(rows)


# ---------- LaTeX ----------

def tex(text):
    return str(text).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%") \
        .replace("_", r"\_").replace("#", r"\#").replace("~", r"$\sim$")


def num(value, digits=3):
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def signed(value):
    return f"{value:+.3f}".replace("-", "$-$")


def tabular(columns, rows, align, breaks=(), note=None):
    """A booktabs tabular. `breaks` are row positions before which a \\midrule goes."""
    lines = [f"% {note}"] if note else []
    lines += [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(columns) + r" \\", r"\midrule"]
    for i, row in enumerate(rows):
        if i in breaks:
            lines.append(r"\midrule")
        lines.append(" & ".join(row) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def family_breaks(table):
    return [i for i in range(1, len(table)) if table["family"].iloc[i] != table["family"].iloc[i - 1]]


def results_tex(table):
    rows = [[tex(r["name"]), "--" if pd.isna(r["year"]) else str(int(r["year"])), "--" if pd.isna(r["params"]) else tex(r["params"]),
             f"{num(r['precision_mean'])} $\\pm$ {num(r['precision_sd'])}",
             f"{num(r['recall_mean'])} $\\pm$ {num(r['recall_sd'])}",
             f"{num(r['f1_mean'])} $\\pm$ {num(r['f1_sd'])}",
             f"{num(r['f1_pooled'])} [{num(r['f1_ci_low'])}, {num(r['f1_ci_high'])}]", num(r["ap"])]
            for _, r in table.iterrows()]
    return tabular(["System", "Year", "Size", "Precision", "Recall", "F1", r"Pooled F1 [95\% CI]", "AP"], rows,
                   "lrrccccc", family_breaks(table),
                   note=f"n = {table['n'].iloc[0]:,} comments; P/R/F1 as mean +- sd over 5 folds; CI: {N_RESAMPLES:,} "
                        "bootstrap resamples; AP needs probabilities (-- for LLMs)")


def data_tex(stats):
    def value(dataset, stat):
        row = stats[(stats["dataset"] == dataset) & (stats["fold"] == "all") & (stats["stat"] == stat)]
        return float(row["value"].iloc[0])
    def row(label, n, toxic):
        return [label, f"{n:,.0f}", f"{toxic:,.0f}", f"{toxic / n:.1%}"]

    subset = pd.read_csv(LLM_EVAL_IDS)
    rows = [row("Raw file", value("raw", "rows"), value("raw", "toxic")),
            row("Without empty messages (leaky setting)", value("leaky", "rows"), value("leaky", "toxic")),
            row("Deduplicated, conflicts removed (clean setting)", value("clean", "rows"), value("clean", "toxic")),
            row("LLM evaluation subset (from clean)", len(subset), subset["is_toxic"].sum())]
    return tabular(["Data", "Comments", "Toxic", "Share"], [[tex(c) for c in r] for r in rows], "lrrr",
                   note=f"Conflicting texts removed: {value('raw', 'conflicting_keys'):.0f} texts "
                        f"({value('raw', 'conflicting_rows'):.0f} rows); leaky test comments with a copy in "
                        f"training: {value('leaky', 'seen_share'):.1%}")


def leakage_tex(table):
    labels = {("clean", "all"): "Clean", ("leaky", "all"): "Leaky, all", ("leaky", "seen"): "Leaky, seen",
              ("leaky", "unseen"): "Leaky, unseen"}
    rows = [[tex(r["name"]), labels[(r["setting"], r["group"])], f"{r['n']:,}", num(r["precision"]),
             num(r["recall"]), f"{num(r['f1'])} [{num(r['f1_ci_low'])}, {num(r['f1_ci_high'])}]"]
            for _, r in table.iterrows()]
    return tabular(["System", "Comments", "$n$", "Precision", "Recall", r"F1 [95\% CI]"], rows, "llrccc", [4])


def paired_tex(table):
    sets = {"clean": "12,722 clean", "leaky": "19,647 leaky", "subset": "2,000 subset"}
    rows = [[r["rq"], f"{tex(r['name_a'])} $-$ {tex(r['name_b'])}", sets[r["comment_set"]],
             f"{num(r['f1_a'])} / {num(r['f1_b'])}", signed(r["f1_diff"]),
             f"[{signed(r['ci_low'])}, {signed(r['ci_high'])}]",
             ("yes" if r["excludes_zero"] else "no") + ("*" if r["borderline"] else "")]
            for _, r in table.iterrows()]
    breaks = [i for i in range(1, len(table)) if table["rq"].iloc[i] != table["rq"].iloc[i - 1]]
    return tabular(["RQ", "A $-$ B", "Comments", "F1 A / B", "Difference", r"95\% CI", "Excl.\\ 0"], rows,
                   "lllcrcc", breaks,
                   note=f"Paired bootstrap, {N_RESAMPLES:,} resamples, seed 42; intervals NOT adjusted for multiple "
                        f"comparisons; * = borderline, an interval end within {BORDERLINE} of 0")


def jargon_tex(table):
    clean = table[table["comment_set"] == "clean"].set_index("system")
    subset = table[table["comment_set"] == "subset"].reset_index(drop=True)
    rows = []
    for _, r in subset.iterrows():
        full = [num(clean.loc[r["system"], "fpr_all"]), num(clean.loc[r["system"], "fpr_jargon"])]             if r["system"] in clean.index else ["--", "--"]
        rows.append([tex(r["name"])] + full + [num(r["fpr_all"]), num(r["fpr_jargon"])])
    n = {s: (part["non_toxic"].iloc[0], part["jargon_non_toxic"].iloc[0])
         for s, part in table.groupby("comment_set")}
    columns = ["System", f"All, clean ({n['clean'][0]:,})", f"Jargon, clean ({n['clean'][1]:,})",
               f"All, subset ({n['subset'][0]:,})", f"Jargon, subset ({n['subset'][1]:,})"]
    return tabular(columns, rows, "lcccc", family_breaks(subset),
                   note="False-positive rate on non-toxic comments (number of comments in brackets); "
                        "LLMs only ran on the subset")


def encoder_runs_tex(table):
    rows = [[tex(r["name"]) + (" (leaky)" if r["setting"] == "leaky" else ""), r["lr_grid"], r["val_f1_per_lr"],
             r["chosen_lr"], r["best_epochs"], r["batch"].replace(" x ", r" $\times$ "), num(r["minutes_per_run"], 1)]
            for _, r in table.iterrows()]
    return tabular(["Model", "LR grid", "Val.\\ F1 per LR (fold 0)", "Chosen", "Best epoch per fold", "Batch",
                    "Min/run"], rows, "lllllcr")


def llm_runs_tex(table):
    rows = [[tex(r["name"]), f"{r['parsed']:,}", str(r["unparsed"]), str(r["refused"]), num(r["latency_mean_s"], 1),
             num(r["latency_median_s"], 1), f"{r['completion_tokens_mean']:.0f}", num(r["wall_clock_h"], 1)]
            for _, r in table.iterrows()]
    return tabular(["System", "Parsed", "Unparsed", "Refused", "Mean s", "Median s", "Tokens", "Hours"], rows,
                   "lrrrrrrr", note="Tokens: mean completion tokens per request, reasoning included; 4 parallel requests")


def confusion_tex(table):
    rows = [[tex(r["family"]), tex(r["name"]), str(r["tp"]), str(r["fp"]), str(r["fn"]), str(r["tn"])]
            for _, r in table.iterrows()]
    return tabular(["Family", "Best system", "TP", "FP", "FN", "TN"], rows, "llrrrr",
                   note="2,000-comment subset; toxic is the positive class; best system chosen on all clean "
                        "comments (LLMs: on the subset)")


def save(table, stem, latex):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(TABLES_DIR / f"{stem}.csv", index=False)
    (TABLES_DIR / f"{stem}.tex").write_text(latex, encoding="utf-8")


# ---------- Figures ----------

def style_axes(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#999999")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors="#333333", length=0)
    ax.set_axisbelow(True)


def save_figure(fig, stem):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # no creation date in the PDF, so rebuilding gives identical files
    for suffix, extra in [("pdf", {"metadata": {"CreationDate": None}}), ("png", {"dpi": 200})]:
        fig.savefig(FIGURES_DIR / f"{stem}.{suffix}", bbox_inches="tight", **extra)
    plt.close(fig)


def f1_figure(t3):
    """F1 with 95% CI on the subset, one row per system, grouped and colored by family."""
    fig, ax = plt.subplots(figsize=(6.3, 6.6))
    y, ticks, labels = 0, [], []
    for family in FAMILIES:
        part = t3[t3["family"] == family]
        for _, row in part.iterrows():
            color = FAMILY_COLORS[family]
            ax.plot([row["f1_ci_low"], row["f1_ci_high"]], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(row["f1_pooled"], y, "o", markersize=7, color=color, markeredgecolor="white",
                    markeredgewidth=1.2, label=family if row.name == part.index[0] else None)
            ticks.append(y)
            labels.append(row["name"])
            y -= 1
        y -= 0.6    # gap between families
    ax.set_yticks(ticks, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 for the toxic class, with 95% bootstrap CI (2,000 comments)")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=8)    # the top left is empty: no system scores below 0.4 there
    save_figure(fig, "F1_subset")


def leakage_figure(t4):
    """RQ1: F1 on clean data, on leaky data, and on leaky seen vs unseen comments."""
    groups = [("clean", "all", "Clean"), ("leaky", "all", "Leaky, all"), ("leaky", "seen", "Leaky, seen"),
              ("leaky", "unseen", "Leaky, unseen")]
    systems = [("tfidf_lr", FAMILY_COLORS[BASELINE]), ("bert_base", FAMILY_COLORS[CLASSIC])]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    width = 0.38
    for offset, (key, color) in zip([-width / 2 - 0.01, width / 2 + 0.01], systems):
        rows = [t4[(t4["system"] == key) & (t4["setting"] == s) & (t4["group"] == g)].iloc[0] for s, g, _ in groups]
        x = [i + offset for i in range(len(groups))]
        f1 = [r["f1"] for r in rows]
        errors = [[r["f1"] - r["f1_ci_low"] for r in rows], [r["f1_ci_high"] - r["f1"] for r in rows]]
        ax.bar(x, f1, width, color=color, label=BY_KEY[key].name)
        ax.errorbar(x, f1, yerr=errors, fmt="none", ecolor="#333333", elinewidth=1, capsize=0)
    ax.set_xticks(range(len(groups)), [label for _, _, label in groups])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (toxic class)")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncols=2)
    save_figure(fig, "F2_leakage")


# ---------- Summary ----------

def ci(row, low="f1_ci_low", high="f1_ci_high", value="f1_pooled"):
    return f"{row[value]:.3f} [{row[low]:.3f}, {row[high]:.3f}]"


def summary_markdown(t2, t3, t4, t5, t6, t7, t8):
    """The key numbers in plain sentences. Describes, does not interpret."""
    lines = ["# Results summary", "",
             "Generated by `src/make_results.py`; all numbers come from the tables in `results/tables/`. "
             "F1 is for the toxic class; [a, b] is a 95% bootstrap interval. Paired comparisons are not "
             "adjusted for multiple comparisons.", ""]

    lines += ["## All 12,722 clean comments (T2, non-LLM systems)", ""]
    for _, row in t2.iterrows():
        lines.append(f"- {row['name']}: F1 {ci(row)}, precision {row['precision_pooled']:.3f}, "
                     f"recall {row['recall_pooled']:.3f}, AP {num(row['ap'])}.")
    top = t2.loc[t2["f1_pooled"].idxmax()]
    lines += ["", f"Highest pooled F1: {top['name']} ({top['f1_pooled']:.3f}).", ""]

    lines += ["## 2,000-comment LLM subset (T3, all systems)", ""]
    for _, row in t3.iterrows():
        lines.append(f"- {row['name']}: F1 {ci(row)}, precision {row['precision_pooled']:.3f}, "
                     f"recall {row['recall_pooled']:.3f}.")
    lines.append("")

    lines += ["## RQ1: duplication (T4)", ""]
    for key in ["tfidf_lr", "bert_base"]:
        part = t4[t4["system"] == key].set_index(["setting", "group"])
        lines.append(f"- {BY_KEY[key].name}: clean {ci(part.loc[('clean', 'all')], value='f1')}; leaky "
                     f"{ci(part.loc[('leaky', 'all')], value='f1')}; leaky seen "
                     f"{ci(part.loc[('leaky', 'seen')], value='f1')} (n = {part.loc[('leaky', 'seen'), 'n']:,}); "
                     f"leaky unseen {ci(part.loc[('leaky', 'unseen')], value='f1')} "
                     f"(n = {part.loc[('leaky', 'unseen'), 'n']:,}).")
    lines.append("")

    lines += ["## Paired comparisons (T5)", ""]
    for _, row in t5.iterrows():
        verdict = "interval excludes 0" if row["excludes_zero"] else "interval includes 0"
        if row["borderline"]:
            verdict += f" (borderline: an interval end is within {BORDERLINE} of 0)"
        lines.append(f"- {row['rq']}, {row['name_a']} minus {row['name_b']} ({row['n']:,} comments): "
                     f"{row['f1_diff']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}], {verdict}.")
    lines.append("")

    lines += ["## Technical-jargon slice (T6)", ""]
    for comment_set, label in [("clean", "all clean comments"), ("subset", "the subset")]:
        part = t6[t6["comment_set"] == comment_set]
        n = part["jargon_non_toxic"].iloc[0]
        note = f" (fewer than {FEW_JARGON}: indicative only)" if n < FEW_JARGON else ""
        lines.append(f"- On {label}: {n} of {part['non_toxic'].iloc[0]:,} non-toxic comments contain a jargon "
                     f"word{note}.")
    for _, row in t6[t6["comment_set"] == "subset"].iterrows():
        lines.append(f"  - {row['name']}: false-positive rate {num(row['fpr_jargon'])} on jargon comments vs "
                     f"{num(row['fpr_all'])} on all non-toxic comments (subset).")
    lines.append("")

    lines += ["## LLM runs (T8)", ""]
    for _, row in t8.iterrows():
        lines.append(f"- {row['name']}: {row['parsed']:,} parsed, {row['unparsed']} unparsed, {row['refused']} "
                     f"refused; mean latency {row['latency_mean_s']:.1f} s; {row['wall_clock_h']:.1f} h.")
    lines.append("")

    lines += ["## Unusual entries", "", "Systems with recall below 0.3 or precision below 0.6 on the subset:", ""]
    for _, row in t3[(t3["recall_pooled"] < 0.3) | (t3["precision_pooled"] < 0.6)].iterrows():
        lines.append(f"- {row['name']}: precision {row['precision_pooled']:.3f}, recall {row['recall_pooled']:.3f}.")
    edge = t7[t7["chosen_lr_at_grid_edge"]]
    searched = t7[t7["val_f1_per_lr"] != "--"]
    lines += ["", f"- In {len(edge)} of {len(searched)} learning-rate searches the chosen value is the lowest or "
                  f"highest in its grid: {', '.join(edge['name'])}."]
    thresholds = pd.read_csv(RESULTS_DIR / "detoxify_thresholds.csv")
    for model, part in thresholds.groupby("model"):
        lines.append(f"- Calibrated Detoxify {model}: thresholds chosen on the training folds range from "
                     f"{part['threshold'].min():.4f} to {part['threshold'].max():.4f} (the default is 0.5).")
    lines.append("- The LLMs give no probabilities, so their average precision is not defined.")
    return "\n".join(lines) + "\n"


def main():
    t2 = results_table(NON_LLM, "clean")
    t3 = results_table(SYSTEMS, "subset")
    t4 = leakage_table()
    t5 = paired_table(t2, t3)
    t6 = jargon_table()
    t7 = encoder_runs_table()
    t8 = llm_runs_table()
    t9 = confusion_table(t2, t3)
    stats = pd.read_csv(RESULTS_DIR / "data_stats.csv")

    save(stats, "T1_data", data_tex(stats))
    save(t2, "T2_main_clean", results_tex(t2))
    per_fold = pd.concat([per_fold_scores(comments(s.key, "clean")).rename_axis("fold").reset_index()
                          .assign(system=s.key, name=s.name) for s in NON_LLM])
    per_fold[["system", "name", "fold", "precision", "recall", "f1"]].round(3).to_csv(
        TABLES_DIR / "T2_per_fold.csv", index=False)
    save(t3, "T3_llm_subset", results_tex(t3))
    save(t4, "T4_leakage", leakage_tex(t4))
    save(t5, "T5_paired", paired_tex(t5))
    save(t6, "T6_jargon", jargon_tex(t6))
    save(t7, "T7_encoder_runs", encoder_runs_tex(t7))
    save(t8, "T8_llm_runs", llm_runs_tex(t8))
    save(t9, "T9_confusion", confusion_tex(t9))
    f1_figure(t3)
    leakage_figure(t4)
    summary = summary_markdown(t2, t3, t4, t5, t6, t7, t8)
    (RESULTS_DIR / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")

    print(t2[["name", "f1_pooled", "f1_ci_low", "f1_ci_high", "ap"]].round(3).to_string(index=False))
    print(f"\nSubset: {len(t3)} systems | paired comparisons: {len(t5)} "
          f"({int(t5['excludes_zero'].sum())} exclude 0) | jargon comments: "
          f"{t6[t6['comment_set'] == 'clean']['jargon_non_toxic'].iloc[0]} (clean), "
          f"{t6[t6['comment_set'] == 'subset']['jargon_non_toxic'].iloc[0]} (subset)")
    print(f"\nWrote {TABLES_DIR} (T1-T9, .csv + .tex)\n      {FIGURES_DIR} (F1_subset, F2_leakage, .pdf + .png)"
          f"\n      {RESULTS_DIR / 'RESULTS_SUMMARY.md'}")


if __name__ == "__main__":
    main()
