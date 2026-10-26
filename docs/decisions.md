# Decisions log

Running log of decisions made during the project: any change to models, hyperparameters,
prompts, metrics or data handling is recorded here with the reason. Newest at the bottom.

| Date | Decision | Reason |
|---|---|---|
| 26 Oct | Data downloaded from ToxiCR commit `e71370aa175aa057d505229e4f25fcc687cf4a2b` (master on 26 Oct; the xlsx last changed in commit `4fb4f4d`, 19 Apr 2023). SHA-256 of `code-review-dataset-full.xlsx`: `51517e250312eadd50c157ef9efc269161938aa864bee97d60c587162b0f03f1` | Pinning the commit and checksum documents exactly which version of the data was used |
| 26 Oct | `id` = row position (0-based) in the original xlsx; the clean data keeps the first occurrence of each normalized text | Same id for a comment in both fold files; after conflicts are removed all copies of a text share the label, so the choice only affects which spelling is kept |
| 26 Oct | TF-IDF baseline: the inner 3-fold CV for choosing C uses `StratifiedKFold(3, shuffle=True, random_state=42)` instead of scikit-learn's default `cv=3` (which splits in file order) | File order may group comments by project, which would make unshuffled inner folds unrepresentative; the setting is otherwise unchanged |
| 26 Oct | TF-IDF baseline: C is chosen with `GridSearchCV(LogisticRegression(...), {"C": grid}, scoring="f1")` and the final model is fit explicitly, instead of `LogisticRegressionCV` | Same procedure. `LogisticRegressionCV` is mid API-transition in scikit-learn 1.8–1.9, and its warm-started optimizer stopped further from the converged solution (leaky F1 0.847 vs 0.861; clean unchanged at 0.763) |
| 26 Oct | TF-IDF baseline: optimizer tolerance `tol=1e-6` instead of the default `1e-4`. Final results: clean F1 0.761, leaky 0.860 (seen 0.955, unseen 0.751) | The default stops before convergence, so results depended on the optimizer's stopping point; `1e-6` and `1e-8` give identical predictions. Changes F1 by at most 0.002 and costs about one extra minute |
| 26 Oct | Fold files created and frozen: clean 12,722 rows (2,451 toxic), leaky 19,647 rows (3,757 toxic); 54.5% of leaky test rows have an exact copy in training | Matches the numbers verified in September |
