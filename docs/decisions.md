# Decisions log

Running log of decisions made during the project: any change to models, hyperparameters,
prompts, metrics or data handling is recorded here with the reason. Newest at the bottom.

| Date | Decision | Reason |
|---|---|---|
| 26 Oct | Data downloaded from ToxiCR commit `e71370aa175aa057d505229e4f25fcc687cf4a2b` (master on 26 Oct; the xlsx last changed in commit `4fb4f4d`, 19 Apr 2023). SHA-256 of `code-review-dataset-full.xlsx`: `51517e250312eadd50c157ef9efc269161938aa864bee97d60c587162b0f03f1` | Pinning the commit and checksum documents exactly which version of the data was used |
| 26 Oct | `id` = row position (0-based) in the original xlsx; the clean data keeps the first occurrence of each normalized text | Same id for a comment in both fold files; after conflicts are removed all copies of a text share the label, so the choice only affects which spelling is kept |
| 26 Oct | Fold files created and frozen: clean 12,722 rows (2,451 toxic), leaky 19,647 rows (3,757 toxic); 54.5% of leaky test rows have an exact copy in training | Matches the numbers verified in September |
