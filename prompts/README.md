# Prompts

System prompts for the three prompted-LLM conditions. Each condition changes one thing:

| File | Condition | Definition of toxicity | Examples |
|---|---|---|---|
| `A_v1.txt` | A, generic | General (Jigsaw/Perspective-style) definition | none |
| `B_v1.txt` | B, rubric | The dataset's annotation rules, paraphrased | none |
| `C_v1.txt` | C, rubric + examples | Template: B's text plus 8 labeled examples | `fewshot_v1.txt` |

A and B are worded identically apart from the definition (same task framing, same note on the
comment tags, same answer format), so A vs B isolates the effect of the definition.

## How the messages are built

- **System message:** the text of `A_v1.txt` or `B_v1.txt`. For condition C, the text of
  `C_v1.txt` with `{rubric}` replaced by the text of `B_v1.txt` and `{examples}` replaced by the
  text of `fewshot_v1.txt` (plain string replacement, so braces inside comments are safe).
- **User message:** only the comment, as `<comment>COMMENT TEXT</comment>`.
- **Expected answer:** exactly one word, `TOXIC` or `NON-TOXIC`.

`fewshot_v1.txt` holds the 8 examples (4 toxic, 4 non-toxic, taken from the few-shot pool, never
from the evaluation subset), in the same format the model sees for real comments:

```
<comment>example comment text</comment>
Answer: TOXIC

<comment>another example</comment>
Answer: NON-TOXIC
```

`fewshot_candidates.csv` lists every candidate the 8 examples were chosen from (random draws per
example category from the few-shot pool, made by `src/find_fewshot_candidates.py`; the `draw` column
marks a second draw where the first had too few usable candidates). The choice itself was made by
hand: one example per category whose label clearly follows the rubric.

## Versioning

The `v1` prompts are frozen (1 Nov 2026), after a pilot on 50 comments from the few-shot pool with
both models. A prompt file is never edited after a full LLM run has used it. Changes after that go into a new
version (`B_v2.txt`, ...), and the raw responses and predictions carry the version in their names.
