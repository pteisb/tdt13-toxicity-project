"""Find candidate few-shot examples for prompt condition C, from the few-shot pool only.

For each of the 8 example slots, a keyword pattern selects matching short comments (3-25 words)
with the right label; comments with slurs or sexual terms are excluded. Five candidates per slot
are drawn at random (seed 42). For slots where the first draw had few usable candidates, a second
draw of further candidates is made (SECOND_DRAW). A person then chooses one per slot; nothing is
chosen here. The output documents every candidate the examples in fewshot_v1.txt were chosen from.

Reads:  data/fewshot_pool.csv   (comments NOT in the LLM evaluation subset)
Writes: prompts/fewshot_candidates.csv   slot, draw, id, is_toxic, n_words, message

Run:  uv run python src/find_fewshot_candidates.py
      uv run python src/find_fewshot_candidates.py --per-slot 10
"""

import argparse
import re

import pandas as pd

from config import FEWSHOT_POOL, PROMPTS_DIR, SEED

OUT_FILE = PROMPTS_DIR / "fewshot_candidates.csv"
MIN_WORDS, MAX_WORDS = 3, 25
# Extra candidates for slots whose first five were mostly unusable (direct insults: most matches
# quote a joke, insult the writer themselves, or criticize code rather than a person)
SECOND_DRAW = {"toxic: direct insult": 10}

PROFANITY = r"fuck\w*|shit\w*|crap\w*|damn\w*|hell|bullshit|ass|asshole|piss\w*|bloody"
# Slurs and sexual terms: such comments are never shown as candidates
EXCLUDE = re.compile(
    r"nigg|fag|retard|cunt|whore|slut|bitch|sex|porn|dick|cock|pussy|tits|boob|rape|gay|homo|"
    r"penis|vagina|nazi|spastic|tranny", re.IGNORECASE)

# slot name -> (label, pattern that must match, pattern that must NOT match)
SLOTS = {
    "toxic: positive profanity": (
        1, (rf"\b({PROFANITY})\b.*\b(thanks?|great|awesome|nice|cool|good|love|yay|works?|lol)\b"
           rf"|\b(thanks?|great|awesome|nice|cool|good|love|yay|lol)\b.*\b({PROFANITY})\b"), None),
    "toxic: harsh critique of someone's code": (
        1, (r"\b(this|that|the|your)\b.*\b(code|patch|change|design|implementation|approach|function|"
           r"commit|test|api)\b.*\b(stupid|crap|crappy|garbage|ugly|horrible|terrible|idiotic|nonsense|"
           r"insane|ridiculous|awful|silly|dumb|broken|mess)\b"
           r"|\b(stupid|crap|crappy|garbage|ugly|horrible|terrible|idiotic|nonsense|insane|ridiculous|"
           r"awful|silly|dumb)\b.*\b(code|patch|change|design|implementation|approach|function|commit)\b"),
        r"\byou(')?(re| are)\b"),
    "toxic: direct insult": (
        1, (r"\byou(')?(re| are)?\b.*\b(idiot|stupid|moron|dumb|incompetent|clueless|lazy|ignorant|"
           r"fool|jerk|pathetic|useless)\b|\b(idiot|moron|fool|jerk)\b"), None),
    "toxic: WTF-type acronym": (
        1, r"\b(wtf|stfu|ffs|omfg|fml|wth|lmfao)\b", None),
    "non-toxic: self-deprecation": (
        0, (r"\b(my bad|stupid me|silly me|dumb me|i'?m (an )?(idiot|stupid|dumb)|i was (stupid|dumb|"
           r"an idiot)|my (stupid|dumb|silly) (mistake|error)|(stupid|dumb|silly) mistake on my part|doh|"
           r"duh)\b"), None),
    "non-toxic: technical kill/dead/abort": (
        0, r"\b(kill|killed|kills|killing|dead|abort|aborts|aborted|die|dies|fatal|zombie)\b", None),
    "non-toxic: neutral review": (
        0, r"\b(please|nit|typo|rebase|add|remove|rename|update|move)\b",
        rf"\b({PROFANITY}|stupid|ugly|wrong|bad|why|kill\w*|dead|abort\w*|die)\b|\?"),
    "non-toxic: polite criticism": (
        0, (r"\b(i think|i'?m not sure|could you|would you|would be (better|cleaner|nicer)|"
           r"please consider|i'?d (suggest|prefer)|maybe)\b.*\b(wrong|incorrect|unnecessary|not needed|"
           r"confusing|redundant|doesn'?t|isn'?t|won'?t|misleading|unclear|overkill)\b"), None),
}


def find_candidates(pool, per_slot):
    pool = pool.assign(n_words=pool["message"].str.split().str.len())
    eligible = pool[pool["n_words"].between(MIN_WORDS, MAX_WORDS)
                    & ~pool["message"].str.contains(EXCLUDE)]
    rows = []
    for slot, (label, include, exclude) in SLOTS.items():
        matches = eligible[(eligible["is_toxic"] == label)
                           & eligible["key"].str.contains(include, flags=re.IGNORECASE)]
        if exclude:
            matches = matches[~matches["key"].str.contains(exclude, flags=re.IGNORECASE)]
        first = matches.sample(min(per_slot, len(matches)), random_state=SEED)
        draws = [first]
        if slot in SECOND_DRAW:
            rest = matches[~matches["id"].isin(first["id"])]
            draws.append(rest.sample(min(SECOND_DRAW[slot], len(rest)), random_state=SEED))
        print(f"{slot}: {len(matches)} matching comments")
        for draw, sample in enumerate(draws, start=1):
            rows += [{"slot": slot, "draw": draw, "id": r.id, "is_toxic": r.is_toxic,
                      "n_words": r.n_words, "message": r.message} for r in sample.itertuples()]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Find few-shot example candidates.")
    parser.add_argument("--per-slot", type=int, default=5)
    args = parser.parse_args()

    candidates = find_candidates(pd.read_csv(FEWSHOT_POOL), args.per_slot)
    PROMPTS_DIR.mkdir(exist_ok=True)
    candidates.to_csv(OUT_FILE, index=False)
    print(f"\nWrote {OUT_FILE} ({len(candidates)} candidates)")


if __name__ == "__main__":
    main()
