"""Prompt construction for the General Tests (Tests 1-3).

Six prompt variants are used throughout the paper:

- ``A``/``B``/``C`` -- *narrative* framings (John / stroke patient / alien);
- ``D``/``E``/``F`` -- *implicit* framings (erase-only editorial instructions).

The same builders cover the single-demonstration default and the
multi-demonstration (2/5/10) Test 1 setting; pass a list of
``(sentence, edited_sentence)`` pairs to use more than one demonstration.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

DemoPair = Tuple[str, str]

IMPLICIT_PROMPT_VARIANTS: Tuple[str, ...] = ("D", "E", "F")
NARRATIVE_PROMPT_VARIANTS: Tuple[str, ...] = ("A", "B", "C")
ALL_PROMPT_VARIANTS: Tuple[str, ...] = IMPLICIT_PROMPT_VARIANTS + NARRATIVE_PROMPT_VARIANTS

_CLOSING = "Answer directly without explanation."

_IMPLICIT_INTROS = {
    "D": (
        "Someone shortens English sentences only by omitting words from the original "
        "sentence-without substituting new synonyms, adding new ideas, or reordering "
        "the whole clause.\n\n"
    ),
    "E": (
        "You follow an editorial checklist: the shorter line must be an **erase-only** "
        "version of the longer one-keep surviving words in the same order, add no new "
        "clauses, and do not swap in synonyms that effectively rewrite the predicate.\n\n"
    ),
    "F": (
        "Treat this as tightening prose under a brevity preference: shorten by **dropping** "
        "words from the source only. What remains should read as the same surface wording "
        "where it appears-no paraphrase-as-rewrite and no stitched-in new phrases.\n\n"
    ),
}


def _normalize_variant(v: str) -> str:
    s = (v or "A").strip().upper()
    if s not in ALL_PROMPT_VARIANTS:
        raise ValueError(f"prompt_variant must be one of {ALL_PROMPT_VARIANTS}; got {v!r}")
    return s


def _escape_for_single_quoted_english(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _example_block(demo_pairs: Sequence[DemoPair], s_test: str) -> str:
    parts: List[str] = []
    if len(demo_pairs) == 1:
        orig, edited = demo_pairs[0]
        parts.append(f'Example - full sentence: "{orig}"')
        parts.append(f'Shortened by omission in the same style: "{edited}"')
    else:
        for idx, (orig, edited) in enumerate(demo_pairs, start=1):
            parts.append(f'Example {idx} - full sentence: "{orig}"')
            parts.append(f'Shortened by omission in the same style: "{edited}"')
    body = "\n".join(parts)
    return f'{body}\n\nNow shorten this sentence in that same way (omit words only):\n"{s_test}"'


def build_implicit_prompt(demo_pairs: Sequence[DemoPair], s_test: str, *, prompt_variant: str = "A") -> str:
    v = _normalize_variant(prompt_variant)
    if v not in IMPLICIT_PROMPT_VARIANTS:
        raise ValueError("build_implicit_prompt supports only D/E/F.")
    return _IMPLICIT_INTROS[v] + _example_block(demo_pairs, s_test) + _CLOSING


def build_narrative_prompt(demo_pairs: Sequence[DemoPair], s_test: str, *, prompt_variant: str) -> str:
    v = _normalize_variant(prompt_variant)
    if v not in NARRATIVE_PROMPT_VARIANTS:
        raise ValueError(f"build_narrative_prompt expects A/B/C; got {prompt_variant!r}")
    escaped = [
        (_escape_for_single_quoted_english(d), _escape_for_single_quoted_english(e))
        for d, e in demo_pairs
    ]
    t = _escape_for_single_quoted_english(s_test)

    if len(escaped) == 1:
        # Single-demonstration phrasing (inline, matching the original prompts).
        d, e = escaped[0]
        if v == "A":
            body = (
                f"John developed a very special way of speaking. For '{d}', he would say '{e}'. "
                f"Please carefully examine John's speaking style, and guess what he would say for the sentence: '{t}'."
            )
        elif v == "B":
            body = (
                f"A man speaks in a really weird way after a stroke. For '{d}', he would say '{e}'. "
                f"What will he say for the sentence: '{t}'?  "
            )
        elif v == "C":
            body = (
                f"An alien is learning our language, but he tends to omit some words in his sentences. For '{d}', "
                f"he would say '{e}'. Please guess how the alien would express the following sentence: '{t}'."
            )
        return body + _CLOSING

    # Multi-demonstration phrasing (one line per pair).
    pair_block = "\n".join(f"For '{d}', he would say '{e}'." for d, e in escaped)
    if v == "A":
        body = (
            "John developed a very special way of speaking.\n"
            f"{pair_block}\n"
            "Please carefully examine John's speaking style, and guess what he would say "
            f"for the sentence: '{t}'."
        )
    elif v == "B":
        body = (
            "A man speaks in a really weird way after a stroke.\n"
            f"{pair_block}\n"
            f"What will he say for the sentence: '{t}'?  "
        )
    elif v == "C":
        body = (
            "An alien is learning our language, but he tends to omit some words in his sentences.\n"
            f"{pair_block}\n"
            f"Please guess how the alien would express the following sentence: '{t}'."
        )
    return body + _CLOSING


def build_prompt(demo_pairs: Sequence[DemoPair], s_test: str, *, prompt_variant: str) -> str:
    """Build the prompt for any variant from one or more demonstration pairs."""
    v = _normalize_variant(prompt_variant)
    if v in IMPLICIT_PROMPT_VARIANTS:
        return build_implicit_prompt(demo_pairs, s_test, prompt_variant=v)
    return build_narrative_prompt(demo_pairs, s_test, prompt_variant=v)


def resolve_demo_list(
    trials: List[Dict[str, Any]],
    flat: int,
    n_demos: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[int]]:
    """Demonstrations for slot ``flat`` in the multi-demo setting.

    Demonstration *k* is borrowed from trial ``(flat + k) mod n``, so each test
    item keeps its own demonstration first and the following trials supply the
    extra ones.  Returns ``(demos, test_item, demo_dataset_indices)``.
    """
    if n_demos < 1:
        raise ValueError("n_demos must be >= 1")
    n = len(trials)
    if n == 0:
        raise ValueError("trials must be non-empty")
    i = flat % n
    indices = [(i + k) % n for k in range(n_demos)]
    demos = [trials[j]["demo"] for j in indices]
    return demos, trials[i]["test"], indices
