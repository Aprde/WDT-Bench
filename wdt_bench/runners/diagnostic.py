"""Runner for the diagnostic tests (Tests 4-6).

One function drives all six diagnostic conditions:

==================  ====  ==========================================
condition           test  description
==================  ====  ==========================================
parallel_english     4    English half of the parallel sentence pairs
parallel_chinese     4    Chinese half of the parallel sentence pairs
nonsense_english     5    English semantically-anomalous sentences
nonsense_chinese     5    Chinese semantically-anomalous sentences
ambiguity_pp         6    PP-attachment ambiguity (English)
ambiguity_adjunct    6    Adjunct/relative-clause ambiguity (English)
==================  ====  ==========================================

Each run performs ``n_trials`` trials; within a trial, ``n_per_trial``
distinct test sentences are sampled without replacement and each is paired
with ``n_shot`` randomly drawn demonstrations.  Results are written to
``results/raw/diagnostic_tests/{condition}__{model}.json`` and runs resume
automatically: incomplete trials are topped up with sentences that have not
yet been answered in that trial.
"""
from __future__ import annotations

import logging
import random

from .. import paths
from ..io_utils import atomic_write_results, load_json, load_results
from ..llm import ApiError, call_llm
from ..prompts.diagnostic import N_PROMPT_VARIANTS, build_prompt

logger = logging.getLogger(__name__)

DEFAULT_N_TRIALS = 30
DEFAULT_N_PER_TRIAL = 24
CHECKPOINT_EVERY_N_NEW_CALLS = 5


def _language_of(condition: str) -> str:
    if condition.endswith("_chinese"):
        return "chinese"
    return "english"


def _load_demo_pairs(condition: str) -> list[tuple[str, str]]:
    """Return ``(sentence, deleted_form)`` pairs for the condition."""
    rows = load_json(paths.diagnostic_demonstrations(condition))
    language = _language_of(condition)
    pairs: list[tuple[str, str]] = []
    for row in rows:
        if "sentence" in row:  # ambiguity conditions
            pairs.append((row["sentence"], row["label"]))
        elif language == "chinese":
            pairs.append((row["chinese"], row["chinese_label"]))
        else:
            pairs.append((row["english"], row["english_label"]))
    return pairs


def _load_test_sentences(condition: str) -> list[str]:
    rows = load_json(paths.diagnostic_test_sentences(condition))
    language = _language_of(condition)
    sentences: list[str] = []
    for row in rows:
        if "sentence" in row:  # ambiguity conditions
            sentences.append(row["sentence"])
        else:
            sentences.append(row[language])
    return sentences


def run_diagnostic_test(
    condition: str,
    model: str,
    provider: str = "openai",
    *,
    prompt_id: int = 0,
    n_shot: int = 1,
    n_trials: int = DEFAULT_N_TRIALS,
    n_per_trial: int = DEFAULT_N_PER_TRIAL,
    temperature: float = 0.0,
    reasoning_effort: str | None = "low",
    seed: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Run one diagnostic condition for one model (resumable).

    Returns a small summary dict (``{"new_calls", "n_results", "path"}``).
    """
    if condition not in paths.DIAGNOSTIC_CONDITIONS:
        raise ValueError(
            f"Unknown condition {condition!r}. "
            f"Choose one of: {', '.join(paths.DIAGNOSTIC_CONDITIONS)}"
        )
    if not 0 <= prompt_id < N_PROMPT_VARIANTS:
        raise ValueError(f"prompt_id must be in [0, {N_PROMPT_VARIANTS - 1}]")

    language = _language_of(condition)
    demo_pairs = _load_demo_pairs(condition)
    test_sentences = _load_test_sentences(condition)
    if n_per_trial > len(test_sentences):
        raise ValueError(
            f"n_per_trial={n_per_trial} exceeds the {len(test_sentences)} "
            f"available test sentences for {condition!r}"
        )

    rng = random.Random(seed)
    out_path = paths.diagnostic_raw(condition, model)

    # ---- Resume: load existing results and index them by trial -------------
    results: list[dict] = []
    if out_path.exists():
        results, _old_meta = load_results(out_path)
        logger.info("Resuming %s: %d existing rows", out_path.name, len(results))

    done_by_trial: dict[int, set[str]] = {}
    for row in results:
        done_by_trial.setdefault(int(row["trial"]), set()).add(row["sentence"])

    meta = {
        "test": paths.TEST_OF_CONDITION[condition],
        "condition": condition,
        "language": language,
        "model": model,
        "prompt_id": prompt_id,
        "n_shot": n_shot,
        "n_trials": n_trials,
    }

    def _save() -> None:
        atomic_write_results(out_path, meta, results)

    new_calls = 0
    for trial in range(n_trials):
        done = done_by_trial.get(trial, set())
        if len(done) >= n_per_trial:
            continue
        remaining = n_per_trial - len(done)
        available = [s for s in test_sentences if s not in done]
        sampled = rng.sample(available, remaining)
        logger.info("[%s | %s] trial %d: %d calls", condition, model, trial, remaining)

        for offset, sentence in enumerate(sampled):
            shots = rng.sample(demo_pairs, n_shot)
            prompt = build_prompt(shots, sentence, language, prompt_id)
            if dry_run:
                logger.info("dry-run prompt:\n%s", prompt)
                continue

            extra_body = None
            if provider == "openai" and reasoning_effort:
                extra_body = {"reasoning_effort": reasoning_effort}
            try:
                reply = call_llm(
                    prompt,
                    model,
                    provider,
                    temperature=temperature,
                    max_tokens=None,
                    extra_body=extra_body,
                )
            except ApiError:
                logger.exception("API failure; saving partial results and stopping.")
                _save()
                raise

            results.append(
                {
                    "trial": trial,
                    "index": len(done) + offset,
                    "sentence": sentence,
                    "response": reply["text"],
                    "prompt": prompt,
                    "demonstrations": [
                        {"sentence": s, "label": lab} for s, lab in shots
                    ],
                    "model_name": reply["model_name"],
                    "tokens": f"{reply['prompt_tokens']}+{reply['completion_tokens']}",
                    "finish_reason": reply["finish_reason"],
                    "latency_s": round(reply["latency_s"], 3),
                }
            )
            new_calls += 1
            if new_calls % CHECKPOINT_EVERY_N_NEW_CALLS == 0:
                _save()

    if not dry_run:
        _save()
    logger.info("Done: %s (%d new calls, %d rows total)", out_path, new_calls, len(results))
    return {"new_calls": new_calls, "n_results": len(results), "path": str(out_path)}
