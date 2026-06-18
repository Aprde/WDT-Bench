"""Unified runner for the General Tests (Tests 1-3).

The three tests share the exact same querying protocol (six prompt variants x
``n_replicates`` x ``n_questions_per_replicate`` slots, deterministic mapping
of slots to dataset trials, immediate deleted-span extraction, slot-based
resume); they differ only in their question file.  This module replaces the
three nearly-identical ``run_llm_1_*.py`` scripts and the multi-demonstration
plumbing.
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from .. import config, paths
from ..io_utils import load_json
from ..llm import ApiError, call_llm
from ..prompts.general import (
    ALL_PROMPT_VARIANTS,
    build_prompt,
    resolve_demo_list,
)
from ..text_utils import (
    parse_first_line_model_output,
    EXTRACTION_OK_CHAR_FALLBACK,
    extract_deleted_span,
    extract_deleted_string_char_level,
    inclusive_token_span_from_char_deletes,
    normalize_llm_output_for_diff,
)
from .resume import (
    CHECKPOINT_EVERY_N_NEW_CALLS,
    TQDM_BAR_FORMAT,
    assert_compatible_resume,
    canonical_slot_keys,
    load_existing_by_slot,
    row_api_succeeded,
    write_checkpoint,
)

# Experiment-protocol constants (Tests 1-3 in the paper).
N_REPLICATES = 100
N_QUESTIONS_PER_REPLICATE = 24

_TEMPERATURE = 0.0
_MAX_TOKENS = 256
_TIMEOUT = 60
_MAX_RETRIES = 3


def parse_variants(csv: str | None) -> tuple[str, ...]:
    if not csv:
        return ALL_PROMPT_VARIANTS
    parts = [p.strip().upper() for p in csv.split(",") if p.strip()]
    if not parts:
        return ALL_PROMPT_VARIANTS
    for p in parts:
        if p not in ALL_PROMPT_VARIANTS:
            raise SystemExit(f"Unknown prompt variant {p!r}; allowed: {ALL_PROMPT_VARIANTS}")
    return tuple(parts)


def run_general_test(
    test_id: int,
    *,
    model: str | None = None,
    provider: str | None = None,
    run_name: str | None = None,
    n_demos: int = 1,
    prompt_variants: tuple[str, ...] = ALL_PROMPT_VARIANTS,
    n_replicates: int = N_REPLICATES,
    n_questions_per_replicate: int = N_QUESTIONS_PER_REPLICATE,
    out_path: Path | None = None,
) -> Path:
    """Run Test ``test_id`` (1, 2 or 3) and write the raw results JSON.

    ``n_demos > 1`` enables the multi-demonstration setting (Test 1 only in
    the paper, but supported uniformly).  Returns the output path.
    """
    if test_id not in (1, 2, 3):
        raise ValueError("test_id must be 1, 2 or 3")
    if n_demos < 1:
        raise ValueError("n_demos must be >= 1")

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        lg = logging.getLogger(name)
        lg.disabled = True
        lg.propagate = False

    model = model or config.default_chat_model()
    provider = provider or config.default_provider()
    if run_name is None:
        run_name = model if n_demos == 1 else f"{model}_{n_demos}demos"
    if out_path is None:
        out_path = paths.general_raw(test_id, run_name)
    out_path = Path(out_path).resolve()

    dataset_path = paths.general_questions(test_id)
    if not dataset_path.is_file():
        raise SystemExit(f"Missing question file: {dataset_path}")
    trials: List[Dict[str, Any]] = load_json(dataset_path)
    if not trials:
        raise SystemExit(f"Question file is empty: {dataset_path}")

    canonical_keys = canonical_slot_keys(prompt_variants, n_replicates, n_questions_per_replicate)
    total_calls = len(canonical_keys)

    by_slot, old_meta = load_existing_by_slot(out_path)
    task = f"1_{test_id}"
    q_rel = str(dataset_path.relative_to(paths.ROOT))
    assert_compatible_resume(
        old_meta,
        question_file_rel=q_rel,
        trials_len=len(trials),
        n_replicates=n_replicates,
        n_questions_per_replicate=n_questions_per_replicate,
        variants=prompt_variants,
        task=task,
    )
    if old_meta and old_meta.get("n_demos") not in (None, n_demos):
        raise SystemExit(
            f"Cannot resume: existing raw output uses n_demos={old_meta.get('n_demos')!r}, "
            f"this run uses n_demos={n_demos}."
        )

    base_meta: Dict[str, Any] = {
        "test": test_id,
        "task": task,
        "model": model,
        "provider": provider,
        "n_demos": n_demos,
        "variants": list(prompt_variants),
        "n_replicates": n_replicates,
        "n_questions_per_replicate": n_questions_per_replicate,
        "dataset_trial_count": len(trials),
        "question_file": q_rel,
        "n_results_expected": total_calls,
    }

    n_pre_done = sum(1 for k in canonical_keys if k in by_slot and row_api_succeeded(by_slot[k]))
    n_err = sum(1 for k in canonical_keys if k in by_slot and not row_api_succeeded(by_slot[k]))
    print(
        f"[test{test_id}] model={model} provider={provider} n_demos={n_demos}\n"
        f"  Output: {out_path}\n"
        f"  Existing slots: {len(by_slot)}; retryable failed slots: {n_err}\n"
        f"  Completed slots: {n_pre_done}/{total_calls}; variants={list(prompt_variants)}",
        flush=True,
    )

    n_new_since_checkpoint = 0
    try:
        with tqdm(
            total=total_calls,
            initial=n_pre_done,
            desc=f"test{test_id}-{n_demos}demo",
            unit="call",
            bar_format=TQDM_BAR_FORMAT,
            dynamic_ncols=False,
            mininterval=0.5,
            miniters=1,
            file=sys.stdout,
            leave=True,
        ) as pbar:
            for variant in prompt_variants:
                vkey = str(variant).strip().upper()
                for rep_1 in range(1, n_replicates + 1):
                    for q_1 in range(1, n_questions_per_replicate + 1):
                        key = (vkey, rep_1, q_1)
                        if key in by_slot and row_api_succeeded(by_slot[key]):
                            continue
                        flat = (rep_1 - 1) * n_questions_per_replicate + (q_1 - 1)
                        row = _run_one_slot(
                            trials, flat, variant, model, provider, n_demos,
                        )
                        row.update({
                            "prompt_variant": variant,
                            "replicate_index": rep_1,
                            "question_index_in_replicate": q_1,
                            "flat_index": flat,
                            "dataset_index": flat % len(trials),
                        })
                        by_slot[key] = row
                        n_new_since_checkpoint += 1
                        if n_new_since_checkpoint >= CHECKPOINT_EVERY_N_NEW_CALLS:
                            write_checkpoint(out_path, base_meta, by_slot, canonical_keys)
                            n_new_since_checkpoint = 0
                        pbar.update(1)
    finally:
        if by_slot:
            write_checkpoint(out_path, base_meta, by_slot, canonical_keys)

    return out_path


def _run_one_slot(
    trials: List[Dict[str, Any]],
    flat: int,
    variant: str,
    model: str,
    provider: str,
    n_demos: int,
) -> Dict[str, Any]:
    rec = trials[flat % len(trials)]
    tid = str(rec.get("trial_id", ""))

    if n_demos > 1:
        demos, test, demo_indices = resolve_demo_list(trials, flat, n_demos)
        demo_pairs = [(d["sentence"], d["edited_sentence"]) for d in demos]
    else:
        demos, demo_indices = [rec["demo"]], []
        test = rec["test"]
        demo_pairs = [(rec["demo"]["sentence"], rec["demo"]["edited_sentence"])]

    prompt = build_prompt(demo_pairs, test["sentence"], prompt_variant=variant)

    raw_response = ""
    latency_s = 0.0
    tokens_used: int | None = None
    err: Dict[str, Any] | None = None
    try:
        resp = call_llm(
            prompt,
            model=model,
            provider=provider,
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
            timeout=_TIMEOUT,
            max_retries=_MAX_RETRIES,
        )
        raw_response = resp.get("text") or ""
        latency_s = float(resp.get("latency_s") or 0.0)
        tu = resp.get("tokens_used")
        tokens_used = int(tu) if tu is not None else None
    except ApiError as exc:
        err = {"type": "ApiError", "message": str(exc)}

    # Immediate deleted-span extraction (strict token diff with a
    # character-level fallback), recorded alongside the raw response.
    model_edited = parse_first_line_model_output(raw_response)
    if model_edited:
        model_edited = normalize_llm_output_for_diff(model_edited)
    deleted_string = None
    delete_span = None
    extraction_status = "EMPTY"
    if model_edited:
        deleted_string, delete_span, extraction_status = extract_deleted_span(
            test["tokens"], model_edited
        )
        if extraction_status != "OK":
            fb = extract_deleted_string_char_level(test["sentence"], model_edited)
            if fb.strip():
                deleted_string = fb
                delete_span = inclusive_token_span_from_char_deletes(
                    test["tokens"], test["sentence"], model_edited
                )
                extraction_status = EXTRACTION_OK_CHAR_FALLBACK

    row: Dict[str, Any] = {
        "trial_id": tid,
        "test": {"sentence": test["sentence"], "tokens": test["tokens"]},
        "model": model,
        "provider": provider,
        "prompt": prompt,
        "raw_response": raw_response,
        "model_edited_test": model_edited,
        "deleted_string": deleted_string,
        "delete_span": list(delete_span) if delete_span else None,
        "extraction_status": extraction_status,
        "latency_s": latency_s,
        "tokens_used": tokens_used,
        "error": err,
    }
    if n_demos > 1:
        row["n_demos"] = n_demos
        row["demo_dataset_indices"] = demo_indices
        for k, demo in enumerate(demos, start=1):
            row[f"demo{k}"] = {"sentence": demo["sentence"], "edited_sentence": demo["edited_sentence"]}
    else:
        row["demo"] = {"sentence": demos[0]["sentence"], "edited_sentence": demos[0]["edited_sentence"]}
    return row
