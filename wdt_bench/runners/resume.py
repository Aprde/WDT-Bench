"""Checkpoint/resume support for the General-Tests runner.

Every API call has a canonical slot ``(variant, replicate, question)``; the
raw-results JSON is rewritten atomically every few calls so an interrupted run
can resume exactly where it stopped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..io_utils import atomic_write_results, load_results

SlotKey = Tuple[str, int, int]

CHECKPOINT_EVERY_N_NEW_CALLS = 5
TQDM_BAR_FORMAT = "{desc}: {percentage:4.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

_VALID_VARIANTS = frozenset({"A", "B", "C", "D", "E", "F"})


def canonical_slot_keys(
    prompt_variants: tuple[str, ...],
    n_replicates: int,
    n_questions_per_replicate: int,
) -> List[SlotKey]:
    return [
        (str(v).strip().upper(), r, q)
        for v in prompt_variants
        for r in range(1, n_replicates + 1)
        for q in range(1, n_questions_per_replicate + 1)
    ]


def slot_key_from_row(row: Dict[str, Any]) -> SlotKey | None:
    try:
        v = str(row.get("prompt_variant") or "").strip().upper()
        if v not in _VALID_VARIANTS:
            return None
        return (v, int(row["replicate_index"]), int(row["question_index_in_replicate"]))
    except (KeyError, TypeError, ValueError):
        return None


def row_api_succeeded(row: Dict[str, Any]) -> bool:
    return row.get("error") is None


def load_existing_by_slot(out_path: Path) -> tuple[Dict[SlotKey, Dict[str, Any]], Dict[str, Any] | None]:
    if not out_path.is_file():
        return {}, None
    rows, meta = load_results(out_path)
    by_slot: Dict[SlotKey, Dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            sk = slot_key_from_row(row)
            if sk is not None:
                by_slot[sk] = row
    return by_slot, (meta or None)


def assert_compatible_resume(
    old_meta: Dict[str, Any] | None,
    *,
    question_file_rel: str,
    trials_len: int,
    n_replicates: int,
    n_questions_per_replicate: int,
    variants: tuple[str, ...],
    task: str | None,
) -> None:
    """Refuse to resume into a file produced under different run settings."""
    if not old_meta:
        return

    def fail(field: str, old: Any, new: Any) -> None:
        raise SystemExit(
            f"Cannot resume: existing raw output was produced with {field}={old!r}, "
            f"but this run uses {field}={new!r}. Move or delete the file to start fresh."
        )

    ov = old_meta.get("variants")
    if isinstance(ov, list):
        old_v = tuple(str(x).strip().upper() for x in ov)
        new_v = tuple(str(x).strip().upper() for x in variants)
        if old_v != new_v:
            fail("variants", old_v, new_v)
    if old_meta.get("n_replicates") not in (None, n_replicates):
        fail("n_replicates", old_meta.get("n_replicates"), n_replicates)
    if old_meta.get("n_questions_per_replicate") not in (None, n_questions_per_replicate):
        fail("n_questions_per_replicate", old_meta.get("n_questions_per_replicate"), n_questions_per_replicate)
    odc = old_meta.get("dataset_trial_count")
    if odc is not None and int(odc) != trials_len:
        fail("dataset_trial_count", odc, trials_len)
    oqf = old_meta.get("question_file")
    if oqf is not None and str(oqf) != str(question_file_rel):
        fail("question_file", oqf, question_file_rel)
    if task is not None and old_meta.get("task") not in (None, task):
        fail("task", old_meta.get("task"), task)


def materialize_results(by_slot: Dict[SlotKey, Dict[str, Any]], canonical_keys: List[SlotKey]) -> List[Dict[str, Any]]:
    return [by_slot[k] for k in canonical_keys if k in by_slot]


def write_checkpoint(
    out_path: Path,
    base_meta: Dict[str, Any],
    by_slot: Dict[SlotKey, Dict[str, Any]],
    canonical_keys: List[SlotKey],
) -> None:
    atomic_write_results(out_path, base_meta, materialize_results(by_slot, canonical_keys))
