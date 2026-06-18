"""Analysis for Test 3 (deletion localization).

Each trial names a specific target constituent to delete; the model's answer
is scored against the gold deletion in three increasingly tolerant ways:

* ``correct_vs_gold``                    - normalised deleted *string* match;
* ``correct_vs_gold_span_delete_field``  - exact token-span match using the
                                           span recorded at run time;
* ``correct_vs_gold_span_recovered``     - as above, but recovering a span
                                           from the raw edited sentence when
                                           none was recorded.

Output: ``results/processed/general_tests/{run}/test3_classified.json``.
"""
from __future__ import annotations

import logging
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .. import paths
from ..io_utils import atomic_write_json, load_json, load_results
from ..prompts.general import ALL_PROMPT_VARIANTS
from ..text_utils import (
    EXTRACTION_OK_CHAR_FALLBACK,
    inclusive_token_span_from_char_deletes,
    normalize_llm_output_for_diff,
    normalize_tokens,
)

logger = logging.getLogger(__name__)


def _gold_norm(gold: Dict[str, Any]) -> List[str]:
    s = (gold.get("expected_deleted_string") or "").strip()
    return normalize_tokens(s)


def _gold_span(ds: Dict[str, Any]) -> Tuple[int, int]:
    g = ds.get("gold") or {}
    lo, hi = g.get("target_leaf_span") or [0, 0]
    return int(lo), int(hi)


def _pred_span_from_delete_field(raw_row: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    sp = raw_row.get("delete_span")
    if sp is not None and len(sp) >= 2:
        return int(sp[0]), int(sp[1])
    return None


def _pred_span_char_recover(
    raw_row: Dict[str, Any],
    ds: Dict[str, Any],
) -> Optional[Tuple[int, int]]:
    test = ds.get("test") or {}
    toks = test.get("tokens") or []
    sent = (test.get("sentence") or "").strip()
    me = (raw_row.get("model_edited_test") or "").strip()
    if not toks or not sent or not me:
        return None
    me2 = normalize_llm_output_for_diff(me)
    sp = inclusive_token_span_from_char_deletes(toks, sent, me2)
    if not sp:
        return None
    return int(sp[0]), int(sp[1])


def row_correct_string(raw_row: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    ext = str(raw_row.get("extraction_status") or "")
    if ext not in ("OK", EXTRACTION_OK_CHAR_FALLBACK):
        return False
    dt = (raw_row.get("deleted_string") or "").strip()
    if not dt:
        return False
    return normalize_tokens(dt) == _gold_norm(gold)


def row_correct_span_strict(raw_row: Dict[str, Any], ds: Dict[str, Any]) -> bool:
    gsp = _gold_span(ds)
    psp = _pred_span_from_delete_field(raw_row)
    return psp is not None and psp == gsp


def row_correct_span_recovered(raw_row: Dict[str, Any], ds: Dict[str, Any]) -> bool:
    gsp = _gold_span(ds)
    psp = _pred_span_from_delete_field(raw_row)
    if psp is None:
        psp = _pred_span_char_recover(raw_row, ds)
    return psp is not None and psp == gsp


def _load_dataset_map() -> Dict[str, Dict[str, Any]]:
    path = paths.general_questions(3)
    if not path.is_file():
        raise FileNotFoundError(f"Question set not found: {path}")
    return {str(rec["trial_id"]): rec for rec in load_json(path)}


def run(run_name: str) -> Dict[str, Any]:
    """Score one Test 3 run and write ``test3_classified.json``."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    raw_path = paths.general_raw(3, run_name)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw model output not found: {raw_path}")
    raw_rows, _meta_in = load_results(raw_path)
    ds_map = _load_dataset_map()

    out_rows: List[Dict[str, Any]] = []
    n_ok = 0
    n_ok_span_del = 0
    n_ok_span_rec = 0
    n_missing = 0
    ext_ctr: Counter[str] = Counter()

    for row in raw_rows:
        ext_ctr[str(row.get("extraction_status") or "MISSING")] += 1
        tid = str(row.get("trial_id", ""))
        ds = ds_map.get(tid)
        if not ds:
            n_missing += 1
            gold: Dict[str, Any] = {}
            ok_s = ok_sd = ok_sr = False
        else:
            gold = ds.get("gold") or {}
            ok_s = row_correct_string(row, gold)
            ok_sd = row_correct_span_strict(row, ds)
            ok_sr = row_correct_span_recovered(row, ds)
            if ok_s:
                n_ok += 1
            if ok_sd:
                n_ok_span_del += 1
            if ok_sr:
                n_ok_span_rec += 1

        er = dict(row)
        er.pop("prompt", None)  # prompt stays in the raw file
        er["gold_expected_deleted_string"] = gold.get("expected_deleted_string") if ds else None
        er["gold_target_leaf_span"] = gold.get("target_leaf_span") if ds else None
        er["gold_chunk_range"] = gold.get("chunk_range") if ds else None
        er["gold_expected_removed_tokens"] = gold.get("expected_removed_tokens") if ds else None
        er["correct_vs_gold"] = ok_s
        er["correct_vs_gold_span_delete_field"] = ok_sd
        er["correct_vs_gold_span_recovered"] = ok_sr
        if not ds:
            er["eval_note_dataset"] = "missing_dataset_trial"
        out_rows.append(er)

    n = len(out_rows)
    if n == 0:
        raise ValueError(f"No rows found in {raw_path}")

    by_variant: Dict[str, Dict[str, Any]] = {}
    for v in ALL_PROMPT_VARIANTS:
        sub = [r for r in out_rows if str(r.get("prompt_variant") or "") == v]
        nv = len(sub)
        n_ok_v = sum(1 for r in sub if r.get("correct_vs_gold"))
        by_variant[v] = {
            "n_rows": nv,
            "n_correct_vs_gold_string": n_ok_v,
            "accuracy_vs_gold": n_ok_v / nv if nv else 0.0,
        }

    summary_row: Dict[str, Any] = {
        "task": "1_3",
        "model": raw_rows[0].get("model", "") if raw_rows else "",
        "provider": raw_rows[0].get("provider", "") if raw_rows else "",
        "n_trials": n,
        "n_correct_vs_gold_string": n_ok,
        "accuracy_vs_gold": n_ok / n if n else 0.0,
        "n_correct_vs_gold_span_delete_field": n_ok_span_del,
        "accuracy_vs_gold_span_delete_field": n_ok_span_del / n if n else 0.0,
        "n_correct_vs_gold_span_recovered": n_ok_span_rec,
        "accuracy_vs_gold_span_recovered": n_ok_span_rec / n if n else 0.0,
        "n_missing_dataset": n_missing,
        "n_ext_OK": ext_ctr.get("OK", 0),
        "n_ext_OK_CHAR_FALLBACK": ext_ctr.get(EXTRACTION_OK_CHAR_FALLBACK, 0),
        "n_ext_EMPTY": ext_ctr.get("EMPTY", 0),
        "n_ext_REORDERED": ext_ctr.get("REORDERED", 0),
        "n_ext_IDENTICAL": ext_ctr.get("IDENTICAL", 0),
        "n_ext_ADDED_TOKENS": ext_ctr.get("ADDED_TOKENS", 0),
        "n_ext_other": sum(
            v
            for k, v in ext_ctr.items()
            if k
            not in (
                "OK",
                EXTRACTION_OK_CHAR_FALLBACK,
                "EMPTY",
                "REORDERED",
                "IDENTICAL",
                "ADDED_TOKENS",
            )
        ),
        "counts_by_prompt_variant": dict(
            Counter(str(r.get("prompt_variant") or "") for r in out_rows)
        ),
        "metrics_by_prompt_variant": by_variant,
    }

    payload = {
        "meta": {
            "task": "1_3",
            "test": 3,
            "run": run_name,
            "model": summary_row.get("model", ""),
            "provider": summary_row.get("provider", ""),
            "n_results": len(out_rows),
            "n_missing_dataset_trial": n_missing,
        },
        "summary": summary_row,
        "results": out_rows,
    }

    out_path = paths.general_classified(3, run_name)
    atomic_write_json(out_path, payload)
    logger.info("[Test 3 | %s] wrote %s (%d rows)", run_name, out_path, n)
    return payload
