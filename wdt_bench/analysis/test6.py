"""Analysis for Test 6 (structural-ambiguity conditions).

Two sub-conditions are classified with rule-based span labels:

* ``ambiguity_pp``      - PP-attachment ambiguity; target label ``np2_pp``
                          (object NP deleted together with the PP).
* ``ambiguity_adjunct`` - adjunct/relative-clause ambiguity; target label
                          ``np2_rc`` (NP2 deleted together with the RC).

For each sub-condition, rates are reported overall and split by structure
type (1 vs 2; see ``diagnostic_utils`` for the sentence lists).  Output goes
to ``results/processed/diagnostic_tests/test6_classified.json``; seeds and
ordering are deterministic across re-runs.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from .. import paths
from ..io_utils import atomic_write_json
from ..stats import bootstrap_ci_proportion
from .diagnostic_utils import (
    classify_ambiguity_adjunct,
    classify_ambiguity_pp,
    extract_deleted,
    flatten_response,
    get_structure_type_adjunct,
    get_structure_type_pp,
    load_condition_rows,
)

logger = logging.getLogger(__name__)

_BOOTSTRAP_SEED = 0


def _classify_row(row: Dict[str, Any], sub_condition: str) -> Dict[str, Any]:
    sentence = str(row.get("sentence", "")).strip()
    response = flatten_response(str(row.get("response", ""))).strip()
    deleted = extract_deleted(sentence, response)
    if sub_condition == "ambiguity_pp":
        label = classify_ambiguity_pp(sentence, deleted)
        structure = get_structure_type_pp(sentence)
    else:
        label = classify_ambiguity_adjunct(sentence, deleted)
        structure = get_structure_type_adjunct(sentence)
    return {
        "sub_condition": sub_condition,
        "model": row.get("model", ""),
        "source": row.get("source", ""),
        "trial": row.get("trial"),
        "sentence": sentence,
        "response": response,
        "deleted_str": deleted,
        "label": label,
        "structure": structure,  # 1 or 2 (0 = unknown)
    }


def _per_trial_rates_by_structure(
    rows: List[Dict[str, Any]],
    target_label: str,
    structures: Tuple[int, ...] = (1, 2),
) -> Dict[int, List[float]]:
    """Per-trial target-label rates, split by structure type."""
    result: Dict[int, List[float]] = {s: [] for s in structures}
    for struct in structures:
        sub = [r for r in rows if r["structure"] == struct]
        by_trial: Dict[int, List[Dict]] = defaultdict(list)
        for r in sub:
            by_trial[int(r["trial"])].append(r)
        for trial in sorted(by_trial.keys()):
            trial_rows = by_trial[trial]
            n = len(trial_rows)
            k = sum(1 for r in trial_rows if r["label"] == target_label)
            result[struct].append(k / n if n else 0.0)
    return result


def _summarize_by_model(
    results: List[Dict[str, Any]],
    sub_cond: str,
    target_label: str,
) -> List[Dict[str, Any]]:
    sub = [r for r in results if r["sub_condition"] == sub_cond]
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for r in sub:
        by_model[r["model"]].append(r)

    summaries = []
    for model, rows in sorted(by_model.items()):
        n = len(rows)
        ctr = Counter(r["label"] for r in rows)
        target_flags = [r["label"] == target_label for r in rows]
        target_rate, ci_lo, ci_hi = bootstrap_ci_proportion(target_flags, seed=_BOOTSTRAP_SEED)

        per_trial_by_struct = _per_trial_rates_by_structure(rows, target_label)

        struct_stats: Dict[str, Any] = {}
        for struct in (1, 2):
            struct_rows = [r for r in rows if r["structure"] == struct]
            ns = len(struct_rows)
            if ns:
                flags = [r["label"] == target_label for r in struct_rows]
                rate, slo, shi = bootstrap_ci_proportion(flags, seed=_BOOTSTRAP_SEED + struct)
            else:
                rate, slo, shi = 0.0, 0.0, 0.0
            struct_stats[f"structure_{struct}"] = {
                "n": ns,
                f"{target_label}_rate": rate,
                "ci_lo": slo,
                "ci_hi": shi,
                "per_trial_rates": per_trial_by_struct[struct],
            }

        summaries.append({
            "sub_condition": sub_cond,
            "model": model,
            "n": n,
            "target_label": target_label,
            f"{target_label}_rate": target_rate,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "label_counts": dict(ctr),
            "by_structure": struct_stats,
        })
    return summaries


def _summarize_overall_by_structure(
    results: List[Dict[str, Any]],
    sub_cond: str,
    target_label: str,
) -> Dict[str, Any]:
    sub = [r for r in results if r["sub_condition"] == sub_cond]
    overall: Dict[str, Any] = {"sub_condition": sub_cond, "target_label": target_label}
    for struct in (1, 2):
        sr = [r for r in sub if r["structure"] == struct]
        n = len(sr)
        k = sum(1 for r in sr if r["label"] == target_label)
        overall[f"structure_{struct}"] = {
            "n": n,
            f"{target_label}_rate": k / n if n else 0.0,
            "label_counts": dict(Counter(r["label"] for r in sr)),
        }
    return overall


def run() -> Dict[str, Any]:
    """Classify Test 6 and write ``test6_classified.json``."""
    pp_rows_raw = load_condition_rows("ambiguity_pp")
    adj_rows_raw = load_condition_rows("ambiguity_adjunct")
    if not pp_rows_raw and not adj_rows_raw:
        raise FileNotFoundError(
            f"No raw results found for Test 6 in {paths.RAW_DIAGNOSTIC}"
        )

    results: List[Dict[str, Any]] = []
    for row in pp_rows_raw:
        results.append(_classify_row(row, "ambiguity_pp"))
    for row in adj_rows_raw:
        results.append(_classify_row(row, "ambiguity_adjunct"))

    pp_results = [r for r in results if r["sub_condition"] == "ambiguity_pp"]
    adj_results = [r for r in results if r["sub_condition"] == "ambiguity_adjunct"]
    s0_pp = sum(1 for r in pp_results if r["structure"] == 0)
    s0_adj = sum(1 for r in adj_results if r["structure"] == 0)
    logger.info("[Test 6] ambiguity_pp: %d rows, unknown structure: %d", len(pp_rows_raw), s0_pp)
    logger.info("[Test 6] ambiguity_adjunct: %d rows, unknown structure: %d", len(adj_rows_raw), s0_adj)

    payload = {
        "meta": {
            "test": 6,
            "description": (
                "ambiguity: NP+PP / NP+RC deletion rate by structure type "
                "(1 vs 2 attachment bias)"
            ),
            "n_pp": len(pp_rows_raw),
            "n_adjunct": len(adj_rows_raw),
            "n_total": len(results),
        },
        "summary_pp_by_model": _summarize_by_model(results, "ambiguity_pp", "np2_pp"),
        "summary_adjunct_by_model": _summarize_by_model(results, "ambiguity_adjunct", "np2_rc"),
        "overall_pp": _summarize_overall_by_structure(results, "ambiguity_pp", "np2_pp"),
        "overall_adjunct": _summarize_overall_by_structure(results, "ambiguity_adjunct", "np2_rc"),
        "results": results,
    }

    out_path = paths.diagnostic_classified(6)
    atomic_write_json(out_path, payload)
    logger.info("[Test 6] wrote %s (%d rows)", out_path, len(results))
    return payload
