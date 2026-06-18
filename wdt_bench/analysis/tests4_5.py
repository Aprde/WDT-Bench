"""Analysis for Test 4 (parallel sentences) and Test 5 (nonsense sentences).

Both tests use the same node-rule vs parent-rule classification against a
constituency-tree bank; they differ only in the stimuli (and Test 5 reports
the Test 4 summary alongside as a reference).  This module therefore exposes
one parameterised entry point::

    run(4)   ->  results/processed/diagnostic_tests/test4_classified.json
    run(5)   ->  results/processed/diagnostic_tests/test5_classified.json

The summary schema, bootstrap seeds and row ordering are identical to the
deterministic: English-condition files first, then trials ascending.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from .. import paths
from ..io_utils import atomic_write_json, load_json
from ..stats import bootstrap_ci_proportion
from ..trees import load_tree_map
from .diagnostic_utils import (
    classify_node_parent_from_tree,
    extract_deleted,
    flatten_response,
    is_single_constituent_from_tree,
    load_condition_rows,
)

logger = logging.getLogger(__name__)

_BOOTSTRAP_SEED = 0
_LABELS = ("node_rule", "parent_rule", "other", "no_deletion")

_TEST_TYPE = {4: "parallel", 5: "nonsense"}
_DESCRIPTION = {
    4: "parallel sentences: node_rule vs parent_rule (tree-based)",
    5: "nonsense sentences: node_rule vs parent_rule (tree-based); vs Test 4 comparison",
}


def _classify_row(row: Dict[str, Any], language: str, tree_map: Dict[str, str]) -> Dict[str, Any]:
    sentence = str(row.get("sentence", "")).strip()
    response = flatten_response(str(row.get("response", ""))).strip()
    deleted = extract_deleted(sentence, response)
    label = classify_node_parent_from_tree(sentence, deleted, tree_map)
    return {
        "condition": row.get("condition", ""),
        "language": language,
        "model": row.get("model", ""),
        "source": row.get("source", ""),
        "trial": row.get("trial"),
        "sentence": sentence,
        "response": response,
        "deleted_str": deleted,
        "label": label,
        "is_single_constituent": is_single_constituent_from_tree(sentence, deleted, tree_map),
    }


def _per_trial_rates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-trial label rates (the unit used for permutation tests in plots)."""
    by_trial: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        by_trial[int(r["trial"])].append(r)

    per_trial = []
    for trial in sorted(by_trial.keys()):
        sub = by_trial[trial]
        n = len(sub)
        ctr = Counter(r["label"] for r in sub)
        n_node = ctr.get("node_rule", 0)
        n_parent = ctr.get("parent_rule", 0)
        n_single = sum(1 for r in sub if r.get("is_single_constituent"))
        entry: Dict[str, Any] = {
            "trial": trial,
            "n": n,
            "single_constituent_n": n_single,
            "single_constituent_rate": n_single / n if n else 0.0,
        }
        for lab in _LABELS:
            entry[f"{lab}_rate"] = ctr.get(lab, 0) / n if n else 0.0
        entry["node_rule_given_single_constituent_rate"] = (
            n_node / n_single if n_single else 0.0
        )
        entry["parent_rule_given_single_constituent_rate"] = (
            n_parent / n_single if n_single else 0.0
        )
        per_trial.append(entry)
    return per_trial


def _summarize_by_model(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_model_lang: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in results:
        by_model_lang[(r["model"], r["language"])].append(r)

    summaries = []
    for (model, lang), rows in sorted(by_model_lang.items()):
        n = len(rows)
        ctr = Counter(r["label"] for r in rows)
        node_flags = [r["label"] == "node_rule" for r in rows]
        parent_flags = [r["label"] == "parent_rule" for r in rows]
        single_rows = [r for r in rows if r.get("is_single_constituent")]
        n_single = len(single_rows)

        # Bootstrap seeds: PP-deletion flags use seed and seed+2,
        # NP-deletion flags seed+1 and seed+3 (node = NP, parent = PP).
        parent_rate, par_ci_lo, par_ci_hi = bootstrap_ci_proportion(
            parent_flags, seed=_BOOTSTRAP_SEED
        )
        node_rate, node_ci_lo, node_ci_hi = bootstrap_ci_proportion(
            node_flags, seed=_BOOTSTRAP_SEED + 1
        )
        if n_single:
            node_single_flags = [r["label"] == "node_rule" for r in single_rows]
            parent_single_flags = [r["label"] == "parent_rule" for r in single_rows]
            parent_single_rate, par_single_ci_lo, par_single_ci_hi = bootstrap_ci_proportion(
                parent_single_flags, seed=_BOOTSTRAP_SEED + 2
            )
            node_single_rate, node_single_ci_lo, node_single_ci_hi = bootstrap_ci_proportion(
                node_single_flags, seed=_BOOTSTRAP_SEED + 3
            )
        else:
            node_single_rate = node_single_ci_lo = node_single_ci_hi = 0.0
            parent_single_rate = par_single_ci_lo = par_single_ci_hi = 0.0

        summaries.append({
            "model": model,
            "language": lang,
            "n": n,
            "single_constituent_n": n_single,
            "single_constituent_rate": n_single / n if n else 0.0,
            "node_rule_rate": node_rate,
            "ci_node_lo": node_ci_lo,
            "ci_node_hi": node_ci_hi,
            "parent_rule_rate": parent_rate,
            "ci_parent_lo": par_ci_lo,
            "ci_parent_hi": par_ci_hi,
            "node_rule_given_single_constituent_rate": node_single_rate,
            "ci_node_given_single_constituent_lo": node_single_ci_lo,
            "ci_node_given_single_constituent_hi": node_single_ci_hi,
            "parent_rule_given_single_constituent_rate": parent_single_rate,
            "ci_parent_given_single_constituent_lo": par_single_ci_lo,
            "ci_parent_given_single_constituent_hi": par_single_ci_hi,
            "label_counts": dict(ctr),
            "per_trial_rates": _per_trial_rates(rows),
        })
    return summaries


def run(test_id: int) -> Dict[str, Any]:
    """Classify Test 4 or Test 5 and write ``test{K}_classified.json``."""
    if test_id not in _TEST_TYPE:
        raise ValueError("test_id must be 4 or 5")
    test_type = _TEST_TYPE[test_id]

    tree_path = paths.diagnostic_trees(test_type)
    if not tree_path.is_file():
        raise FileNotFoundError(f"Tree bank not found: {tree_path}")
    tree_map = load_tree_map(tree_path)
    logger.info("[Test %d] loaded %d tree entries (case-doubled)", test_id, len(tree_map))

    en_rows_raw = load_condition_rows(f"{test_type}_english")
    zh_rows_raw = load_condition_rows(f"{test_type}_chinese")
    if not en_rows_raw and not zh_rows_raw:
        raise FileNotFoundError(
            f"No raw results found for Test {test_id} in {paths.RAW_DIAGNOSTIC}"
        )

    results: List[Dict[str, Any]] = []
    for row in en_rows_raw:
        results.append(_classify_row(row, "english", tree_map))
    for row in zh_rows_raw:
        results.append(_classify_row(row, "chinese", tree_map))

    total = len(results)
    not_other = sum(1 for r in results if r["label"] != "other")
    logger.info(
        "[Test %d] %d rows, %d non-'other' (%.1f%%)",
        test_id, total, not_other, 100 * not_other / total if total else 0.0,
    )

    payload: Dict[str, Any] = {
        "meta": {
            "test": test_id,
            "description": _DESCRIPTION[test_id],
            "tree_bank": tree_path.name,
            "n_english": len(en_rows_raw),
            "n_chinese": len(zh_rows_raw),
            "n_total": total,
        },
        "summary_by_model": _summarize_by_model(results),
        "results": results,
    }

    if test_id == 5:
        ref_path = paths.diagnostic_classified(4)
        if ref_path.is_file():
            ref = load_json(ref_path)
            payload["parallel_reference"] = {
                "summary_by_model": ref.get("summary_by_model", [])
            }
        else:
            payload["parallel_reference"] = {}
            logger.warning(
                "Test 4 classified file not found; run `analyze_results.py --test 4` "
                "first to embed the parallel reference."
            )

    out_path = paths.diagnostic_classified(test_id)
    atomic_write_json(out_path, payload)
    logger.info("[Test %d] wrote %s (%d rows)", test_id, out_path, total)
    return payload
