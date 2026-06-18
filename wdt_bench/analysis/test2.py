"""Analysis for Test 2 (deletion preference: node rule vs parent rule).

Each Type-1 sentence offers two natural single-constituent deletions.  The
demonstration deletes an NP, so a test deletion of the NP inside the PP
generalizes the *node category* (``node_rule``), while deleting the whole PP
generalizes the *parent category* (``parent_rule``).  The question set stores
the candidate spans accordingly: ``expected_for_node_rule`` is the NP and
``expected_for_parent_rule`` is the PP, and the labels follow these fields
directly.

Output: ``results/processed/general_tests/{run}/test2_classified.json``.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Sequence

import numpy as np

from .. import paths
from ..io_utils import atomic_write_json, load_json, load_results
from ..stats import paired_bootstrap_node_minus_parent, stars_from_p
from ..text_utils import normalize_for_constituent_label

logger = logging.getLogger(__name__)

_BOOTSTRAP_SEED = 0

def _same_span(a: Sequence[str], b: Sequence[str], *, norm) -> bool:
    return norm(list(a)) == norm(list(b))


def _label_one(
    extraction_status: str,
    test_tokens: List[str],
    constituent_spans: set,
    delete_span: List[int] | None,
    expected_node: List[str],
    expected_parent: List[str],
    *,
    norm,
) -> str:
    """Label one deletion span.

    ``expected_node`` is the NP (node rule: delete the same category as the
    demonstration); ``expected_parent`` is the PP (parent rule: delete the
    parent constituent).
    """
    if extraction_status != "OK" or delete_span is None:
        return "other"
    start, end = int(delete_span[0]), int(delete_span[1])
    span_toks = test_tokens[start : end + 1]

    if _same_span(span_toks, expected_node, norm=norm):
        return "node_rule"
    if _same_span(span_toks, expected_parent, norm=norm):
        return "parent_rule"
    if (start, end) in constituent_spans:
        return "other_constituent"
    return "non_constituent"


def _bootstrap_ratio_subset(
    flags: Sequence[bool],
    seed: int,
    n_boot: int = 1000,
) -> tuple[float, float, float]:
    arr = np.asarray([1.0 if x else 0.0 for x in flags], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    obs = float(arr.mean())
    boots: List[float] = []
    n = arr.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(arr[idx].mean()))
    b = np.asarray(boots)
    low = float(np.quantile(b, 0.025))
    high = float(np.quantile(b, 0.975))
    return obs, low, high


def _load_dataset_map() -> Dict[str, Dict[str, Any]]:
    path = paths.general_questions(2)
    if not path.is_file():
        raise FileNotFoundError(f"Question set not found: {path}")
    return {str(rec["trial_id"]): rec for rec in load_json(path)}


def run(run_name: str) -> Dict[str, Any]:
    """Classify one Test 2 run and write ``test2_classified.json``."""
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from nltk.tree import Tree
    from ..trees import _constituent_subtree_spans

    raw_path = paths.general_raw(2, run_name)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw model output not found: {raw_path}")
    raw_rows, _meta_in = load_results(raw_path)
    ds_map = _load_dataset_map()

    classified: List[Dict[str, Any]] = []
    labels: List[str] = []
    n_missing_ds = 0

    # Cache parsed trees and their constituent spans.
    _tree_span_cache: Dict[str, set] = {}

    def _get_constituent_spans(ds: Dict[str, Any]) -> set:
        tid = str(ds.get("trial_id", ""))
        if tid not in _tree_span_cache:
            tree_str = (ds["test"].get("parsed_tree_string") or "").strip()
            try:
                tree = Tree.fromstring(tree_str) if tree_str else None
            except Exception:
                tree = None
            _tree_span_cache[tid] = _constituent_subtree_spans(tree) if tree else set()
        return _tree_span_cache[tid]

    for row in raw_rows:
        row = {k: v for k, v in row.items() if k != "prompt"}  # prompt stays in the raw file
        tid = str(row.get("trial_id"))
        ds = ds_map.get(tid)
        if not ds:
            n_missing_ds += 1
            classified.append(
                {
                    **row,
                    "label": "other",
                    "normalized_deleted": [],
                    "eval_note_dataset": "missing_dataset_trial",
                }
            )
            labels.append("other")
            continue

        test_tokens = list(ds["test"]["tokens"])
        constituent_spans = _get_constituent_spans(ds)
        exp_n = list(ds.get("expected_for_node_rule") or [])
        exp_p = list(ds.get("expected_for_parent_rule") or [])

        lab = _label_one(
            str(row.get("extraction_status") or ""),
            test_tokens,
            constituent_spans,
            row.get("delete_span"),
            exp_n,
            exp_p,
            norm=normalize_for_constituent_label,
        )
        labels.append(lab)

        norm_del: List[str] = []
        if row.get("delete_span"):
            a, b = int(row["delete_span"][0]), int(row["delete_span"][1])
            norm_del = normalize_for_constituent_label(test_tokens[a : b + 1])

        classified.append({**row, "label": lab, "normalized_deleted": norm_del})

    n = len(labels)
    if n == 0:
        raise ValueError(f"No rows found in {raw_path}")

    n_other = sum(1 for x in labels if x == "other")

    explain_mask = [x in ("node_rule", "parent_rule", "other_constituent") for x in labels]
    node_m = [x == "node_rule" for x in labels]
    par_m = [x == "parent_rule" for x in labels]

    n_node = sum(node_m)
    n_parent = sum(par_m)
    n_oc = sum(1 for x in labels if x == "other_constituent")

    explained_denom = n_node + n_parent + n_oc
    er_node = n_node / explained_denom if explained_denom else float("nan")
    er_parent = n_parent / explained_denom if explained_denom else float("nan")

    node_subset = [node_m[i] for i in range(n) if explain_mask[i]]
    parent_subset = [par_m[i] for i in range(n) if explain_mask[i]]

    # Bootstrap seeds: the PP-deletion subset uses seed+1 and the
    # NP-deletion subset seed+2.
    _, ci_parent_low, ci_parent_high = _bootstrap_ratio_subset(parent_subset, _BOOTSTRAP_SEED + 1)
    _, ci_node_low, ci_node_high = _bootstrap_ratio_subset(node_subset, _BOOTSTRAP_SEED + 2)

    p_np = paired_bootstrap_node_minus_parent(labels, seed=_BOOTSTRAP_SEED + 3)

    summary_row: Dict[str, Any] = {
        "task": "1_2",
        "model": raw_rows[0].get("model", "") if raw_rows else "",
        "provider": raw_rows[0].get("provider", "") if raw_rows else "",
        "n_trials": n,
        "n_other": n_other,
        "constituent_rate": "",
        "ci_low": "",
        "ci_high": "",
        "baseline_constituent_rate": "",
        "p_vs_baseline": "",
        "explained_node": er_node,
        "explained_parent": er_parent,
        "ci_node_low": ci_node_low,
        "ci_node_high": ci_node_high,
        "ci_parent_low": ci_parent_low,
        "ci_parent_high": ci_parent_high,
        "p_node_vs_parent": p_np,
        "p_node_rule_all": n_node / n if n else 0.0,
        "p_parent_rule_all": n_parent / n if n else 0.0,
        "p_other_constituent_all": n_oc / n if n else 0.0,
        "p_non_constituent_all": sum(1 for x in labels if x == "non_constituent") / n if n else 0.0,
        "p_other_all": n_other / n if n else 0.0,
        "significance_stars_node_vs_parent": stars_from_p(p_np),
    }

    ctr_variant = Counter(str(r.get("prompt_variant") or "") for r in classified)
    summary_row["counts_by_prompt_variant"] = dict(ctr_variant)

    payload = {
        "meta": {
            "task": "1_2",
            "test": 2,
            "run": run_name,
            "model": summary_row.get("model", ""),
            "provider": summary_row.get("provider", ""),
            "n_results": len(classified),
            "n_missing_dataset_trial": n_missing_ds,
        },
        "summary": summary_row,
        "results": classified,
    }

    out_path = paths.general_classified(2, run_name)
    atomic_write_json(out_path, payload)
    logger.info("[Test 2 | %s] wrote %s (%d rows)", run_name, out_path, n)
    return payload
