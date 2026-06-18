"""Analysis for Test 1 (constituent recognition).

Each raw row is classified with the 5-way tree-span taxonomy
(``tree_span_category``): the deleted span -- taken from the **actual
deleted token positions** recorded at extraction time (the question set's
constituency-tree leaves align one-to-one with the test tokens) -- is
compared against the constituency tree and labelled

* ``single_constituent``                     - exactly one constituent-level node;
* ``multiple_constituents``               - a tiling of several phrasal nodes;
* ``partial_constituent``               - a proper sub-span of one node;
* ``constituent_plus_partial``   - a node plus extra material;
* ``other``                                 - no valid deletion to classify
                                              (see ``eval_note``).

Bare POS leaves such as ``(DT the)`` never count as complete nodes, while
one-word phrasal projections (e.g. ``(NP (PRP it))``) do.
``pos_leaf_only_match`` flags single-token
deletions whose only exact match in the tree is a POS leaf.

Output: ``results/processed/general_tests/{run}/test1_classified.json``.
"""
from __future__ import annotations

import logging
import warnings
from collections import Counter
from typing import Any, Dict, List

from .. import paths
from ..io_utils import atomic_write_json, load_json, load_results
from ..stats import bootstrap_ci_proportion, p_value_vs_heterogeneous_baseline
from ..text_utils import (
    EXTRACTION_OK_CHAR_FALLBACK,
    extract_deleted_string_char_level,
    inclusive_token_span_from_char_deletes,
    normalize_for_constituent_label,
    normalize_llm_output_for_diff,
    normalize_tokens,
)
from ..trees import (
    baseline_constituent_probability_from_tree,
    classify_leaf_span,
    deleted_tokens_to_leaf_span,
    is_pos_leaf_only_span,
)

logger = logging.getLogger(__name__)

_BOOTSTRAP_SEED = 0

SPAN_CATEGORIES = (
    "single_constituent",
    "multiple_constituents",
    "partial_constituent",
    "constituent_plus_partial",
    "other",
)


def _load_dataset_map() -> Dict[str, Dict[str, Any]]:
    path = paths.general_questions(1)
    if not path.is_file():
        raise FileNotFoundError(f"Question set not found: {path}")
    return {str(rec["trial_id"]): rec for rec in load_json(path)}


def run(run_name: str) -> Dict[str, Any]:
    """Classify one Test 1 run and write ``test1_classified.json``."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from nltk.tree import Tree

    raw_path = paths.general_raw(1, run_name)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw model output not found: {raw_path}")
    raw_rows, _meta_in = load_results(raw_path)
    ds_map = _load_dataset_map()

    classified: List[Dict[str, Any]] = []
    baseline_probs: List[float] = []
    n_missing_ds = 0
    one_node_flags: List[bool] = []

    tree_cache: Dict[str, Any] = {}
    leaves_align_cache: Dict[str, bool] = {}

    def _tree_for(tid: str, ds: Dict[str, Any]):
        if tid not in tree_cache:
            ptb_s = (ds["test"].get("parsed_tree_string") or "").strip()
            try:
                tree_cache[tid] = Tree.fromstring(ptb_s) if ptb_s else None
            except Exception:
                tree_cache[tid] = None
            t = tree_cache[tid]
            leaves_align_cache[tid] = bool(
                t is not None
                and [str(x) for x in t.leaves()] == list(ds["test"]["tokens"])
            )
        return tree_cache[tid]

    for row in raw_rows:
        tid = str(row.get("trial_id"))
        ds = ds_map.get(tid)
        if not ds:
            n_missing_ds += 1

        test_tokens = list(ds["test"]["tokens"]) if ds else []

        tree = _tree_for(tid, ds) if ds else None
        bp = baseline_constituent_probability_from_tree(tree) if tree is not None else 0.0
        baseline_probs.append(bp)

        out_row: Dict[str, Any] = dict(row)
        # The full prompt stays in the raw file only; keep classified rows lean.
        out_row.pop("prompt", None)

        ext = str(row.get("extraction_status") or "")

        # ---- deleted text + eval_note ladder --------------------------------
        del_text = ""
        if ds:
            if ext in ("OK", EXTRACTION_OK_CHAR_FALLBACK):
                del_text = (row.get("deleted_string") or "").strip()
                if not del_text and row.get("delete_span"):
                    a, b = int(row["delete_span"][0]), int(row["delete_span"][1])
                    del_text = " ".join(test_tokens[a : b + 1])
            else:
                me = (row.get("model_edited_test") or "").strip()
                if me:
                    me = normalize_llm_output_for_diff(me)
                    del_text = extract_deleted_string_char_level(
                        ds["test"]["sentence"], me
                    )
            del_text = del_text.strip()

        tree = _tree_for(tid, ds) if ds else None
        del_toks = normalize_tokens(del_text) if del_text else []

        if not ds:
            out_row["eval_note"] = "missing_dataset_trial"
        elif not del_text:
            out_row["eval_note"] = (
                f"no_deletion_text_llm_status_{ext}" if ext else "no_deletion_text"
            )
        elif not del_toks:
            out_row["eval_note"] = "empty_normalized_deleted_tokens"
        elif not (ds["test"].get("parsed_tree_string") or "").strip():
            out_row["eval_note"] = "missing_parsed_tree_string"
        elif tree is None:
            out_row["eval_note"] = "ptb_parse_error"
        else:
            out_row["eval_note"] = None

        # ---- recover the actual deleted token span --------------------------
        span_use: List[int] | None = None
        if ds:
            raw_sp = row.get("delete_span")
            if raw_sp is not None and len(raw_sp) >= 2:
                span_use = [int(raw_sp[0]), int(raw_sp[1])]
            elif ext in ("OK", EXTRACTION_OK_CHAR_FALLBACK) and (row.get("model_edited_test") or "").strip():
                me = normalize_llm_output_for_diff((row.get("model_edited_test") or "").strip())
                sp = inclusive_token_span_from_char_deletes(
                    test_tokens,
                    ds["test"]["sentence"],
                    me,
                )
                if sp:
                    span_use = [sp[0], sp[1]]
                    out_row["delete_span"] = span_use

        # ---- tree-span classification ---------------------------------------
        # The tree leaves align one-to-one with the test tokens, so the
        # extraction-time ``delete_span`` (the positions the model actually
        # deleted) is used directly as the leaf span; token alignment is the
        # fallback for rows where no span was recorded.
        tree_span_category = "other"
        pos_leaf_only = False
        if tree is not None and out_row["eval_note"] is None:
            leaf_sp = None
            if span_use is not None and leaves_align_cache.get(tid):
                leaf_sp = (int(span_use[0]), int(span_use[1]))
            elif del_toks:
                leaf_sp = deleted_tokens_to_leaf_span(tree, del_toks)
            if leaf_sp is not None:
                lo, hi = leaf_sp
                try:
                    tree_span_category = classify_leaf_span(tree, lo, hi)
                    pos_leaf_only = is_pos_leaf_only_span(tree, lo, hi)
                except Exception:
                    tree_span_category = "other"
                    pos_leaf_only = False
        out_row["tree_span_category"] = tree_span_category
        out_row["pos_leaf_only_match"] = pos_leaf_only
        one_node_flags.append(tree_span_category == "single_constituent")

        demo = (ds or {}).get("demo") or {}
        out_row["demo_removed_node_path"] = demo.get("demo_removed_node_path") or ""
        out_row["demo_removed_node_path_labels"] = demo.get("demo_removed_node_path_labels") or []

        norm_del: List[str] = []
        if ds and span_use:
            a, b = span_use[0], span_use[1]
            norm_del = normalize_for_constituent_label(test_tokens[a : b + 1])

        out_row["baseline_constituent_prob_trial"] = bp
        out_row["normalized_deleted"] = norm_del

        classified.append(out_row)

    n = len(classified)
    if n == 0:
        raise ValueError(f"No rows found in {raw_path}")

    mean_bp = sum(baseline_probs) / n if n else 0.0
    ctr_span = Counter(str(r.get("tree_span_category") or "other") for r in classified)
    n_pos_leaf = sum(1 for r in classified if r.get("pos_leaf_only_match"))
    n_skipped = sum(1 for r in classified if r.get("eval_note") is not None)

    _, occ_ci_low, occ_ci_high = bootstrap_ci_proportion(one_node_flags, seed=_BOOTSTRAP_SEED)
    occ_p = p_value_vs_heterogeneous_baseline(one_node_flags, baseline_probs, seed=_BOOTSTRAP_SEED)

    summary_row: Dict[str, Any] = {
        "task": "1_1",
        "test": 1,
        "model": raw_rows[0].get("model", "") if raw_rows else "",
        "provider": raw_rows[0].get("provider", "") if raw_rows else "",
        "n_trials": n,
        "baseline_constituent_rate": mean_bp,
        "n_eval_skipped": n_skipped,
        "p_eval_skipped": n_skipped / n if n else 0.0,
        "n_pos_leaf_only_matches": n_pos_leaf,
        "p_pos_leaf_only_matches": n_pos_leaf / n if n else 0.0,
        "single_constituent_ci_low": occ_ci_low,
        "single_constituent_ci_high": occ_ci_high,
        "single_constituent_p_vs_baseline": occ_p,
    }
    for cat in SPAN_CATEGORIES:
        summary_row[f"n_{cat}"] = ctr_span.get(cat, 0)
        summary_row[f"p_{cat}"] = ctr_span.get(cat, 0) / n if n else 0.0

    summary_row["counts_by_prompt_variant"] = dict(
        Counter(str(r.get("prompt_variant") or "") for r in classified)
    )

    payload = {
        "meta": {
            "task": "1_1",
            "test": 1,
            "run": run_name,
            "model": summary_row.get("model", ""),
            "provider": summary_row.get("provider", ""),
            "n_results": len(classified),
            "n_missing_dataset_trial": n_missing_ds,
        },
        "summary": summary_row,
        "results": classified,
    }

    out_path = paths.general_classified(1, run_name)
    atomic_write_json(out_path, payload)
    logger.info("[Test 1 | %s] wrote %s (%d rows)", run_name, out_path, n)
    return payload
