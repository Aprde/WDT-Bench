"""Quick span-level analysis for one diagnostic condition + model.

A lighter-weight companion to ``tests4_5``: each response is bucketed into
``constituent`` / ``nonconstituent`` / ``multi_span`` / ``identical`` /
``fail_to_follow`` (the response is not a subsequence of the stimulus) /
``no_tree``, and constituent deletions additionally report the tree path of
the deleted node (e.g. ``S-VP-PP``).  Useful for eyeballing a single run.

Output: ``results/processed/diagnostic_tests/span_analysis_{condition}__{model}.json``.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .. import paths
from ..io_utils import atomic_write_json, load_results
from ..trees import load_tree_map

logger = logging.getLogger(__name__)


def parse_tree(tree_str: str) -> Dict[Tuple[str, ...], str]:
    """Map each constituent's word tuple to its path from the root (``S-VP-PP``)."""
    constituents: Dict[Tuple[str, ...], str] = {}

    def extract_words(s: str) -> List[str]:
        words = re.findall(r"\([A-Z$\-\w]+\s+([^()]+)\)", s)
        return [w.strip() for w in words]

    def find_constituents(s: str, path: str) -> None:
        depth = 0
        start = None
        for i, ch in enumerate(s):
            if ch == "(":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and start is not None:
                    substr = s[start : i + 1]
                    label_match = re.match(r"\(([A-Z$\-\w]+)", substr)
                    label = label_match.group(1) if label_match else ""
                    base_label = label.split("-")[0]
                    current_path = f"{path}-{base_label}" if path else base_label
                    words = extract_words(substr)
                    # Preterminals like (NN dog) contain no nested brackets;
                    # only constituent-level groups are recorded, matching
                    # the single-constituent rule used elsewhere.
                    is_preterminal_group = "(" not in substr[1:]
                    if words and not is_preterminal_group:
                        key = tuple(words)
                        if key not in constituents:
                            constituents[key] = current_path
                    inner = substr[1:]
                    first_space = inner.find(" ")
                    if first_space != -1:
                        find_constituents(inner[first_space + 1 : -1], current_path)

    find_constituents(tree_str, "")
    return constituents


def is_subset_sequence(response_words: List[str], sentence_words: List[str]) -> bool:
    it = iter(sentence_words)
    return all(w in it for w in response_words)


def get_deleted_spans(sentence_words: List[str], response_words: List[str]) -> List[int]:
    resp_idx = 0
    deleted = []
    for i, w in enumerate(sentence_words):
        if resp_idx < len(response_words) and w == response_words[resp_idx]:
            resp_idx += 1
        else:
            deleted.append(i)
    return deleted


def is_contiguous(indices: List[int]) -> bool:
    if not indices:
        return True
    return indices[-1] - indices[0] == len(indices) - 1


def tokenize_chinese(text: str) -> List[str]:
    """Split Chinese text into characters (spaces removed)."""
    return list(text.replace(" ", ""))


def analyze_chinese(sentence: str, response: str, tree_str: Optional[str]) -> Tuple[str, str, str]:
    sent_chars = tokenize_chinese(sentence)
    resp_chars = tokenize_chinese(response)

    if not all(c in sentence for c in response):
        if response not in sentence and not is_subset_sequence(resp_chars, sent_chars):
            return "fail_to_follow", "", "none"

    if response == sentence:
        return "identical", "", "none"

    if response in sentence:
        deleted_str = sentence.replace(response, "", 1)
        start_idx = sentence.find(response)
        if start_idx == 0:
            deleted_str = sentence[len(response):]
        elif start_idx + len(response) == len(sentence):
            deleted_str = sentence[:start_idx]
        else:
            return "multi_span", deleted_str, "none"
    else:
        deleted_indices = get_deleted_spans(sent_chars, resp_chars)
        if not deleted_indices:
            return "identical", "", "none"
        deleted_str = "".join(sent_chars[i] for i in deleted_indices)
        if not is_contiguous(deleted_indices):
            return "multi_span", deleted_str, "none"

    if tree_str is None:
        return "no_tree", deleted_str, "none"

    constituents = parse_tree(tree_str)
    deleted_tuple = tuple(tokenize_chinese(deleted_str))

    for span, path in constituents.items():
        if span == deleted_tuple:
            return "constituent", deleted_str, path

    joined = {"".join(span): path for span, path in constituents.items()}
    if deleted_str in joined:
        return "constituent", deleted_str, joined[deleted_str]

    return "nonconstituent", deleted_str, "none"


def analyze_english(sentence: str, response: str, tree_str: Optional[str]) -> Tuple[str, str, str]:
    sent_words = sentence.lower().split()
    resp_words = response.lower().split()

    if not is_subset_sequence(resp_words, sent_words):
        return "fail_to_follow", "", "none"

    deleted_indices = get_deleted_spans(sent_words, resp_words)
    if not deleted_indices:
        return "identical", "", "none"

    deleted_words = [sent_words[i] for i in deleted_indices]
    if not is_contiguous(deleted_indices):
        return "multi_span", " ".join(deleted_words), "none"

    if tree_str is None:
        return "no_tree", " ".join(deleted_words), "none"

    constituents = parse_tree(tree_str)
    constituents_lower = {tuple(w.lower() for w in k): v for k, v in constituents.items()}

    deleted_span = tuple(deleted_words)
    if deleted_span in constituents_lower:
        return "constituent", " ".join(deleted_words), constituents_lower[deleted_span]

    return "nonconstituent", " ".join(deleted_words), "none"


def run(condition: str, model: str) -> Dict[str, Any]:
    """Analyse one ``{condition}__{model}`` raw file and write a summary JSON."""
    if condition not in paths.DIAGNOSTIC_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    raw_path = paths.diagnostic_raw(condition, model)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw results not found: {raw_path}")

    test_type = condition.rsplit("_", 1)[0] if condition.startswith(("parallel", "nonsense")) else None
    tree_map: Dict[str, str] = {}
    if test_type:
        tree_path = paths.diagnostic_trees(test_type)
        if tree_path.is_file():
            tree_map = load_tree_map(tree_path)

    language = "chinese" if condition.endswith("_chinese") else "english"
    rows, _meta = load_results(raw_path)

    results: List[Dict[str, Any]] = []
    for row in rows:
        sentence = str(row.get("sentence", "")).strip()
        response = str(row.get("response", "")).strip()
        response = response.strip("'\"\u2018\u2019\u201c\u201d")
        response = re.sub(r"[.\u3002\uff0c,!\uff01?\uff1f]+$", "", response).strip()

        tree_str = tree_map.get(sentence) or tree_map.get(sentence.lower())

        if language == "chinese":
            category, deleted, node_label = analyze_chinese(sentence, response, tree_str)
        else:
            category, deleted, node_label = analyze_english(sentence, response, tree_str)

        results.append({
            "trial": row.get("trial"),
            "sentence": sentence,
            "response": response,
            "category": category,
            "deleted_words": deleted,
            "node_label": node_label,
        })

    total = len(results)
    cat_counts = Counter(r["category"] for r in results)
    label_counts = Counter(
        r["node_label"] for r in results if r["category"] == "constituent"
    )

    logger.info("=== Category distribution (%s | %s) ===", condition, model)
    for cat, cnt in cat_counts.most_common():
        logger.info("  %s: %d (%.1f%%)", cat, cnt, 100 * cnt / total if total else 0.0)
    if label_counts:
        logger.info("=== Constituent node labels ===")
        for label, cnt in label_counts.most_common():
            logger.info("  %s: %d", label, cnt)

    payload = {
        "meta": {
            "condition": condition,
            "model": model,
            "language": language,
            "n_results": total,
        },
        "category_counts": dict(cat_counts),
        "category_rates": {k: v / total for k, v in cat_counts.items()} if total else {},
        "constituent_node_label_counts": dict(label_counts),
        "results": results,
    }

    out_path = paths.PROCESSED_DIAGNOSTIC / f"span_analysis_{condition}__{model}.json"
    atomic_write_json(out_path, payload)
    logger.info("Wrote %s", out_path)
    return payload
