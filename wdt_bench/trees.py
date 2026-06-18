"""Constituency-tree utilities.

- :func:`load_tree_map` reads the JSON tree files for the diagnostic tests.
- :func:`classify_leaf_span` classifies a contiguous leaf span against a PTB
  tree (Test 1 fine-grained categories).

Categories returned by :func:`classify_leaf_span`:

- ``single_constituent``: the span is the exact yield of one subtree.
- ``multiple_constituents``: the span tiles into >= 2 phrasal subtrees.
- ``partial_constituent``: strict subset of one subtree's yield without a
  full phrasal node inside.
- ``constituent_plus_partial``: contains the yield of at least one
  phrasal subtree as a proper sub-interval plus extra material.
- ``other``: cannot be classified.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, FrozenSet, Optional, Set, Tuple

from nltk.tree import Tree

from .io_utils import load_json
from .text_utils import normalize_tokens

LeafSpan = Tuple[int, int]


def load_tree_map(path: Path) -> Dict[str, str]:
    """Read a JSON list of ``{"sentence", "tree"}`` into a lookup dict.

    Each sentence is stored both verbatim and lower-cased so loose lookups by
    surface form succeed.
    """
    result: Dict[str, str] = {}
    for rec in load_json(path):
        sent = (rec.get("sentence") or "").strip()
        tree = (rec.get("tree") or "").strip()
        if sent and tree:
            result[sent] = tree
            result[sent.lower()] = tree
    return result


def tree_leaf_span_in_root(root: Tree, node: Tree) -> Optional[LeafSpan]:
    """Inclusive leaf indices of ``node`` within ``root``."""
    all_leaves = root.leaves()
    part = node.leaves()
    if not part:
        return None
    plen = len(part)
    for i in range(0, len(all_leaves) - plen + 1):
        if all_leaves[i : i + plen] == part:
            return i, i + plen - 1
    return None


def _all_subtree_spans(root: Tree) -> Set[LeafSpan]:
    spans: Set[LeafSpan] = set()
    for st in root.subtrees():
        if isinstance(st, Tree):
            sp = tree_leaf_span_in_root(root, st)
            if sp:
                spans.add(sp)
    return spans


def is_preterminal(st: Tree) -> bool:
    """True for POS-level nodes, whose children are all bare leaves."""
    return not any(isinstance(ch, Tree) for ch in st)


def _constituent_subtree_spans(root: Tree) -> Set[LeafSpan]:
    """Leaf spans of constituent-level (non-preterminal) subtrees.

    These are the spans that may count as ``single_constituent``: POS
    preterminals never qualify; one-word phrasal projections (e.g.
    ``(NP (PRP it))``) do.
    """
    out: Set[LeafSpan] = set()
    for st in root.subtrees():
        if not isinstance(st, Tree) or is_preterminal(st):
            continue
        sp = tree_leaf_span_in_root(root, st)
        if sp:
            out.add(sp)
    return out


def is_pos_leaf_only_span(root: Tree, lo: int, hi: int) -> bool:
    """True when ``[lo, hi]`` matches only a POS leaf, not a constituent.

    Single-token spans whose only exact match in the tree is a preterminal
    are not legitimate constituent deletions.
    """
    if lo != hi:
        return False
    return (lo, hi) not in _constituent_subtree_spans(root)


def _phrasal_subtree_spans(root: Tree, *, min_leaves: int = 2) -> Set[LeafSpan]:
    out: Set[LeafSpan] = set()
    for st in root.subtrees():
        if isinstance(st, Tree) and len(st.leaves()) >= min_leaves:
            sp = tree_leaf_span_in_root(root, st)
            if sp:
                out.add(sp)
    return out


def _min_phrasal_tiles(lo: int, hi: int, spans_ph: FrozenSet[LeafSpan]) -> int:
    """Minimum number of phrasal spans tiling [lo, hi]; INF when impossible."""
    INF = 10**9
    memo: dict[int, int] = {}

    def dp(a: int) -> int:
        if a > hi:
            return 0
        if a in memo:
            return memo[a]
        best = INF
        for s, e in spans_ph:
            if s == a and e <= hi:
                rest = dp(e + 1)
                if rest < INF:
                    best = min(best, 1 + rest)
        memo[a] = best
        return best

    return dp(lo)


def classify_leaf_span(root: Tree, lo: int, hi: int, *, min_leaves_phrasal: int = 2) -> str:
    if lo > hi or lo < 0:
        return "other"

    spans_const = _constituent_subtree_spans(root)
    spans_ph = _phrasal_subtree_spans(root, min_leaves=min_leaves_phrasal)

    # Only constituent-level (non-preterminal) nodes qualify: a span that
    # coincides with nothing but a POS leaf such as (DT the) or (JJ old) is
    # NOT a complete node and falls through to the partial/other buckets.
    if (lo, hi) in spans_const:
        return "single_constituent"

    mt = _min_phrasal_tiles(lo, hi, frozenset(spans_ph))
    if 2 <= mt < 10**9:
        return "multiple_constituents"

    # Strictly contains a phrasal subtree yield as a proper sub-interval.
    for a, b in spans_ph:
        if lo <= a and b <= hi and (lo < a or b < hi):
            return "constituent_plus_partial"

    # Strict subset of one subtree yield, without a phrasal node inside.
    for st in root.subtrees():
        if not isinstance(st, Tree):
            continue
        sp = tree_leaf_span_in_root(root, st)
        if not sp:
            continue
        L, R = sp
        if L <= lo and hi <= R and (L < lo or hi < R):
            return "partial_constituent"

    return "other"


def _flatten_leaf_str(x: object) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, tuple) and x:
        return str(x[0])
    return str(x)


def deleted_tokens_to_leaf_span(root: Tree, deleted_tokens: list[str]) -> Optional[LeafSpan]:
    """First contiguous leaf span whose normalized tokens equal ``deleted_tokens``."""
    if not deleted_tokens:
        return None
    flat: list[str] = []
    for leaf in root.leaves():
        flat.extend(normalize_tokens(_flatten_leaf_str(leaf)))
    n, m = len(flat), len(deleted_tokens)
    if m > n or m == 0:
        return None
    for i in range(0, n - m + 1):
        if flat[i : i + m] == deleted_tokens:
            return i, i + m - 1
    return None


def baseline_constituent_probability_from_tree(root: Tree) -> float:
    """Probability that a uniformly-random contiguous span (length 2..n-1)
    is a single constituent in *root* -- the random-deletion baseline.

    Uses the same criterion as :func:`classify_leaf_span`: the span must
    match a non-preterminal subtree (POS leaves excluded, one-word phrasal
    projections included).
    """
    n = len(root.leaves())
    if n < 3:
        return 0.0
    const_spans = _constituent_subtree_spans(root)
    total = hits = 0
    for length in range(2, n):
        for start in range(0, n - length):
            end = start + length - 1
            total += 1
            if (start, end) in const_spans:
                hits += 1
    return hits / total if total else 0.0


