"""Shared helpers for the diagnostic-test analyses (Tests 4-6).

The only storage-format-related behaviour to note is
that raw results are now read from the unified JSON files in
``results/raw/diagnostic_tests`` instead of per-trial TSV files.
"""
from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Tuple

from .. import paths
from ..io_utils import load_results
from ..trees import is_preterminal

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def _is_cjk(text: str) -> bool:
    """True when more than a quarter of the characters are CJK."""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cjk > len(text) * 0.25


def _normalize_text(s: str) -> str:
    """Strip whitespace and trailing sentence punctuation (EN + ZH)."""
    return s.strip().rstrip(".,!?\u3002\uff0c\uff01\uff1f")


def _norm_compare(s: str) -> str:
    """Casefolded, punctuation- and whitespace-normalised comparison key."""
    s = s.lower().strip().rstrip(".,!?\u3002\uff0c\uff01\uff1f")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def flatten_response(text: str) -> str:
    """Flatten a (possibly multi-line) model response to single-line form.

    Control characters are replaced by their two-character escape sequences
    (newline -> ``\\n`` etc.), mirroring the single-line storage format of
    the experiment logs: a multi-line response counts as a
    failed/over-generated answer rather than being silently truncated.
    """
    return text.replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")


# ---------------------------------------------------------------------------
# Deletion extraction (diff between stimulus and model response)
# ---------------------------------------------------------------------------


def extract_deleted(sentence: str, response: str) -> str:
    """Return the words of ``sentence`` that are absent from ``response``.

    Strategy: prefix match first (fast path for the common "response is a
    prefix of the sentence" case), then a token-level diff.  Multiple deleted
    spans are joined with ``" ... "``; an empty string means nothing was
    deleted (or the response could not be aligned).
    """
    s = _normalize_text(sentence)
    r = _normalize_text(response)

    if not r:
        return s

    s_lo = s.lower()
    r_lo = r.lower()
    if s_lo.startswith(r_lo):
        remaining = s[len(r):].strip()
        return remaining

    s_toks = s.split()
    r_toks = r.split()
    sm = difflib.SequenceMatcher(
        None,
        [t.lower() for t in s_toks],
        [t.lower() for t in r_toks],
        autojunk=False,
    )
    deleted_spans: List[Tuple[int, int]] = []
    for op, i1, i2, _j1, _j2 in sm.get_opcodes():
        if op == "delete":
            deleted_spans.append((i1, i2))
    if len(deleted_spans) == 1:
        i1, i2 = deleted_spans[0]
        return " ".join(s_toks[i1:i2])
    if len(deleted_spans) > 1:
        parts = [" ".join(s_toks[i1:i2]) for i1, i2 in deleted_spans]
        return " ... ".join(parts)

    return ""


# ---------------------------------------------------------------------------
# Tree-based labels for Tests 4-5 (node rule vs parent rule)
# ---------------------------------------------------------------------------


def _find_pp_np_leaves(tree_str: str) -> Tuple[List[str], List[str]]:
    """Locate the first PP inside the (first) VP and the NP inside that PP.

    Returns ``(pp_leaves, np_leaves)``; both empty when the parse fails or
    the VP contains no PP.
    """
    try:
        from nltk.tree import Tree

        tree = Tree.fromstring(tree_str)
    except Exception:
        return [], []

    def _search_vp(subtree) -> Tuple[List[str], List[str]]:
        if not hasattr(subtree, "label"):
            return [], []
        label = subtree.label()

        if label == "VP":
            for child in subtree:
                if not hasattr(child, "label"):
                    continue
                if child.label().startswith("PP"):
                    pp_leaves = child.leaves()
                    for pp_child in child:
                        if hasattr(pp_child, "label") and pp_child.label().startswith("NP"):
                            return pp_leaves, pp_child.leaves()
                    return pp_leaves, []
            return [], []

        for child in subtree:
            result = _search_vp(child)
            if result[0]:
                return result
        return [], []

    return _search_vp(tree)


# Chinese localiser suffixes that may be dropped together with the NP.
_ZH_LOC = "\u4e0a\u4e2d\u91cc\u65c1\u5185\u5916\u4e0b\u524d\u540e\u5904\u8fb9"


def classify_node_parent_from_tree(
    sentence: str,
    deleted_str: str,
    tree_map: Dict[str, str],
) -> str:
    """Label a deletion as ``node_rule`` (the NP inside the PP -- same
    category as the demonstration's deleted NP) or ``parent_rule`` (the whole
    PP -- the parent constituent), plus ``no_deletion`` / ``other``, using
    the constituency tree."""
    d = _normalize_text(deleted_str)
    if not d:
        return "no_deletion"

    tree_str = (
        tree_map.get(sentence)
        or tree_map.get(sentence.lower())
        or tree_map.get(_normalize_text(sentence))
        or tree_map.get(_normalize_text(sentence).lower())
    )
    if not tree_str:
        return "other"

    pp_leaves, np_leaves = _find_pp_np_leaves(tree_str)
    if not pp_leaves:
        return "other"

    is_cjk_sent = _is_cjk(sentence)
    if is_cjk_sent:
        pp_str = "".join(pp_leaves)
        np_str = "".join(np_leaves)
    else:
        pp_str = " ".join(pp_leaves)
        np_str = " ".join(np_leaves)

    d_cmp = _norm_compare(d)
    pp_cmp = _norm_compare(pp_str)
    np_cmp = _norm_compare(np_str)

    if np_cmp and d_cmp == np_cmp:
        return "node_rule"
    if pp_cmp and d_cmp == pp_cmp:
        return "parent_rule"

    # Chinese: tolerate a dropped/kept localiser suffix on either side.
    if is_cjk_sent and np_cmp:
        np_no_loc = np_cmp.rstrip(_ZH_LOC)
        d_no_loc = d_cmp.rstrip(_ZH_LOC)
        if d_cmp == np_no_loc or d_no_loc == np_cmp or d_no_loc == np_no_loc:
            return "node_rule"
        pp_no_loc = pp_cmp.rstrip(_ZH_LOC)
        if d_cmp == pp_no_loc or d_no_loc == pp_cmp or d_no_loc == pp_no_loc:
            return "parent_rule"

    return "other"


def is_single_constituent_from_tree(
    sentence: str,
    deleted_str: str,
    tree_map: Dict[str, str],
) -> bool:
    """True when the deleted string is the exact yield of one constituent.

    Only constituent-level (non-preterminal) subtrees qualify: bare POS
    leaves such as ``(NN dog)`` do not count, while one-word phrasal
    projections such as ``(NP (PRP it))`` do -- the same rule as Test 1.
    This is intentionally broader than the task-specific node/parent labels:
    a deletion can be a single constituent without being the target PP or the
    target NP-inside-PP used for the node/parent-rule contrast.
    """
    d = _normalize_text(deleted_str)
    if not d:
        return False

    tree_str = (
        tree_map.get(sentence)
        or tree_map.get(sentence.lower())
        or tree_map.get(_normalize_text(sentence))
        or tree_map.get(_normalize_text(sentence).lower())
    )
    if not tree_str:
        return False

    try:
        from nltk.tree import Tree

        tree = Tree.fromstring(tree_str)
    except Exception:
        return False

    is_cjk_sent = _is_cjk(sentence)
    d_cmp = _norm_compare(d)
    for subtree in tree.subtrees():
        # ``subtrees()`` also yields POS preterminals like (NN dog), whose
        # single-leaf yield would otherwise let every deleted word match.
        if is_preterminal(subtree):
            continue
        leaves = [str(x) for x in subtree.leaves()]
        if not leaves:
            continue
        span_str = "".join(leaves) if is_cjk_sent else " ".join(leaves)
        if _norm_compare(span_str) == d_cmp:
            return True
    return False


# ---------------------------------------------------------------------------
# Structure types for Test 6 (PP-attachment ambiguity)
# ---------------------------------------------------------------------------
# Structure 1: the PP attaches to the object NP (NP-attachment reading).
# Structure 2: the PP attaches to the VP (instrument/comitative reading).

_PP_STRUCT1_LOWER = frozenset([
    "bill cut the paper with a very bright colour",
    "bruce likes the rock band with different instruments",
    "charles runs a photo studio with two floors",
    "dan entered the room with a window",
    "jasmine sang the song with romantic lyrics",
    "michelle watched the movie with english subtitles",
    "mom ate the chocolate with chopped nuts",
    "queen elizabeth ii ate the cake with a red rose",
    "the basketball player won the game with lots of audience",
    "the coach led the team with three members",
    "the comedian saw the crowd with fresh flowers",
    "the doctor noticed the boy using big headphones",
    "the guy caught the rat with a scar",
    "the magician thanked the participant with a big nose",
    "the police investigated the house with a garden",
    "the professor recruited the students with black eyes",
    "van gogh drew the paintings with starry night",
    "the cook ate the meal with some cauliflowers",
])

_PP_STRUCT2_LOWER = frozenset([
    "bill cut the paper with a very sharp knife",
    "bruce leads the rock band with his voice",
    "charles runs a photo studio with two friends",
    "dan entered the room with a key",
    "jasmine sang the song with the lovely students",
    "michelle watched the movie with her boyfriend",
    "mom melted the chocolate with an iron pot",
    "queen elizabeth ii ate the cake with a silver fork",
    "the basketball player won the game with his team",
    "the coach motivated the team with inspiring words",
    "the comedian entertained the crowd with witty jokes",
    "the doctor saved the boy using a pacemaker",
    "the guy caught the rat with a trap",
    "the magician amazed the participant with a fancy scene",
    "the police investigated the house with the permission",
    "the professor challenged the students with his projects",
    "van gogh sold the paintings with a broken heart",
    "the cook made the meal with a chinese cleaver",
])


def get_structure_type_pp(sentence: str) -> int:
    """1 = NP-attachment, 2 = VP-attachment, 0 = unknown sentence."""
    key = _normalize_text(sentence).lower()
    if key in _PP_STRUCT1_LOWER:
        return 1
    if key in _PP_STRUCT2_LOWER:
        return 2
    return 0


# ---------------------------------------------------------------------------
# Structure types for Test 6 (adjunct / relative-clause ambiguity)
# ---------------------------------------------------------------------------
# Structure 1: the relative clause is semantically compatible with NP2 (low
# attachment); structure 2: compatible with NP1 (high attachment).

_ADJ_STRUCT1_LOWER = frozenset([
    "the car of the driver that had the moustache was pretty cool",
    "the church of the bishop that had the funny eyebrows looked odd",
    "the drugs of the supplier that had a nasty effect hurt everyone",
    "the gang of the criminal that had a long scar disappeared last monday",
    "the gold of the miner that had the impurities was worthless",
    "the house of the painter that had the small windows looked odd",
    "the letter of the writer that had blonde hair arrived this morning",
    "the machine of the inventor that had the goatee was amazing",
    "the restaurant of the chef that had the blue tiles pleased us",
    "the song of the singer that had long eyelashes was very smart",
    "the thesis of the editor that had the big nose made a lot of sense",
])

_ADJ_STRUCT2_LOWER = frozenset([
    "the bishop of the church that had the funny eyebrows looked odd",
    "the chef of the restaurant that had the blue tiles pleased us",
    "the criminal of the gang that had a long scar disappeared last monday",
    "the driver of the car that had the moustache was pretty cool",
    "the editor of the thesis that had the big nose made a lot of sense",
    "the flowers of the valley that had the old castle excited the tourists",
    "the inventor of the machine that had the goatee was amazing",
    "the miner of the gold that had the impurities was worthless",
    "the painter of the house that had the small windows looked odd",
    "the singer of the song that had long eyelashes was very smart",
    "the supplier of the drugs that had a nasty effect hurt everyone",
    "the valley of the flowers that had the old castle excited the tourists",
    "the writer of the letter that had blonde hair arrived this morning",
])


def get_structure_type_adjunct(sentence: str) -> int:
    """1 = low-attachment biased, 2 = high-attachment biased, 0 = unknown."""
    key = _normalize_text(sentence).lower()
    if key in _ADJ_STRUCT1_LOWER:
        return 1
    if key in _ADJ_STRUCT2_LOWER:
        return 2
    return 0


# ---------------------------------------------------------------------------
# Rule-based labels for Test 6
# ---------------------------------------------------------------------------


def classify_ambiguity_pp(sentence: str, deleted_str: str) -> str:
    """Label a deletion in a PP-attachment-ambiguous sentence.

    Labels: ``pp_np_only`` (NP inside the PP), ``pp_full_only`` (whole PP),
    ``np2_pp`` (object NP + PP), ``np2_only`` (object NP), ``no_deletion``,
    ``other``.
    """
    d = _normalize_text(deleted_str)
    if not d:
        return "no_deletion"

    s = _normalize_text(sentence)
    parts = s.rsplit(" with ", 1)

    if len(parts) != 2:
        parts = s.rsplit(" using ", 1)
        if len(parts) != 2:
            return "other"
        connector = " using "
    else:
        connector = " with "

    before_with = parts[0]
    pp_np = parts[1]
    pp_full = connector.strip() + " " + pp_np

    d_lo = d.lower()
    pp_np_lo = pp_np.lower()
    pp_full_lo = pp_full.lower()

    if d_lo == pp_np_lo:
        return "pp_np_only"
    if d_lo == pp_full_lo:
        return "pp_full_only"

    before_lo = before_with.lower()
    if d_lo.endswith(" " + pp_full_lo) or d_lo == pp_full_lo:
        return "np2_pp"
    if d_lo.endswith(" " + pp_np_lo) and connector.strip() not in d_lo:
        return "pp_np_only"

    connector_word = connector.strip()
    if connector_word not in d_lo:
        if before_lo.endswith(" " + d_lo) or before_lo.endswith(d_lo):
            return "np2_only"
        if d_lo and before_lo.endswith(d_lo.split()[-1]):
            return "np2_only"

    return "other"


def classify_ambiguity_adjunct(sentence: str, deleted_str: str) -> str:
    """Label a deletion in an adjunct-ambiguous sentence.

    Labels: ``rc_only`` (relative clause), ``np2_rc`` (NP2 + relative
    clause), ``np2_only`` (NP2), ``no_deletion``, ``other``.
    """
    d = _normalize_text(deleted_str)
    if not d:
        return "no_deletion"

    d_lo = d.lower()
    s = _normalize_text(sentence)
    s_lo = s.lower()

    of_idx = s_lo.find(" of ")
    if of_idx == -1:
        return "other"
    after_of = s[of_idx + 4:]
    after_of_lo = after_of.lower()

    that_idx = after_of_lo.find(" that ")
    if that_idx == -1:
        if d_lo in after_of_lo or after_of_lo.startswith(d_lo):
            return "np2_only"
        return "other"

    np2 = after_of[:that_idx]
    np2_lo = np2.lower().strip()

    if d_lo.startswith("that ") or d_lo == "that":
        return "rc_only"
    if d_lo.startswith(np2_lo) and "that" in d_lo:
        return "np2_rc"
    if d_lo == np2_lo or (after_of_lo.startswith(d_lo) and "that" not in d_lo):
        return "np2_only"
    if "that" in d_lo:
        return "np2_rc"
    return "other"


# ---------------------------------------------------------------------------
# Raw-result loading
# ---------------------------------------------------------------------------


def load_condition_rows(condition: str) -> List[Dict[str, Any]]:
    """Load every model's raw results for one condition.

    Files are read in sorted filename order (i.e. alphabetical by model name,
    a deterministic ordering), and rows keep the on-disk order
    (trial ascending, then within-trial order).  Each row gains ``model``,
    ``condition`` and ``source`` fields.
    """
    rows: List[Dict[str, Any]] = []
    pattern = f"{condition}__*.json"
    for path in sorted(paths.RAW_DIAGNOSTIC.glob(pattern)):
        result_rows, meta = load_results(path)
        model = meta.get("model") or path.stem.split("__", 1)[1]
        for row in result_rows:
            out = dict(row)
            out["model"] = model
            out["condition"] = condition
            out["source"] = path.name
            rows.append(out)
    return rows
