"""Text normalization and deleted-span extraction.

Merges the former ``nlp_utils.py`` (token normalization, purity checks) and
``matching.py`` (strict diff extraction of the deleted string/span) into one
module shared by all six tests.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)

EXTRACTION_OK_CHAR_FALLBACK = "OK_CHAR_FALLBACK"

SPECIAL_SYMBOL_CHARS = set("$%€£¥§©®™")
_DIGIT_RE = re.compile(r"[0-9]")
_PUNCT_TOKEN_RE = re.compile(r"^[^A-Za-z0-9]+$")

try:
    from nltk.tokenize import word_tokenize
except ImportError:  # pragma: no cover
    word_tokenize = None


# ---------------------------------------------------------------------------
# Token normalization
# ---------------------------------------------------------------------------
def normalize_tokens(text: str) -> List[str]:
    """Lowercase, strip non-alphanumeric characters, split on whitespace.

    This is the project-wide normalization rule; all strict evaluation
    comparisons must use it.
    """
    if not text:
        return []
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in cleaned.split() if t]


def normalize_joined(text: str) -> str:
    return " ".join(normalize_tokens(text))


def is_punctuation_only_token(tok: str) -> bool:
    t = tok.strip()
    return bool(t) and bool(_PUNCT_TOKEN_RE.match(t))


def normalize_token_list(tokens: Sequence[str]) -> List[str]:
    """Lowercase, strip, drop empties; trim leading/trailing punctuation-only tokens."""
    out: List[str] = []
    for t in tokens:
        t = t.strip().lower()
        if t:
            out.append(t)
    while out and is_punctuation_only_token(out[0]):
        out.pop(0)
    while out and is_punctuation_only_token(out[-1]):
        out.pop()
    return out


def strip_trailing_punct_chunk_tokens(tokens: Sequence[str]) -> List[str]:
    toks = list(tokens)
    while toks and is_punctuation_only_token(toks[-1]):
        toks.pop()
    return toks


def normalize_for_constituent_label(tokens: Sequence[str]) -> List[str]:
    """Normalization pipeline for constituent-equality checks (Test 1)."""
    return strip_trailing_punct_chunk_tokens(normalize_token_list(list(tokens)))


def token_has_digit(token: str) -> bool:
    return bool(_DIGIT_RE.search(token))


def token_has_special_symbol(token: str) -> bool:
    return any(ch in SPECIAL_SYMBOL_CHARS for ch in token)


def passes_purity(tokens: Sequence[str], pos_tags: Sequence[str]) -> bool:
    """Reject sentences containing CD tags, digits, or currency/special symbols."""
    if len(tokens) != len(pos_tags):
        raise ValueError("tokens and pos_tags must have the same length")
    for w, pos in zip(tokens, pos_tags):
        if pos.upper() == "CD":
            return False
        if not w.strip():
            continue
        if token_has_digit(w) or token_has_special_symbol(w):
            return False
    return True


def normalize_llm_output_for_diff(text: str) -> str:
    """Map curly quotes/dashes to ASCII so the diff is not derailed by them."""
    if not text:
        return text
    t = text
    for a, b in (
        ("\u201c", '"'), ("\u201d", '"'),
        ("\u2018", "'"), ("\u2019", "'"), ("\u2032", "'"), ("\u00b4", "'"),
        ("\u2013", "-"), ("\u2014", "-"),
    ):
        t = t.replace(a, b)
    return t


def tokenize_model_output(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    if word_tokenize is not None:
        try:
            return word_tokenize(text.strip())
        except LookupError:
            import nltk

            nltk.download("punkt")
            return word_tokenize(text.strip())
        except Exception as exc:  # pragma: no cover
            logger.debug("word_tokenize failed, fallback to whitespace split: %s", exc)
    return text.strip().split()


# ---------------------------------------------------------------------------
# Character-level diff helpers
# ---------------------------------------------------------------------------
def _char_delete_ranges(original: str, transformed: str) -> List[Tuple[int, int]]:
    """Half-open character intervals [i1, i2) removed/replaced on *original*."""
    sm = SequenceMatcher(None, original, transformed)
    return [(i1, i2) for tag, i1, i2, _j1, _j2 in sm.get_opcodes() if tag in ("delete", "replace") and i1 < i2]


def compute_token_char_spans(sentence: str, tokens: Sequence[str]) -> List[Tuple[int, int]] | None:
    """Map each token to [start, end) character offsets in ``sentence``.

    Returns None when ``sentence`` does not match the tokenization.
    """
    pos = 0
    spans: List[Tuple[int, int]] = []
    for tok in tokens:
        while pos < len(sentence) and sentence[pos].isspace():
            pos += 1
        if pos + len(tok) > len(sentence) or sentence[pos : pos + len(tok)] != tok:
            return None
        spans.append((pos, pos + len(tok)))
        pos += len(tok)
    while pos < len(sentence) and sentence[pos].isspace():
        pos += 1
    if pos != len(sentence):
        return None
    return spans


def inclusive_token_span_from_char_deletes(
    tokens_test: Sequence[str],
    original_sentence: str,
    model_edited_test: str,
) -> Tuple[int, int] | None:
    """Recover inclusive token indices [start, end] from a character-level diff."""
    spans = compute_token_char_spans(original_sentence, tokens_test)
    if not spans:
        return None
    ranges = _char_delete_ranges(original_sentence, model_edited_test)
    if not ranges:
        return None
    touched: set[int] = set()
    for ds, de in ranges:
        for ti, (ts, te) in enumerate(spans):
            if ts < de and te > ds:
                touched.add(ti)
    if not touched:
        return None
    return min(touched), max(touched)


def extract_deleted_string_char_level(original: str, transformed: str) -> str:
    """Concatenate all character-level delete/replace pieces of the diff."""
    sm = SequenceMatcher(None, original, transformed)
    parts: List[str] = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            piece = original[i1:i2].strip()
            if piece:
                parts.append(piece)
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Strict token-level deleted-span extraction (Tests 1-3)
# ---------------------------------------------------------------------------
def _is_subsequence(small: Sequence[str], big: Sequence[str]) -> bool:
    if not small:
        return True
    j = 0
    for b in big:
        if j < len(small) and small[j] == b:
            j += 1
    return j == len(small)


def _core_norm_stream(tokens: Sequence[str]) -> List[Tuple[str, int]]:
    """Normalized, tail-trimmed token stream mapped back to original indices."""
    stream: List[Tuple[str, int]] = []
    for i, t in enumerate(tokens):
        nt = t.strip().lower()
        if nt:
            stream.append((nt, i))
    while stream and is_punctuation_only_token(tokens[stream[0][1]]):
        stream.pop(0)
    while stream and is_punctuation_only_token(tokens[stream[-1][1]]):
        stream.pop()
    return stream


def extract_deleted_span(
    tokens_test: Sequence[str],
    model_edited_test: str,
) -> Tuple[str | None, Tuple[int, int] | None, str]:
    """Returns (deleted_string, inclusive (start, end) on ``tokens_test``, status)."""
    orig = list(tokens_test)
    if not orig:
        return None, None, "EMPTY"

    out_toks = tokenize_model_output(model_edited_test)
    o_stream = _core_norm_stream(orig)
    if not o_stream:
        return None, None, "EMPTY"

    o_norm = [x[0] for x in o_stream]
    o_map = [x[1] for x in o_stream]
    m_norm = normalize_token_list(out_toks)

    if not m_norm:
        start_orig, end_orig = o_map[0], o_map[-1]
        deleted_tokens = orig[start_orig : end_orig + 1]
        return " ".join(deleted_tokens).strip(), (start_orig, end_orig), "OK"

    if o_norm == m_norm:
        return None, None, "IDENTICAL"
    if len(m_norm) > len(o_norm):
        return None, None, "ADDED_TOKENS"

    i = k = 0
    while i < len(o_norm) and k < len(m_norm) and o_norm[i] == m_norm[k]:
        i += 1
        k += 1
    i_end, k_end = len(o_norm) - 1, len(m_norm) - 1
    while i_end >= i and k_end >= k and o_norm[i_end] == m_norm[k_end]:
        i_end -= 1
        k_end -= 1

    if k <= k_end:
        return None, None, "REORDERED"
    if i > i_end:
        return None, None, "IDENTICAL"

    rebuilt = o_norm[:i] + o_norm[i_end + 1 :]
    if rebuilt != m_norm:
        if not _is_subsequence(m_norm, o_norm):
            return None, None, "ADDED_TOKENS"
        return None, None, "REORDERED"

    start_orig, end_orig = o_map[i], o_map[i_end]
    deleted_tokens = orig[start_orig : end_orig + 1]
    return " ".join(deleted_tokens).strip(), (start_orig, end_orig), "OK"


def parse_first_line_model_output(raw: str) -> str:
    """First line of a model response, with symmetric surrounding quotes removed."""
    if not raw:
        return ""
    first = raw.strip().splitlines()[0] if raw.strip() else ""
    first = first.strip()
    if len(first) >= 2 and first[0] in "\"'" and first[-1] == first[0]:
        first = first[1:-1].strip()
    return first
