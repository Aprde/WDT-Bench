"""CoNLL-2000 sentence reading and chunk-span extraction."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from .text_utils import is_punctuation_only_token

Row = Tuple[str, str, str]  # (word, POS, chunk tag)


def read_conll_sentences(path: Path) -> List[List[Row]]:
    """Read a CoNLL-2000 flat text file into a list of sentences."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    sentences: List[List[Row]] = []
    for block in text.strip().split("\n\n"):
        rows: List[Row] = []
        for line in block.splitlines():
            parts = line.strip().split()
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
        if rows:
            sentences.append(rows)
    return sentences


def sentence_tokens(rows: Sequence[Row]) -> Tuple[List[str], List[str]]:
    return [r[0] for r in rows], [r[1] for r in rows]


def chunk_spans_ordered(rows: Sequence[Row]) -> List[Tuple[int, int, str]]:
    """Extract contiguous chunks from IOB tags; ``O`` tags are skipped."""
    spans: List[Tuple[int, int, str]] = []
    i, n = 0, len(rows)
    while i < n:
        tag = rows[i][2].strip()
        if tag == "O" or "-" not in tag:
            i += 1
            continue
        bio, ctype = tag.split("-", 1)
        if bio != "B":
            i += 1
            continue
        start = i
        i += 1
        while i < n:
            t = rows[i][2].strip()
            if t == "O" or t.startswith("B-"):
                break
            if t.startswith("I-"):
                i += 1
                continue
            break
        spans.append((start, i - 1, ctype))
    return spans


def get_chunk_type_sequence(rows: Sequence[Row]) -> List[str]:
    return [ctype for _, _, ctype in chunk_spans_ordered(rows)]


def is_punctuation_only_chunk(tokens: Sequence[str]) -> bool:
    if not tokens:
        return True
    return all(is_punctuation_only_token(t) for t in tokens)


def chunks_as_records(rows: Sequence[Row]) -> List[dict]:
    """Build chunk dicts with token spans for downstream tasks."""
    toks, _ = sentence_tokens(rows)
    out: List[dict] = []
    for start, end, ctype in chunk_spans_ordered(rows):
        out.append({"type": ctype, "start": start, "end": end, "tokens": list(toks[start : end + 1])})
    return out


def sentence_join(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def chunk_sequence_non_punct(rows: Sequence[Row]) -> List[str]:
    """Chunk type sequence excluding chunks whose tokens are all punctuation."""
    return [c["type"] for c in chunks_as_records(rows) if not is_punctuation_only_chunk(c["tokens"])]
