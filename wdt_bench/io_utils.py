"""JSON I/O helpers.

All result files in this repository share one format::

    {"meta": {...}, "results": [ {...}, ... ]}

Writes are atomic (write to ``*.tmp``, then replace) so interrupted runs never
leave a half-written file behind.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_results(path: Path) -> tuple[list[dict], dict]:
    """Load a ``{"meta", "results"}`` document; a bare list is also accepted."""
    doc = load_json(path)
    if isinstance(doc, dict) and "results" in doc:
        rows = doc["results"]
        meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    elif isinstance(doc, list):
        rows, meta = doc, {}
    else:
        raise SystemExit(f"{path}: expected a {{meta, results}} object or a JSON list")
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: 'results' must be a list")
    return rows, meta


def _replace_file_robust(tmp: Path, target: Path, *, text: str, max_attempts: int = 10) -> None:
    """``os.replace`` with retries (Windows can transiently lock the target)."""
    last_exc: OSError | None = None
    for attempt in range(max_attempts):
        try:
            os.replace(str(tmp), str(target))
            return
        except OSError as exc:
            last_exc = exc
            if attempt + 1 < max_attempts:
                time.sleep(0.2 * (attempt + 1))
    try:  # last resort: write the target directly
        target.write_text(text, encoding="utf-8")
        tmp.unlink(missing_ok=True)
        return
    except OSError:
        pass
    raise last_exc if last_exc is not None else OSError(f"cannot write {target}")


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _replace_file_robust(tmp, path, text=text)


def atomic_write_results(path: Path, meta: dict, results: list[dict]) -> None:
    atomic_write_json(path, {"meta": dict(meta), "results": results})
