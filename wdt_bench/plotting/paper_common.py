"""Shared loaders and style constants for the paper figures.

All paths come from :mod:`wdt_bench.paths`.  Loaders group rows by the
row-level ``model`` field, so figures stay correct even if classified files
are copied between run folders, and they degrade gracefully when some runs
are not present in the repository.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

import matplotlib.pyplot as plt

from .. import paths
from ..io_utils import load_json

logger = logging.getLogger(__name__)

# flash -> plus -> max  ==  increasing model size
MODEL_ORDER = ["qwen-flash", "qwen-plus", "qwen-max"]
MODEL_LABELS = {
    "qwen-flash": "Qwen Flash",
    "qwen-plus": "Qwen Plus",
    "qwen-max": "Qwen Max",
}

MODEL_COLORS = {
    "qwen-flash": "#B6B3D6",  # purple
    "qwen-plus": "#F8B2A2",   # salmon
    "qwen-max": "#E9687A",    # pink-red
}
MODEL_COLORS_DARK = {
    "qwen-flash": "#8C88B8",
    "qwen-plus": "#E58A72",
    "qwen-max": "#D44A60",
}

# Scatter-cloud-only colours for parent/node plots: deliberately spread
# across the colour wheel so clouds stay distinguishable at low alpha.
MODEL_COLORS_SCATTER = {
    "qwen-flash": "#7268B0",  # blue-violet (~270 deg)
    "qwen-plus": "#C8A830",   # amber (~50 deg)
    "qwen-max": "#C03060",    # crimson (~340 deg)
}

CAT11_ORDER = [
    "single_constituent",
    "multiple_constituents",
    "partial_constituent",
    "constituent_plus_partial",
    "other",
]
CAT11_LABELS = {
    "single_constituent": "One complete node",
    "multiple_constituents": "Multiple complete nodes",
    "partial_constituent": "Partial inside one node",
    "constituent_plus_partial": "Complete + other incomplete",
    "other": "Other",
}
CAT11_COLORS = {
    "single_constituent": "#B6B3D6",                    # purple
    "multiple_constituents": "#D5D3DE",              # pale purple-gray
    "partial_constituent": "#F6DFD6",              # cream
    "constituent_plus_partial": "#F8B2A2",  # salmon
    "other": "#E9687A",                                # pink-red
}

MEAN_DIAMOND_HALO_SIZE = 170
MEAN_DIAMOND_CORE_SIZE = 105
MEAN_DIAMOND_EDGE_WIDTH = 0.9


def apply_font() -> None:
    """Serif font used by the final paper figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Palatino Linotype", "Georgia", "Garamond",
            "Times New Roman", "DejaVu Serif",
        ],
    })


# -- Loaders -------------------------------------------------------------------


def _is_multi_demo_run(folder_name: str, rows: List[dict]) -> bool:
    """True for multi-demonstration runs (row ``n_demos`` > 1 or a
    ``_{N}demos`` folder suffix).

    Multi-demo rows carry the same ``model`` value as the standard run, so
    they must be kept out of the per-model pooling used by the standard
    (single-demonstration) panels.
    """
    nd = rows[0].get("n_demos") if rows else None
    if nd is not None:
        return int(nd) > 1
    return re.search(r"_\d+demos$", folder_name) is not None


def load_subtask(test_id: int) -> Dict[str, List[dict]]:
    """``{internal_model_name: [classified rows]}`` for Test 1, 2 or 3.

    Scans every run folder under ``results/processed/general_tests`` and
    groups rows by the row-level ``model`` field.  **Only standard
    single-demonstration runs are included**: the ``*_{N}demos`` runs share
    the same ``model`` value and would otherwise contaminate the per-model
    rates (keeping panel a consistent with the
    1-demonstration point of panel c).
    """
    out: Dict[str, List[dict]] = {}
    if not paths.PROCESSED_GENERAL.is_dir():
        return out
    for folder in sorted(p for p in paths.PROCESSED_GENERAL.iterdir() if p.is_dir()):
        path = folder / f"test{test_id}_classified.json"
        if not path.is_file():
            continue
        doc = load_json(path)
        rows = doc.get("results", []) if isinstance(doc, dict) else []
        if not rows or _is_multi_demo_run(folder.name, rows):
            continue
        model = rows[0].get("model")
        out.setdefault(model, []).extend(rows)
    return out


def _n_demos_of_run(folder_name: str, rows: List[dict]) -> int | None:
    """Number of demonstrations of a run (row field first, folder name second)."""
    nd = rows[0].get("n_demos") if rows else None
    if nd is not None:
        return int(nd)
    m = re.search(r"_(\d+)demos$", folder_name)
    if m:
        return int(m.group(1))
    return None


def load_test1_by_demos(model: str) -> Dict[int, List[dict]]:
    """``{n_demos: [rows]}`` for one internal model on Test 1.

    Single-demo runs come from folders without a ``demos`` suffix; the
    multi-demonstration runs from ``*_{N}demos`` folders.
    """
    out: Dict[int, List[dict]] = {}
    if not paths.PROCESSED_GENERAL.is_dir():
        return out

    for folder in sorted(p for p in paths.PROCESSED_GENERAL.iterdir() if p.is_dir()):
        path = folder / "test1_classified.json"
        if not path.is_file():
            continue
        rows = load_json(path).get("results", [])
        if not rows or rows[0].get("model") != model:
            continue
        if not _is_multi_demo_run(folder.name, rows):
            if 1 in out:
                logger.warning(
                    "Multiple single-demonstration folders found for model %s; "
                    "keeping the first and ignoring %s.", model, folder.name,
                )
            out.setdefault(1, rows)
        else:
            nd = _n_demos_of_run(folder.name, rows)
            if nd is not None:
                out[nd] = rows
    return out


def out_path(name: str):
    """Path of a paper figure under ``figures/paper``."""
    p = paths.FIGURES_PAPER / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
