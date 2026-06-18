"""Central registry of repository paths.

Every script and module resolves files through this module, so the layout is
defined in exactly one place:

    data/                         benchmark stimuli (versioned inputs)
      general_tests/              Tests 1-3 question sets + CoNLL-2000 sources
      diagnostic_tests/           Tests 4-6 stimuli (JSON)
    results/
      raw/                        unmodified model outputs (JSON)
      processed/                  classified outputs and summary metrics (JSON)
    figures/                      per-model figures and paper figures
"""
from __future__ import annotations

from pathlib import Path

# Repository root = parent of the ``wdt_bench`` package.
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
DATA_GENERAL = DATA / "general_tests"
DATA_DIAGNOSTIC = DATA / "diagnostic_tests"

RESULTS = ROOT / "results"
RAW_GENERAL = RESULTS / "raw" / "general_tests"
RAW_DIAGNOSTIC = RESULTS / "raw" / "diagnostic_tests"
PROCESSED_GENERAL = RESULTS / "processed" / "general_tests"
PROCESSED_DIAGNOSTIC = RESULTS / "processed" / "diagnostic_tests"

FIGURES = ROOT / "figures"
FIGURES_GENERAL = FIGURES / "general_tests"
FIGURES_PAPER = FIGURES / "paper"


# -- General tests (Tests 1-3) ------------------------------------------------
def general_questions(test_id: int) -> Path:
    """Question set for Test 1, 2 or 3."""
    return DATA_GENERAL / f"questions_test{test_id}.json"


def general_raw(test_id: int, run_name: str) -> Path:
    """Raw model output for one run (e.g. ``qwen-max`` or ``qwen-max_5demos``)."""
    return RAW_GENERAL / run_name / f"test{test_id}.json"


def general_classified(test_id: int, run_name: str) -> Path:
    return PROCESSED_GENERAL / run_name / f"test{test_id}_classified.json"


SUMMARY_METRICS = PROCESSED_GENERAL / "summary_metrics.json"


# -- Diagnostic tests (Tests 4-6) ----------------------------------------------
DIAGNOSTIC_CONDITIONS = (
    "parallel_english", "parallel_chinese",
    "nonsense_english", "nonsense_chinese",
    "ambiguity_pp", "ambiguity_adjunct",
)

TEST_OF_CONDITION = {
    "parallel_english": 4, "parallel_chinese": 4,
    "nonsense_english": 5, "nonsense_chinese": 5,
    "ambiguity_pp": 6, "ambiguity_adjunct": 6,
}


def diagnostic_demonstrations(condition: str) -> Path:
    """Demonstration set; the nonsense test reuses the parallel demonstrations."""
    base = "parallel" if condition.startswith(("parallel", "nonsense")) else condition
    return DATA_DIAGNOSTIC / f"{base}_demonstrations.json"


def diagnostic_test_sentences(condition: str) -> Path:
    base = condition.rsplit("_", 1)[0] if condition.startswith(("parallel", "nonsense")) else condition
    return DATA_DIAGNOSTIC / f"{base}_test_sentences.json"


def diagnostic_trees(test_type: str) -> Path:
    """Constituency trees for ``parallel`` (Test 4) or ``nonsense`` (Test 5)."""
    return DATA_DIAGNOSTIC / f"{test_type}_trees.json"


def diagnostic_raw(condition: str, model: str) -> Path:
    return RAW_DIAGNOSTIC / f"{condition}__{model}.json"


def diagnostic_classified(test_id: int) -> Path:
    return PROCESSED_DIAGNOSTIC / f"test{test_id}_classified.json"
