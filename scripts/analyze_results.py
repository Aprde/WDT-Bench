#!/usr/bin/env python3
"""Classify raw model outputs and compute the summary statistics.

General tests (1-3) operate per run (one folder under
``results/raw/general_tests``); diagnostic tests (4-6) pool every
``{condition}__{model}.json`` file present under
``results/raw/diagnostic_tests``.  Outputs land next to the inputs under
``results/processed``.


Examples
--------
python scripts/analyze_results.py --test 1 --run qwen-flash
python scripts/analyze_results.py --test 4
python scripts/analyze_results.py --all            # everything available
python scripts/analyze_results.py --span-analysis parallel_english qwen-max
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wdt_bench import paths  # noqa: E402

logger = logging.getLogger("analyze_results")


def _general_runs() -> list[str]:
    base = paths.RAW_GENERAL
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def _analyze_general(test_id: int, run_name: str) -> None:
    from wdt_bench.analysis import test1, test2, test3

    mod = {1: test1, 2: test2, 3: test3}[test_id]
    mod.run(run_name)


def _analyze_diagnostic(test_id: int) -> None:
    if test_id in (4, 5):
        from wdt_bench.analysis import tests4_5

        tests4_5.run(test_id)
    else:
        from wdt_bench.analysis import test6

        test6.run()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--test", type=int, choices=range(1, 7),
                      help="Test number (1-3 general, 4-6 diagnostic).")
    what.add_argument("--all", action="store_true",
                      help="Analyse every test for which raw data is present.")
    what.add_argument("--span-analysis", nargs=2, metavar=("CONDITION", "MODEL"),
                      help="Quick span-level analysis of one diagnostic run.")
    ap.add_argument("--run", default=None,
                    help="General tests: run folder name (e.g. qwen-flash). "
                         "Omit to analyse every run folder found.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if args.span_analysis:
        from wdt_bench.analysis import span_analysis

        condition, model = args.span_analysis
        span_analysis.run(condition, model)
        return

    if args.all:
        for run in _general_runs():
            for test_id in (1, 2, 3):
                if paths.general_raw(test_id, run).is_file():
                    logger.info("== Test %d | %s ==", test_id, run)
                    _analyze_general(test_id, run)
        for test_id in (4, 5, 6):
            logger.info("== Test %d ==", test_id)
            _analyze_diagnostic(test_id)
        return

    if args.test in (1, 2, 3):
        runs = [args.run] if args.run else _general_runs()
        if not runs:
            raise SystemExit("No run folders found under results/raw/general_tests.")
        for run in runs:
            if not paths.general_raw(args.test, run).is_file():
                logger.warning("Skipping %s: no raw file for Test %d.", run, args.test)
                continue
            logger.info("== Test %d | %s ==", args.test, run)
            _analyze_general(args.test, run)
    else:
        _analyze_diagnostic(args.test)


if __name__ == "__main__":
    main()
