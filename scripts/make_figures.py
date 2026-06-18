#!/usr/bin/env python3
"""Generate the four publication figures from the classified results.

Figures are written to ``figures/paper/``.  A figure whose input data is not
present is skipped with an explanatory message (e.g. when only part of the
runs is shipped).

Example
-------
python scripts/make_figures.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("make_figures")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    from wdt_bench.plotting import fig_test1, fig_test6, fig_tests2_3, fig_tests4_5

    for name, mod in (
        ("fig_test1", fig_test1),
        ("fig_tests2_3", fig_tests2_3),
        ("fig_tests4_5", fig_tests4_5),
        ("fig_test6", fig_test6),
    ):
        try:
            mod.main()
        except RuntimeError as exc:
            logger.warning("[%s] skipped: %s", name, exc)


if __name__ == "__main__":
    main()
