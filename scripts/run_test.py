#!/usr/bin/env python3
"""Run a WDT-Bench test against a model.

General tests (1-3) query the model on the English question sets; diagnostic
tests (4-6) run one of the six diagnostic conditions.  Both runners are
resumable: re-running the same command continues from the last checkpoint.

Examples
--------
# Test 1 with the default provider/model from environment variables:
python scripts/run_test.py --test 1 --model qwen-flash

# Test 1, multi-demonstration setting (run name becomes qwen-max_5demos):
python scripts/run_test.py --test 1 --model qwen-max --n-demos 5

# Test 2 with another model, 20 runs per prompt variant (6 x 20 x 24 = 2,880 calls):
python scripts/run_test.py --test 2 --model gpt-5.5 --provider openai --n-runs 20

# Test 4 (both languages = two conditions) for one model:
python scripts/run_test.py --test 4 --model qwen-max
python scripts/run_test.py --condition parallel_english --model qwen-max

# Smoke-test the prompt assembly without API calls:
python scripts/run_test.py --condition ambiguity_pp --model qwen-max --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wdt_bench import paths  # noqa: E402
from wdt_bench.prompts.general import ALL_PROMPT_VARIANTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--test", type=int, choices=range(1, 7),
                        help="Test number (1-3 general, 4-6 diagnostic).")
    target.add_argument("--condition", choices=paths.DIAGNOSTIC_CONDITIONS,
                        help="Run a single diagnostic condition instead of a whole test.")
    ap.add_argument("--model", required=True, help="Model name as sent to the API.")
    ap.add_argument("--provider", default=None,
                    help="API provider (default: openai-compatible endpoint from env).")
    ap.add_argument("--run-name", default=None,
                    help="General tests: results folder name (default: model name, "
                         "or model_{N}demos when --n-demos > 1).")
    ap.add_argument("--n-demos", type=int, default=1,
                    help="General tests: number of in-context demonstrations (default 1).")
    ap.add_argument("--n-runs", type=int, default=100,
                    help="General tests: number of runs per prompt variant "
                         "(default 100). Each run always covers 24 questions; "
                         "resuming with a different value is refused.")
    ap.add_argument("--prompt-variants", default=None,
                    help="General tests: comma-separated subset of "
                         f"{','.join(ALL_PROMPT_VARIANTS)} (default: all).")
    ap.add_argument("--n-trials", type=int, default=30,
                    help="Diagnostic tests: trials per condition (default 30).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None,
                    help="Diagnostic tests: API-side sampling seed.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Diagnostic tests: assemble prompts without calling the API.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if args.test in (1, 2, 3):
        from wdt_bench.runners.general import run_general_test

        variants = (
            tuple(v.strip() for v in args.prompt_variants.split(","))
            if args.prompt_variants
            else ALL_PROMPT_VARIANTS
        )
        out = run_general_test(
            args.test,
            model=args.model,
            provider=args.provider,
            run_name=args.run_name,
            n_demos=args.n_demos,
            prompt_variants=variants,
            n_replicates=args.n_runs,
        )
        print(f"Raw results written to {out}")
        return

    from wdt_bench.runners.diagnostic import run_diagnostic_test

    if args.condition:
        conditions = [args.condition]
    else:
        conditions = [c for c, t in paths.TEST_OF_CONDITION.items() if t == args.test]
    for condition in conditions:
        kwargs = dict(
            n_trials=args.n_trials,
            temperature=args.temperature,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        if args.provider:
            kwargs["provider"] = args.provider
        info = run_diagnostic_test(condition, args.model, **kwargs)
        print(f"[{condition}] {info['n_results']} results "
              f"({info['new_calls']} new calls) -> {info['path']}")


if __name__ == "__main__":
    main()
