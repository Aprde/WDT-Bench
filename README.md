# WDT-Bench

Data and code for **WDT-Bench**, a Word Deletion benchmark for implicitly
evaluating the syntactic competence of large language models.  The benchmark
standardises the word deletion task into six tests, organised into two components:

- **General Tests (Tests 1-3)** - natural CoNLL-2000 sentences, for
  large-scale evaluation of constituent-based processing:
  - **Test 1** Constituent Recognition
  - **Test 2** Constituent Preference (node- vs. parent-category rule)
  - **Test 3** Constituent Localization

  Since the constituent annotations derived from CoNLL-2000 do not capture all
  possible constituents, the reported constituent rate may be underestimated.

- **Diagnostic Tests (Tests 4-6)** - carefully constructed sentences, for
  targeted analysis:

  - **Test 4** Cross-linguistic Generalization (Chinese-English parallel sentences)
  - **Test 5** Semantic Independence (nonsense sentences)
  - **Test 6** Syntax-Semantics Integration (PP- and adjunct-attachment ambiguity)

All six tests share one pipeline: **run** (query a model, write raw JSON) →
**analyze** (classify deletions, compute statistics, write processed JSON) →
**plot** (the paper figures).

## Repository layout

```text
wdt_bench/                  Python package
  paths.py                  central path registry for data/results/figures
  config.py, llm.py         API configuration and the unified chat client
  text_utils.py, trees.py,  deletion extraction, tree-span classification,
  conll.py, stats.py        CoNLL chunking, statistics
  prompts/                  prompt construction
  runners/                  resumable model querying
  analysis/                 classification + statistics for Tests 1-6
  plotting/                 per-model report figures and the 4 paper figures
scripts/
  run_test.py               run a test against a model
  analyze_results.py        classify raw outputs, compute statistics
  make_figures.py           generate report and paper figures
data/
  general_tests/            question sets (JSON)
  diagnostic_tests/         demonstrations, test sentences, constituency trees (JSON)
results/
  raw/                      raw model outputs (JSON; one file per run/condition)
  processed/                classified results + summary statistics (JSON)
figures/
  paper/                    publication figures (SVG)
```

Naming note: the `task` fields inside result files use a compact scheme:
`1_1`/`1_2`/`1_3` = Tests 1/2/3 and `2_1`/`2_2`/`2_3` = Tests 4/5/6.

## Installation

Python ≥ 3.10.

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"   # only needed for new runs
```

## API configuration

No API key is stored in the repository.  Credentials come from environment
variables:

| Variable | Meaning |
| --- | --- |
| `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` | key for any OpenAI-compatible endpoint |
| `OPENAI_BASE_URL` / `DASHSCOPE_BASE_URL` | endpoint base URL (e.g. DashScope's compatible-mode URL) |
| `ANTHROPIC_API_KEY` | key for the `anthropic` provider |
| `DEFAULT_CHAT_MODEL`, `DEFAULT_LLM_PROVIDER` | optional defaults for `--model` / `--provider` |

## Running the benchmark

```bash
# General tests: per prompt variant, 100 runs x 24 questions each
# (6 variants -> 14,400 calls per model per test by default)
python scripts/run_test.py --test 1 --model qwen-flash

# Test 1 in the multi-demonstration setting (run name: qwen-max_5demos)
python scripts/run_test.py --test 1 --model qwen-max --n-demos 5

# Test 4 (runs both the parallel_english and parallel_chinese conditions)
python scripts/run_test.py --test 4 --model qwen-max

# A single diagnostic condition
python scripts/run_test.py --condition ambiguity_pp --model gpt-5.5
```

### Evaluating your own model

1. **Point the client at your endpoint.**  Set the environment variables
   (`OPENAI_API_KEY` + `OPENAI_BASE_URL` for any OpenAI-compatible endpoint,
   or `ANTHROPIC_API_KEY` with `--provider anthropic`).
2. **Pick a test and a model name** with `--test {1..6}` and
   `--model <name>`; general-test results land in
   `results/raw/general_tests/<model>/`, diagnostic conditions in
   `results/raw/diagnostic_tests/`.
3. **Choose how many runs.**  For Tests 1-3, `--n-runs N` performs N runs per
   prompt variant; each run always covers 24 questions, giving
   6 x N x 24 calls in total. For Tests 4-6, `--n-trials N` plays the same
   role (N x 24 calls per condition). Resuming an interrupted run
   re-uses the same command; resuming with different settings is refused.

For Tests 1-3, `--prompt-variants A,B` restricts the run to a subset of the six prompt framings (D/E/F implicit, A/B/C narrative); the default runs all six.

```bash
# Example: evaluate gpt-5.5 on Test 1 with 10 runs with prompt A, then analyse
python scripts/run_test.py --test 1 --model gpt-5.5 --n-runs 10 --prompt-variants A
python scripts/analyze_results.py --test 1 --run gpt-5.5
python scripts/make_figures.py
```

Add `--dry-run` (diagnostic tests) to inspect prompt assembly without API calls.

## Analysing results

```bash
python scripts/analyze_results.py --test 1 --run qwen-flash   # one general run
python scripts/analyze_results.py --test 4                    # pools all Test-4 raw files
python scripts/analyze_results.py --all                       # everything available
python scripts/analyze_results.py --span-analysis parallel_english qwen-max
```

Outputs are written to `results/processed/.../test{K}_classified.json` with a
`meta` / `summary*` / `results` structure (per-row classifications plus
bootstrap CIs and significance tests).

Test 1 classifies each deletion with the 5-way tree-span taxonomy
(`tree_span_category`), computed from the deleted token positions
(`delete_span`); the category names match the figure legend
(`single_constituent`, `multiple_constituents`, `partial_constituent`,
`constituent_plus_partial`, `other`). `other` corresponds to rows without a
valid deletion (`eval_note` explains why).

## Figures

```bash
python scripts/make_figures.py
```

## Data

- `data/general_tests/questions_test{1,2,3}.json` - the frozen question sets
  (245 / 200 / 100 items; trees and gold targets included per
  item). 
- `data/diagnostic_tests/*.json` - demonstrations, test sentences and
  constituency trees for the six diagnostic conditions (the nonsense
  conditions reuse the parallel demonstrations).
- `results/raw/` ships six general-test runs -- `qwen-flash`, `qwen-max` and
  `qwen-plus` (all three tests) plus the multi-demonstration
  `qwen-max_{2,5,10}demos` runs (Test 1 only) -- and all 36 diagnostic
  `{condition}__{model}` runs used in the paper (six conditions x six models).
