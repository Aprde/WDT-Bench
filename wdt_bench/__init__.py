"""WDT-Bench: a word-deletion benchmark for the syntactic competence of LLMs.

The package is organised into four layers that mirror the experimental
pipeline of the paper:

- ``wdt_bench.prompts``  -- prompt construction for all six tests
- ``wdt_bench.runners``  -- model querying (raw results, JSON)
- ``wdt_bench.analysis`` -- classification and statistics (processed results, JSON)
- ``wdt_bench.plotting`` -- paper-level figures
"""

__version__ = "2.0.0"
