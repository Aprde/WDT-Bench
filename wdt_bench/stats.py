"""Bootstrap confidence intervals and significance tests.

This module merges the statistical helpers that were previously scattered
across the General-Tests ``stats.py``, the Diagnostic-Tests
``significance_test.py``, and the inline copies inside the figure scripts.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci_mean(
    values: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI for the mean of ``values`` -> (mean, low, high)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float(arr.mean())
    boots: List[float] = []
    n = arr.size
    for _ in range(n_boot):
        samp = rng.choice(arr, size=n, replace=True)
        boots.append(float(samp.mean()))
    boots_arr = np.asarray(boots)
    return obs, float(np.quantile(boots_arr, alpha / 2)), float(np.quantile(boots_arr, 1 - alpha / 2))


def bootstrap_ci_proportion(
    successes: Sequence[bool] | Sequence[int],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    vals = np.asarray([1.0 if bool(x) else 0.0 for x in successes], dtype=float)
    return bootstrap_ci_mean(vals, n_boot=n_boot, alpha=alpha, seed=seed)


def bootstrap_two_prop_diff_ci(
    node_flags: Sequence[bool],
    parent_flags: Sequence[bool],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean(node) - mean(parent) on paired indicator vectors."""
    rng = np.random.default_rng(seed)
    n_arr = np.asarray([1.0 if x else 0.0 for x in node_flags], dtype=float)
    p_arr = np.asarray([1.0 if x else 0.0 for x in parent_flags], dtype=float)
    if n_arr.size != p_arr.size:
        raise ValueError("node_flags and parent_flags length mismatch")
    n = n_arr.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float((n_arr - p_arr).mean())
    boots: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float((n_arr[idx] - p_arr[idx]).mean()))
    b = np.asarray(boots)
    return obs, float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------------
def p_value_vs_heterogeneous_baseline(
    constituent_flags: Sequence[bool],
    baseline_probs: Sequence[float],
    n_boot: int = 1000,
    seed: int = 0,
) -> float:
    """Null: each trial *i* is an independent Bernoulli(``baseline_probs[i]``).

    Two-sided p-value for the deviation of the observed rate from the mean
    baseline.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray([1.0 if x else 0.0 for x in constituent_flags], dtype=float)
    b = np.asarray(baseline_probs, dtype=float)
    n = y.size
    if n == 0:
        return 1.0
    obs_rate = float(y.mean())
    mean_b = float(b.mean())
    obs_diff = abs(obs_rate - mean_b)
    extreme = 0
    for _ in range(n_boot):
        synth = (rng.random(n) < b).astype(float)
        if abs(float(synth.mean()) - mean_b) >= obs_diff - 1e-12:
            extreme += 1
    return max(1, extreme) / n_boot


def paired_bootstrap_node_minus_parent(
    labels: Sequence[str],
    n_boot: int = 1000,
    seed: int = 0,
) -> float:
    """Full-trial paired bootstrap: node_rule -> +1, parent_rule -> -1, else 0.

    Two-sided p-value for the mean encoding being 0 (no net preference).
    """
    rng = np.random.default_rng(seed)
    enc = np.asarray(
        [1 if lab == "node_rule" else (-1 if lab == "parent_rule" else 0) for lab in labels],
        dtype=float,
    )
    n = enc.size
    if n == 0:
        return 1.0
    obs_mean = float(enc.mean())
    extreme = 0
    for _ in range(n_boot):
        samp = rng.choice(enc, size=n, replace=True)
        if abs(float(samp.mean())) >= abs(obs_mean) - 1e-12:
            extreme += 1
    return max(1, extreme) / n_boot


def paired_bootstrap_test(a: Sequence[float], b: Sequence[float], num_bootstrap: int = 10000) -> float:
    """Paired bootstrap test on per-system score vectors (two-sided)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    means = [float(np.mean(np.random.choice(diff, size=len(diff), replace=True))) for _ in range(num_bootstrap)]
    means_arr = np.asarray(means)
    a_count = min(int(np.sum(means_arr > 0)), int(np.sum(means_arr < 0)))
    return (2 * (a_count + 1)) / (num_bootstrap + 1)


def unpaired_bootstrap_test(a: Sequence[float], b: Sequence[float], num_bootstrap: int = 10000) -> float:
    """Unpaired bootstrap test on two independent score vectors (two-sided)."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diffs = []
    for _ in range(num_bootstrap):
        pa = np.random.choice(a_arr, size=len(a_arr), replace=True)
        pb = np.random.choice(b_arr, size=len(b_arr), replace=True)
        diffs.append(float(np.mean(pa) - np.mean(pb)))
    diffs_arr = np.asarray(diffs)
    a_count = min(int(np.sum(diffs_arr > 0)), int(np.sum(diffs_arr < 0)))
    return (2 * (a_count + 1)) / (num_bootstrap + 1)


def permutation_p(a: Sequence[float], b: Sequence[float], *, seed: int, n_perm: int = 10000) -> float:
    """Two-sided permutation test on per-trial rate vectors.

    Uses a paired sign-flip test when ``len(a) == len(b)`` and an unpaired
    label permutation otherwise (the convention used by the paper figures).
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_arr = a_arr[np.isfinite(a_arr)]
    b_arr = b_arr[np.isfinite(b_arr)]
    if len(a_arr) == 0 or len(b_arr) == 0:
        return 1.0

    rng = np.random.default_rng(seed)
    if len(a_arr) == len(b_arr):
        diff = a_arr - b_arr
        obs = abs(float(np.mean(diff)))
        if obs <= 1e-15:
            return 1.0
        extreme = 0
        for _ in range(n_perm):
            signs = rng.choice(np.array([-1.0, 1.0]), size=len(diff))
            if abs(float(np.mean(diff * signs))) >= obs - 1e-12:
                extreme += 1
        return (extreme + 1) / (n_perm + 1)

    pooled = np.concatenate([a_arr, b_arr])
    obs = abs(float(np.mean(a_arr) - np.mean(b_arr)))
    if obs <= 1e-15:
        return 1.0
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if abs(float(np.mean(perm[: len(a_arr)]) - np.mean(perm[len(a_arr):]))) >= obs - 1e-12:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson_interval_from_rate(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval given an already-computed rate (used by paper figures)."""
    if n == 0:
        return p, p
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def binomial_two_sided_p_equal_split(k: int, n: int) -> float:
    """Exact two-sided binomial test of H0: p = 0.5 (log-space, overflow-safe)."""
    if n <= 0:
        return 1.0
    k = max(0, min(int(k), int(n)))
    n = int(n)

    def log_pmf(i: int) -> float:
        return (
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            - n * math.log(2.0)
        )

    log_obs = log_pmf(k)
    log_terms = [log_pmf(i) for i in range(n + 1) if log_pmf(i) <= log_obs + 1e-9]
    if not log_terms:
        return 1.0
    m = max(log_terms)
    total = sum(math.exp(t - m) for t in log_terms)
    return min(1.0, total * math.exp(m))


def two_proportion_z_p_value(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided z-test for independent proportions (large-sample)."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    if p_pool <= 0 or p_pool >= 1:
        return 1.0 if abs(p1 - p2) < 1e-12 else 0.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se < 1e-15:
        return 1.0
    z = abs(p1 - p2) / se
    p = math.erfc(z / math.sqrt(2))
    return min(1.0, max(0.0, p))


def stars_from_p(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."
