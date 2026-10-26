"""Probability scoring, cluster-bootstrap uncertainty, and calibration tables.

Games on the same date share conditions (and in betting, bankroll), so
confidence intervals resample whole dates rather than individual games.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-15


def per_game_log_loss(y, p) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def per_game_brier(y, p) -> np.ndarray:
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def score(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "n": int(len(y)),
        "log_loss": float(per_game_log_loss(y, p).mean()),
        "brier": float(per_game_brier(y, p).mean()),
        "accuracy": float(((p >= 0.5) == (y == 1)).mean()),
    }


def cluster_bootstrap_ratio(numerator, denominator, clusters, n_boot: int = 1000,
                            seed: int = 0, alpha: float = 0.05) -> dict:
    """CI for sum(numerator)/sum(denominator), resampling whole clusters.

    With denominator == 1 this is a cluster-bootstrapped mean.
    """
    frame = pd.DataFrame({"num": numerator, "den": denominator, "c": clusters})
    agg = frame.groupby("c")[["num", "den"]].sum()
    nums, dens = agg["num"].to_numpy(), agg["den"].to_numpy()
    k = len(agg)
    point = float(nums.sum() / dens.sum())
    if k < 2:
        return {"estimate": point, "ci_low": float("nan"), "ci_high": float("nan"), "n_clusters": k}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(n_boot, k))
    stats = nums[draws].sum(axis=1) / dens[draws].sum(axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi), "n_clusters": k}


def compare_log_loss(y, p_model, p_baseline, clusters, n_boot=1000, seed=0) -> dict:
    """Mean log-loss difference (model - baseline); negative favours the model."""
    diff = per_game_log_loss(y, p_model) - per_game_log_loss(y, p_baseline)
    return cluster_bootstrap_ratio(diff, np.ones_like(diff), clusters, n_boot, seed)


def wilson_interval(successes, n, z: float = 1.96):
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    phat = successes / n
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table. Wilson intervals treat games as independent, so read
    them as approximate."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            continue
        wins = y[mask].sum()
        lo, hi = wilson_interval(wins, n)
        rows.append({
            "bin_low": edges[b], "bin_high": edges[b + 1], "n": n,
            "mean_predicted": float(p[mask].mean()), "observed_rate": float(wins / n),
            "observed_ci_low": float(lo), "observed_ci_high": float(hi),
        })
    return pd.DataFrame(rows)
