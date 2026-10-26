"""Automated look-ahead checks for any feature builder.

The core test is a perturbation test: flip every result (scores, box scores and
labels) on or after a cutoff date, rebuild features, and require that features
for games *on* the cutoff date are bit-for-bit unchanged. A feature that moves
must have read that day's or a later day's outcome, including results of other
games played the same night.

A secondary screen flags features almost perfectly correlated with the label,
which catches copies of the outcome that a perturbation can miss (e.g. a feature
computed outside the builder and merged in).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .sources import BOX_STATS


class LeakageError(AssertionError):
    pass


def outcome_columns(games: pd.DataFrame) -> list:
    box = [f"{s}_{c}" for s in ("home", "away") for c in BOX_STATS]
    return [c for c in ["home_score", "away_score", "home_win", *box] if c in games.columns]


def perturb_outcomes(games: pd.DataFrame, mask) -> pd.DataFrame:
    """Swap home/away results (scores and box totals) on `mask` rows, so every winner flips."""
    out = games.copy()
    mask = np.asarray(mask, dtype=bool) & out["played"].to_numpy()
    pairs = [("home_score", "away_score")] + [(f"home_{c}", f"away_{c}") for c in BOX_STATS
                                              if f"home_{c}" in out.columns]
    for h, a in pairs:
        out.loc[mask, [h, a]] = out.loc[mask, [a, h]].to_numpy()
    out.loc[mask, "home_win"] = 1.0 - out.loc[mask, "home_win"]
    return out


def default_cutoffs(games: pd.DataFrame, n: int = 3) -> list:
    dates = np.sort(games.loc[games["played"], "date"].unique())
    if len(dates) < 3:
        raise ValueError("need at least 3 played dates for a leakage check")
    picks = np.linspace(len(dates) * 0.3, len(dates) * 0.9, n).astype(int)
    return [pd.Timestamp(dates[i]) for i in np.unique(picks)]


def find_leaking_features(games: pd.DataFrame, feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
                          feature_cols: list, cutoffs: list | None = None) -> dict:
    """Return {feature: rows changed} for features that depend on same-day or future results."""
    cutoffs = cutoffs or default_cutoffs(games)
    base = feature_fn(games)
    leaks: dict = {}
    for cutoff in cutoffs:
        tampered = feature_fn(perturb_outcomes(games, games["date"] >= cutoff))
        on = (games["date"] == cutoff).to_numpy()
        for col in feature_cols:
            a = base.loc[on, col].to_numpy(dtype=float)
            b = tampered.loc[on, col].to_numpy(dtype=float)
            changed = int((~np.isclose(a, b, rtol=0, atol=1e-12, equal_nan=True)).sum())
            if changed:
                leaks[col] = leaks.get(col, 0) + changed
    return leaks


def suspicious_label_correlation(df: pd.DataFrame, feature_cols: list, label: str = "home_win",
                                 threshold: float = 0.9) -> dict:
    played = df[df[label].notna()]
    varying = [c for c in feature_cols if played[c].nunique(dropna=True) > 1]
    corr = played[varying].corrwith(played[label]).abs()
    return {k: float(v) for k, v in corr.items() if v >= threshold}


def assert_no_leakage(games: pd.DataFrame, feature_fn, feature_cols: list, cutoffs=None) -> None:
    leaks = find_leaking_features(games, feature_fn, feature_cols, cutoffs)
    suspicious = suspicious_label_correlation(feature_fn(games), feature_cols)
    if leaks or suspicious:
        raise LeakageError(f"look-ahead detected: perturbation={leaks}, label-correlation={suspicious}")
