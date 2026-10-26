"""Odds conversions and expected-value math for two-way moneyline markets.

All functions accept scalars or array-likes and return numpy values.
Invalid inputs raise ValueError instead of silently producing nonsense.
"""
from __future__ import annotations

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _check_prob(p: np.ndarray) -> None:
    if np.any(~np.isfinite(p)) or np.any((p <= 0) | (p >= 1)):
        raise ValueError("probabilities must be strictly between 0 and 1")


def _check_decimal(d: np.ndarray) -> None:
    if np.any(~np.isfinite(d)) or np.any(d <= 1.0):
        raise ValueError("decimal odds must be greater than 1.0")


def american_to_decimal(american):
    a = _arr(american)
    if np.any(~np.isfinite(a)) or np.any(np.abs(a) < 100):
        raise ValueError("American odds must be <= -100 or >= +100")
    return np.where(a > 0, 1.0 + a / 100.0, 1.0 + 100.0 / np.abs(a))


def decimal_to_american(decimal):
    d = _arr(decimal)
    _check_decimal(d)
    with np.errstate(divide="ignore"):
        return np.where(d >= 2.0, (d - 1.0) * 100.0, -100.0 / (d - 1.0))


def implied_probability(decimal):
    """Raw implied probability (includes the bookmaker margin)."""
    d = _arr(decimal)
    _check_decimal(d)
    return 1.0 / d


def break_even_probability(decimal):
    """Win probability needed for a bet at these odds to have zero EV."""
    return implied_probability(decimal)


def overround(decimal_a, decimal_b):
    """Bookmaker margin of a complete two-way market (0.045 == 4.5%)."""
    return implied_probability(decimal_a) + implied_probability(decimal_b) - 1.0


def remove_vig_two_way(decimal_a, decimal_b):
    """No-vig probabilities via proportional (multiplicative) normalisation.

    This is the simplest de-vig method; it ignores favourite-longshot bias.
    """
    qa = implied_probability(decimal_a)
    qb = implied_probability(decimal_b)
    total = qa + qb
    return qa / total, qb / total


def fair_decimal_odds(prob):
    p = _arr(prob)
    _check_prob(p)
    return 1.0 / p


def fair_american_odds(prob):
    return decimal_to_american(fair_decimal_odds(prob))


def expected_value(prob, decimal):
    """Expected profit per 1 unit staked, under the *model's* probability.

    This is an estimate conditional on the model being correct, not a
    realised or guaranteed return.
    """
    p = _arr(prob)
    d = _arr(decimal)
    _check_prob(p)
    _check_decimal(d)
    return p * d - 1.0


def kelly_fraction(prob, decimal):
    """Full-Kelly bankroll fraction (clipped at 0). Assumes `prob` is exact,
    which it never is -- callers should apply a fractional multiplier and caps."""
    p = _arr(prob)
    d = _arr(decimal)
    _check_prob(p)
    _check_decimal(d)
    return np.clip((p * d - 1.0) / (d - 1.0), 0.0, None)
