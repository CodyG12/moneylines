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


def american_implied_probability(american):
    """Raw implied probability of an American price (includes the margin)."""
    return implied_probability(american_to_decimal(american))


def margin_from_american(home_american, away_american):
    """Sportsbook margin (overround) of a two-way market quoted in American odds."""
    return overround(american_to_decimal(home_american), american_to_decimal(away_american))


def probability_edge(model_prob, market_prob_novig):
    """Model minus market (no-vig) probability for the same side. Positive does not mean profitable:
    the bettor still has to beat the price *including* the margin (see expected_value)."""
    return _arr(model_prob) - _arr(market_prob_novig)


def expected_roi(prob, decimal):
    """Expected return per unit staked under `prob`; identical to expected_value, named for reports."""
    return expected_value(prob, decimal)


def closing_line_value(bet_decimal, close_decimal_side, close_decimal_other):
    """CLV of a bet: how much better the taken price was than the no-vig closing price.

    Returns bet_decimal / fair_close_decimal - 1, where fair_close_decimal = 1 / no-vig closing
    probability of the bet's side. Positive CLV means the bet beat the closing market.
    """
    p_close, _ = remove_vig_two_way(close_decimal_side, close_decimal_other)
    return _arr(bet_decimal) * p_close - 1.0


def fair_odds_quote(home_prob, home_decimal=None, away_decimal=None) -> dict:
    """Fair model prices for one game and, when given, a comparison with a book's offered prices.

    "fair_*" fields are the model's zero-margin prices; "book_*" fields are what was offered.
    """
    p = float(home_prob)
    q = {"model_home_prob": p, "model_away_prob": 1 - p,
         "fair_home_decimal": float(fair_decimal_odds(p)), "fair_away_decimal": float(fair_decimal_odds(1 - p)),
         "fair_home_american": int(round(float(fair_american_odds(p)))),
         "fair_away_american": int(round(float(fair_american_odds(1 - p))))}
    if home_decimal is not None and away_decimal is not None:
        h, a = float(home_decimal), float(away_decimal)
        nv_h, nv_a = remove_vig_two_way(h, a)
        q.update({"book_home_decimal": h, "book_away_decimal": a,
                  "book_home_american": int(round(float(decimal_to_american(h)))),
                  "book_away_american": int(round(float(decimal_to_american(a)))),
                  "book_margin": float(overround(h, a)),
                  "break_even_home": float(break_even_probability(h)), "break_even_away": float(break_even_probability(a)),
                  "market_home_prob_novig": float(nv_h), "market_away_prob_novig": float(nv_a),
                  "edge_home": float(probability_edge(p, nv_h)), "edge_away": float(probability_edge(1 - p, nv_a)),
                  "ev_home": float(expected_value(p, h)), "ev_away": float(expected_value(1 - p, a))})
    return q
