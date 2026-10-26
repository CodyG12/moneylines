import numpy as np
import pytest

from nbamodel import odds


@pytest.mark.parametrize("american,decimal", [(100, 2.0), (150, 2.5), (-200, 1.5), (-110, 1 + 100 / 110)])
def test_american_decimal_round_trip(american, decimal):
    assert odds.american_to_decimal(american) == pytest.approx(decimal)
    assert odds.decimal_to_american(decimal) == pytest.approx(american)


@pytest.mark.parametrize("bad", [0, 50, -99.9, np.nan])
def test_invalid_american_rejected(bad):
    with pytest.raises(ValueError):
        odds.american_to_decimal(bad)


def test_invalid_decimal_and_prob_rejected():
    with pytest.raises(ValueError):
        odds.implied_probability(1.0)
    with pytest.raises(ValueError):
        odds.fair_decimal_odds(0.0)
    with pytest.raises(ValueError):
        odds.fair_decimal_odds(1.0)


def test_vig_removal_standard_line():
    d = odds.american_to_decimal(-110)
    assert odds.overround(d, d) == pytest.approx(2 * 110 / 210 - 1)
    a, b = odds.remove_vig_two_way(d, d)
    assert a == pytest.approx(0.5) and b == pytest.approx(0.5)


def test_vig_removal_sums_to_one():
    a, b = odds.remove_vig_two_way([1.5, 1.2], [2.6, 4.5])
    np.testing.assert_allclose(a + b, 1.0)
    assert np.all(a > b)


def test_fair_odds_and_ev():
    assert odds.fair_american_odds(0.5) == pytest.approx(100)
    assert odds.fair_american_odds(0.6) == pytest.approx(-150)
    assert odds.fair_american_odds(0.4) == pytest.approx(150)
    # fair odds have zero EV by construction
    assert odds.expected_value(0.6, odds.fair_decimal_odds(0.6)) == pytest.approx(0.0)
    assert odds.expected_value(0.55, 2.0) == pytest.approx(0.10)
    assert odds.break_even_probability(2.5) == pytest.approx(0.4)


def test_kelly():
    assert odds.kelly_fraction(0.55, 2.0) == pytest.approx(0.10)
    assert odds.kelly_fraction(0.40, 2.0) == 0.0  # negative edge -> abstain
