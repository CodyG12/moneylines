import numpy as np
import pytest

from nbamodel import odds


def test_sixty_percent_fair_line_and_book_comparison():
    q = odds.fair_odds_quote(0.60, home_decimal=odds.american_to_decimal(-130), away_decimal=odds.american_to_decimal(110))
    assert q["fair_home_american"] == -150 and q["fair_away_american"] == 150
    assert q["fair_home_decimal"] == pytest.approx(1 / 0.6)
    assert q["book_home_american"] == -130 and q["book_away_american"] == 110
    assert q["book_margin"] == pytest.approx(130 / 230 + 100 / 210 - 1)
    assert q["market_home_prob_novig"] + q["market_away_prob_novig"] == pytest.approx(1)
    assert q["edge_home"] == pytest.approx(0.6 - q["market_home_prob_novig"])
    assert q["ev_home"] == pytest.approx(0.6 * (1 + 100 / 130) - 1)
    assert q["break_even_home"] == pytest.approx(130 / 230)
    assert q["model_home_prob"] + q["model_away_prob"] == pytest.approx(1)


def test_fair_quote_without_book():
    q = odds.fair_odds_quote(0.5)
    assert q["fair_home_american"] == 100 and "book_margin" not in q


def test_american_helpers():
    assert odds.american_implied_probability(-200) == pytest.approx(2 / 3)
    assert odds.margin_from_american(-110, -110) == pytest.approx(2 * 110 / 210 - 1)


def test_closing_line_value():
    # took +120 (2.20); close was -110/-110 (no-vig 50%): CLV = 2.2 * 0.5 - 1
    d = odds.american_to_decimal(-110)
    assert odds.closing_line_value(2.2, d, d) == pytest.approx(0.10)
    assert odds.closing_line_value(1.8, d, d) < 0


def test_edge_is_not_ev():
    """A positive no-vig edge can still be a negative-EV bet once the margin is paid."""
    h, a = odds.american_to_decimal(-110), odds.american_to_decimal(-110)
    p = 0.51
    assert odds.probability_edge(p, odds.remove_vig_two_way(h, a)[0]) > 0
    assert odds.expected_roi(p, h) < 0


def test_probability_inputs_validated():
    with pytest.raises(ValueError):
        odds.fair_odds_quote(1.0)
    with pytest.raises(ValueError):
        odds.fair_odds_quote(np.nan)
