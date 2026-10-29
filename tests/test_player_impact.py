"""Player-impact features on a synthetic league with known player ratings."""
import numpy as np
import pandas as pd
import pytest

from nbamodel import player_impact as pi
from nbamodel.injuries import NOT_SUBMITTED

TEAMS = ["AAA", "BBB", "CCC", "DDD"]


def _league(n_days=120, star_out_from=None, seed=0):
    """Each team has a star (true +8 per 48), four starters (0), three bench players (-3) and a
    deep-bench player (-5, about 6 minutes: the replacement pool).
    Margins are generated from those ratings plus noise. Stars rest one game in five at random and
    minutes vary game to game: without that variation individual ratings are not identifiable
    (only the team total is), exactly as with real box scores. The AAA star also sits every game
    from `star_out_from` on."""
    rng = np.random.default_rng(seed)
    roster = {t: [(f"{t}-star", 8.0, 34.0)] + [(f"{t}-s{i}", 0.0, 32.0) for i in range(4)]
              + [(f"{t}-b{i}", -3.0, 16.0) for i in range(3)] + [(f"{t}-r0", -5.0, 6.0)] for t in TEAMS}
    rows, start = [], pd.Timestamp("2025-01-01")
    for d in range(n_days):
        day = start + pd.Timedelta(days=d)
        order = rng.permutation(TEAMS)
        for k in range(2):
            home, away = order[2 * k], order[2 * k + 1]
            eid = f"e{d}-{k}"
            lines = {}
            for side, team in (("home", home), ("away", away)):
                plist = []
                for pid, r, mins in roster[team]:
                    out = team == "AAA" and pid == "AAA-star" and star_out_from is not None and d >= star_out_from
                    rest = pid.endswith("star") and star_out_from is None and rng.random() < 0.2
                    plist.append((pid, r, 0.0 if out or rest else max(4.0, mins + rng.normal(0, 4))))
                spare = 240 - sum(m for _, _, m in plist)          # absent minutes go to the bench
                plist = [(p, r, m + (spare / 3 if p.split("-")[1].startswith("b") else 0)) for p, r, m in plist]
                lines[side] = plist
            strength = {s: sum(r * m / 48 for _, r, m in lines[s]) for s in lines}
            margin = strength["home"] - strength["away"] + 2.5 + rng.normal(0, 8)
            for side, team in (("home", home), ("away", away)):
                share = margin / 2 if side == "home" else -margin / 2
                for pid, r, mins in lines[side]:
                    pts = (100 + share) * mins / 240 if mins > 0 else 0
                    rows.append({"espn_event_id": eid, "player_id": pid, "player_name": pid.replace("-", " "),
                                 "game_key": eid, "game_date": day.date().isoformat(), "team": team,
                                 "opponent": away if side == "home" else home, "home": int(side == "home"),
                                 "did_not_play": int(mins == 0), "minutes": mins, "pts": pts})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def league():
    return _league()


def test_ratings_recover_star_and_use_only_earlier_games(league):
    r, rep = pi.fit_ratings(league, "2025-05-01", pi.ImpactConfig(ridge_lambda=5))
    stars = r[[f"{t}-star" for t in TEAMS]]
    bench = r[[i for i in r.index if "-b" in i]]
    assert (stars > 4).all() and stars.mean() > r.mean() + 4
    assert bench.mean() < r.mean()
    assert rep < 0                                     # replacement = the low-minute bench
    # a blowout after the cutoff changes nothing
    late = league[league["espn_event_id"] == "e0-0"].assign(espn_event_id="late", game_date="2025-06-01",
                                                            pts=lambda d: np.where(d["home"] == 1, 500, 0))
    r2, _ = pi.fit_ratings(pd.concat([league, late]), "2025-05-01", pi.ImpactConfig(ridge_lambda=5))
    pd.testing.assert_series_equal(r, r2)


def test_ridge_shrinks_small_samples_toward_average():
    lg = _league(n_days=6)
    loose, _ = pi.fit_ratings(lg, "2025-02-01", pi.ImpactConfig(ridge_lambda=0.5))
    tight, _ = pi.fit_ratings(lg, "2025-02-01", pi.ImpactConfig(ridge_lambda=500))
    assert tight.abs().max() < loose.abs().max() and tight.abs().max() < 1


def test_minutes_recent_vs_healthy():
    lg = _league(n_days=60, star_out_from=57)          # AAA star missed the last 3 days' games
    m = pi.minutes_asof(lg, "2025-03-02").set_index("player_id")
    star = m.loc["AAA-star"]
    assert star["healthy"] == pytest.approx(34.0, abs=4)
    assert 0 < star["recent"] < star["healthy"]
    assert m.loc["BBB-star", "recent"] == pytest.approx(m.loc["BBB-star", "healthy"], abs=4)
    assert m.loc["AAA-star", "player_key"] == "aaastar"


def _team_inputs(lg, cutoff, asof):
    r, rep = pi.fit_ratings(lg, cutoff, pi.ImpactConfig(ridge_lambda=5))
    return pi.minutes_asof(lg, asof), r, rep


def _status(team, key, status):
    return pd.DataFrame({"team": [team], "player_key": [key], "status": [status]})


PROBS = dict(pi.ImpactConfig().default_play_prob)


def test_star_newly_out_lowers_impact_but_long_absence_does_not():
    lg = _league()
    minutes, r, rep = _team_inputs(lg, "2025-04-25", "2025-04-30")
    fresh = pi.team_availability("AAA", _status("AAA", "aaastar", "out"), minutes, r, rep, PROBS)
    healthy = pi.team_availability("AAA", _status("AAA", "aaab0", "probable"), minutes, r, rep, PROBS)
    healthy_min = minutes.set_index("player_id").loc["AAA-star", "healthy"]
    expected = (r["AAA-star"] - rep) * healthy_min / 48
    assert fresh["impact"] == pytest.approx(healthy["impact"] - expected * PROBS["available"], rel=0.05)
    assert fresh["impact"] < -3 and fresh["n_rotation_out"] == 1
    long_out = _league(star_out_from=90)                # out for the last 30 days: in the team numbers
    minutes2, r2, rep2 = _team_inputs(long_out, "2025-04-25", "2025-04-30")
    still = pi.team_availability("AAA", _status("AAA", "aaastar", "out"), minutes2, r2, rep2, PROBS)
    assert abs(still["impact"]) < 1.0


def test_questionable_is_a_distribution_not_a_guess():
    minutes, r, rep = _team_inputs(_league(), "2025-04-25", "2025-04-30")
    q = pi.team_availability("AAA", _status("AAA", "aaastar", "questionable"), minutes, r, rep, PROBS)
    out = pi.team_availability("AAA", _status("AAA", "aaastar", "out"), minutes, r, rep, PROBS)
    ok = pi.team_availability("AAA", _status("AAA", "aaab0", "available"), minutes, r, rep, PROBS)
    assert out["impact"] < q["impact"] < ok["impact"]
    assert q["impact_sd"] > ok["impact_sd"] and q["n_rotation_uncertain"] == 1


def test_missing_report_is_nan_and_unknown_players_are_reported():
    minutes, r, rep = _team_inputs(_league(), "2025-04-25", "2025-04-30")
    none = pi.team_availability("AAA", None, minutes, r, rep, PROBS)
    assert none["report_available"] == 0 and np.isnan(none["impact"])
    rookie = pi.team_availability("AAA", _status("AAA", "brandnewguy", "out"), minutes, r, rep, PROBS)
    assert rookie["unmatched"] == ["brandnewguy"] and rookie["report_available"] == 1
    pending = pi.team_availability("AAA", _status("AAA", "", NOT_SUBMITTED), minutes, r, rep, PROBS)
    assert pending["impact_sd"] > 0                     # not submitted yet: everyone uncertain


def test_play_probabilities_measured_from_past_games():
    lg = _league(n_days=40)
    stars = lg[lg["player_id"].str.endswith("star")][["game_key", "team", "player_name"]].copy()
    stars["player_key"] = stars["player_name"].str.replace(" ", "").str.lower()
    stars["status"] = "questionable"
    probs = pi.play_probabilities(stars, lg, min_n=10)
    star_rows = lg[lg["player_id"].str.endswith("star")]
    assert probs["questionable"] == pytest.approx((star_rows["minutes"] > 0).mean())   # the rate they played
    assert probs[pi.UNLISTED] == pytest.approx(1.0)                # everyone else always plays
    assert probs["out"] == 0.0                           # not enough cases: conservative default kept


def _changes(rows):
    df = pd.DataFrame(rows, columns=["game_date", "matchup", "team", "player_key", "status", "published_at"])
    return df.assign(player_name=df["player_key"], reason=None)


def test_availability_features_use_only_reports_before_prediction_time():
    lg = _league()
    day = "2025-04-30"
    rows = pd.DataFrame({"game_key": ["g1"], "date": [day], "season": [2024], "home_team": ["AAA"],
                         "away_team": ["BBB"], "prediction_timestamp": ["2025-04-30T22:00:00+00:00"]})
    changes = _changes([[day, "BBB@AAA", "AAA", "aaastar", "questionable", "2025-04-30T17:00:00+00:00"],
                        [day, "BBB@AAA", "AAA", "aaastar", "out", "2025-04-30T23:00:00+00:00"]])  # after: ignored
    first = pd.Series({day: "2025-04-30T17:00:00+00:00"})
    f = pi.availability_features(rows, changes, first, lg, pi.ImpactConfig(ridge_lambda=5)).iloc[0]
    assert f["home_report_available"] == 1 and f["home_n_rotation_uncertain"] == 1 and f["home_n_rotation_out"] == 0
    # everyone unlisted carries the small unlisted no-show risk; the questionable star adds to it
    assert f["home_impact_sd"] > f["away_impact_sd"] > 0
    assert f["injury_impact_diff"] == pytest.approx(f["home_impact"] - f["away_impact"])
    # no report yet at prediction time: NaN, not "everyone healthy"
    early = rows.assign(prediction_timestamp="2025-04-30T16:00:00+00:00")
    g = pi.availability_features(early, changes, first, lg, pi.ImpactConfig(ridge_lambda=5)).iloc[0]
    assert g["home_report_available"] == 0 and np.isnan(g["injury_impact_diff"])


def test_team_missing_from_a_published_report_counts_as_unlisted():
    lg = _league()
    day = "2025-04-30"
    rows = pd.DataFrame({"game_key": ["g1"], "date": [day], "season": [2024], "home_team": ["CCC"],
                         "away_team": ["DDD"], "prediction_timestamp": ["2025-04-30T22:00:00+00:00"]})
    changes = _changes([[day, "BBB@AAA", "AAA", "aaastar", "out", "2025-04-30T17:00:00+00:00"]])  # other game only
    first = pd.Series({day: "2025-04-30T17:00:00+00:00"})
    f = pi.availability_features(rows, changes, first, lg, pi.ImpactConfig(ridge_lambda=5)).iloc[0]
    assert f["home_report_available"] == 1 and not np.isnan(f["injury_impact_diff"])


def test_rating_cutoff_is_the_monday_of_the_game_week():
    assert pi.rating_cutoff("2026-10-28") == pd.Timestamp("2026-10-26")      # Wednesday -> Monday
    assert pi.rating_cutoff("2026-10-26") == pd.Timestamp("2026-10-26")      # Monday: that day's games excluded
