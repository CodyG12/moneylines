import json

import pandas as pd
import pytest

from nbamodel import odds_api, odds_collect, odds_schedule as osch, store
from tests.odds_fixtures import GAMES

T = pd.Timestamp


def test_targets_are_offsets_before_tipoff():
    tg = osch.targets_for(GAMES.iloc[[0]], osch.ScheduleConfig())
    assert list(tg["offset"]) == [1440, 360, 60, 15]
    assert list(tg["target"]) == [T("2026-10-25T23:00Z"), T("2026-10-26T17:00Z"), T("2026-10-26T22:00Z"),
                                  T("2026-10-26T22:45Z")]


def test_call_inside_window_satisfies_but_not_after_tipoff():
    cfg = osch.ScheduleConfig(offsets_minutes=(15,))
    tg = osch.targets_for(GAMES.iloc[[0]], cfg)          # target 22:45, window 22:40-22:53
    assert osch.satisfied(tg, ["2026-10-26T22:50Z"]).all()
    assert not osch.satisfied(tg, ["2026-10-26T22:30Z", "2026-10-26T23:01Z"]).any()
    assert osch.due(tg, [], T("2026-10-26T22:44Z")).empty           # target time not reached yet
    assert len(osch.due(tg, [], T("2026-10-26T22:45Z"))) == 1
    assert osch.due(tg, ["2026-10-26T22:45Z"], T("2026-10-26T22:50Z")).empty


def test_one_call_serves_games_with_the_same_tipoff():
    two = pd.concat([GAMES.iloc[[0]], GAMES.iloc[[0]].assign(game_key="g2", home_team="BOS", away_team="NYK")])
    cfg = osch.ScheduleConfig(offsets_minutes=(60, 15))
    calls = osch.simulate(osch.targets_for(two, cfg), T("2026-10-26T12:00Z"), T("2026-10-27T00:00Z"))
    assert calls == [T("2026-10-26T22:00Z"), T("2026-10-26T22:45Z")]
    # staggered tip-offs (23:00 and 02:00) need separate calls
    calls = osch.simulate(osch.targets_for(GAMES, cfg), T("2026-10-26T12:00Z"), T("2026-10-27T03:00Z"))
    assert len(calls) == 4


def test_budget_guard_drops_lowest_priority_offsets_first():
    cfg = osch.ScheduleConfig(reserve_credits=0)
    now = T("2026-10-26T12:00Z")
    assert osch.affordable_offsets(GAMES, cfg, now, credits_remaining=100) == (1440, 360, 60, 15)
    assert osch.affordable_offsets(GAMES, cfg, now, credits_remaining=4) == (60, 15)
    assert osch.affordable_offsets(GAMES, cfg, now, credits_remaining=0) == ()


def test_plan_reports_calls_and_credits():
    p = osch.plan(GAMES, osch.ScheduleConfig(), T("2026-10-25T00:00Z"), T("2026-10-27T03:00Z"))
    assert p["n_games"] == 2 and p["credits"] == p["n_calls"] > 0


# ------------------------------------------------------------------ provider retries and the collector
def test_retry_backs_off_exponentially_then_succeeds():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise odds_api.OddsApiError("rate limited", 429)
        return "ok"

    assert odds_api.with_retries(flaky, retries=4, base_delay=2, sleep=sleeps.append) == "ok"
    assert sleeps == [2, 4]


def test_no_retry_on_auth_errors():
    sleeps = []
    with pytest.raises(odds_api.OddsApiError):
        odds_api.with_retries(lambda: (_ for _ in ()).throw(odds_api.OddsApiError("bad key", 401)),
                              sleep=sleeps.append)
    assert sleeps == []


def test_parse_american_keeps_quoted_prices():
    ev = [{"id": "e", "commence_time": "2026-10-26T23:00:00Z", "home_team": "Charlotte Hornets",
           "away_team": "Washington Wizards", "bookmakers": [{"key": "dk", "markets": [{"key": "h2h", "outcomes": [
               {"name": "Charlotte Hornets", "price": -150}, {"name": "Washington Wizards", "price": 130}]}]}]}]
    r = odds_api.parse_events(ev, "2026-10-26T18:00:00Z", "american").iloc[0]
    assert (r.home_american, r.away_american) == (-150, 130)
    assert r.home_price == pytest.approx(5 / 3) and r.away_price == pytest.approx(2.3)


def _events_file(tmp_path, price_home=1.60):
    ev = [{"id": "ev1", "commence_time": "2026-10-26T23:00:00Z", "home_team": "Charlotte Hornets",
           "away_team": "Washington Wizards", "bookmakers": [{"key": "draftkings", "markets": [{
               "key": "h2h", "last_update": "2026-10-26T21:55:00Z",
               "outcomes": [{"name": "Charlotte Hornets", "price": price_home},
                            {"name": "Washington Wizards", "price": 2.40}]}]}]}]
    p = tmp_path / "ev.json"
    p.write_text(json.dumps(ev))
    return p


def test_collector_skips_when_nothing_due_and_logs_runs(tmp_path):
    con = store.connect(tmp_path / "p.db")
    cfg = osch.ScheduleConfig(offsets_minutes=(60, 15))
    prov = odds_collect.FileProvider(_events_file(tmp_path), "2026-10-26T22:00:00Z")
    r = odds_collect.collect(con, prov, GAMES, cfg, now=T("2026-10-26T21:30Z"))
    assert r["status"] == "skipped"
    r = odds_collect.collect(con, prov, GAMES, cfg, now=T("2026-10-26T22:00Z"))
    assert r["status"] == "ok" and r["n_inserted"] == 1 and r["targets_due"] == 1
    r = odds_collect.collect(con, prov, GAMES, cfg, now=T("2026-10-26T22:05Z"))
    assert r["status"] == "skipped"                     # same target already satisfied
    runs = con.execute("SELECT status FROM odds_collection_runs").fetchall()
    assert runs == [("ok",)]


def test_collector_records_provider_failure_without_rows(tmp_path):
    class Down:
        name = "down"

        def fetch_current(self):
            raise odds_api.OddsApiError("HTTP 503", 503)

    con = store.connect(tmp_path / "p.db")
    r = odds_collect.collect(con, Down(), GAMES, osch.ScheduleConfig(), now=T("2026-10-26T22:00Z"), force=True)
    assert r["status"] == "error"
    assert con.execute("SELECT status, error FROM odds_collection_runs").fetchone() == ("error", "HTTP 503")
    assert con.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0] == 0
