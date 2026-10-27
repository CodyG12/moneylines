"""Integration: Phase 3 commands end to end on SYNTHETIC games with a replayed odds response.

The odds file is a hand-written fixture replayed through FileProvider into a temporary
database; nothing here calls the real API or touches the real archive.
"""
import json

import pandas as pd
import pytest

from nbamodel import cli, registry, store
from nbamodel.odds_api import TEAM_CODES

NAMES = {v: k for k, v in TEAM_CODES.items() if k != "LA Clippers"}


@pytest.fixture(scope="module")
def env(tmp_path_factory, multi_season_raw):
    d = tmp_path_factory.mktemp("phase3")
    raw = multi_season_raw.copy()
    codes = dict(zip(sorted(set(raw["home_team"]) | set(raw["away_team"])), sorted(NAMES)))
    raw[["home_team", "away_team"]] = raw[["home_team", "away_team"]].replace(codes)
    raw.to_csv(d / "games.csv", index=False)
    cfg = json.loads(open("config/phase2.json").read())
    cfg["paths"] = {k: str(d / k) for k in cfg["paths"]}
    cfg["paths"].update(games=str(d / "games.csv"), odds="db", book="draftkings", db=str(d / "p.db"),
                        dataset=str(d / "ds.csv"), final_test_log=str(d / "ft.jsonl"), upcoming=str(d / "up.csv"),
                        dashboard=str(d / "dash.html"), blend=str(d / "blend.json"), blend_dir=str(d / "blend"))
    cfg["walk_forward"].update(retrain="season", calibration_dates=8, final_test_start="2022-08-01",
                               min_train_seasons=1)
    cfg["evaluation"]["n_boot"] = 30
    cfg["paper"].update(decision_min_lead_minutes=0, decision_max_lead_minutes=24 * 60, ev_threshold=-1.0)
    (d / "cfg.json").write_text(json.dumps(cfg))
    return d, cfg


def run(env, *args):
    d, _ = env
    return cli.main(["--config", str(d / "cfg.json"), *args])


def _odds_file(d, games: pd.DataFrame):
    """Provider-style payload for the given scheduled games, using full team names like the real API."""
    ev = []
    for i, g in enumerate(games.itertuples()):
        h, a = NAMES.get(g.home_team, g.home_team), NAMES.get(g.away_team, g.away_team)
        ev.append({"id": f"e{i}", "commence_time": g.tipoff_utc, "home_team": h, "away_team": a,
                   "bookmakers": [{"key": "draftkings", "markets": [{"key": "h2h", "outcomes": [
                       {"name": h, "price": 1.80}, {"name": a, "price": 2.05}]}]}]})
    p = d / "odds.json"
    p.write_text(json.dumps(ev))
    return p


def test_phase3_workflow(env, capsys):
    d, cfg = env
    assert run(env, "walk-forward") == 0 and run(env, "train") == 0
    assert run(env, "blend-study") == 0
    b = json.loads(open(cfg["paths"]["blend"]).read())
    assert set(b) >= {"blend_id", "base_col", "final_params", "enabled"}

    raw = pd.read_csv(cfg["paths"]["games"])
    sched = raw[raw["home_score"].isna()]
    first_day = sched["date"].min()
    day = sched[sched["date"] == first_day]
    tip = pd.Timestamp(day["tipoff_utc"].iloc[0])
    obs = (tip - pd.Timedelta(hours=1)).isoformat()
    assert run(env, "collect-odds", "--from-file", str(_odds_file(d, day)), "--observed-at", obs) == 0
    with store.connect(cfg["paths"]["db"]) as con:
        n_snap = con.execute("SELECT COUNT(*) FROM odds_snapshots WHERE match_status='matched'").fetchone()[0]
    known = day["home_team"].isin(NAMES) & day["away_team"].isin(NAMES)
    assert n_snap == int(known.sum())

    capsys.readouterr()
    as_of = (tip - pd.Timedelta(minutes=50)).isoformat()
    assert run(env, "predict", "--days", "1", "--as-of", as_of) == 0
    with store.connect(cfg["paths"]["db"]) as con:
        preds = pd.read_sql("SELECT * FROM predictions", con)
        po = pd.read_sql("SELECT * FROM prediction_odds", con)
    assert preds["p_elo"].notna().all() and preds["p_blend"].notna().all()
    assert len(po) == n_snap and (pd.to_datetime(po["observed_at"]) <= pd.Timestamp(as_of)).all()
    version = registry.active_version(cfg["paths"]["artifacts"])
    assert set(preds["model_version"]) == {version}

    # --as-of replays never create paper decisions; the paper command does, from the logged predictions
    assert run(env, "paper-trade", "--now", as_of) == 0
    with store.connect(cfg["paths"]["db"]) as con:
        dec = pd.read_sql("SELECT * FROM paper_decisions", con)
    assert len(dec) == len(preds["game_key"].unique())
    assert set(dec.loc[dec["decision"] == "bet", "game_key"]) <= set(po.merge(preds, left_on="prediction_id",
                                                                            right_on="id")["game_key"])

    raw.loc[raw["home_score"].isna(), ["home_score", "away_score"]] = [101, 99]
    raw.to_csv(cfg["paths"]["games"], index=False)
    assert run(env, "update-results") == 0
    out = capsys.readouterr().out
    assert "paper ledger: settled" in out
    assert run(env, "market-benchmark") == 0 and run(env, "odds-status") == 0 and run(env, "odds-plan") == 0
    assert run(env, "dashboard") == 0
    page = open(cfg["paths"]["dashboard"]).read()
    for heading in ("Odds monitoring", "Model versus market", "Paper trading", "Early-season Elo blending"):
        assert heading in page
    assert "SIMULATED" in page and "LIVE PREDICTIONS" in page and "HISTORICAL EVALUATION" in page


def test_collect_without_key_explains(env, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    d, _ = env
    assert cli.main(["--config", str(d / "cfg.json"), "collect-odds", "--force"]) == 2
    assert "the-odds-api.com" in capsys.readouterr().err
