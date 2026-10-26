"""Integration test: every Phase 2 command on a deterministic SYNTHETIC dataset."""
import json

import pandas as pd
import pytest

from nbamodel import cli, registry, store


@pytest.fixture(scope="module")
def env(tmp_path_factory, multi_season_raw):
    d = tmp_path_factory.mktemp("phase2")
    multi_season_raw.to_csv(d / "games.csv", index=False)
    cfg = json.loads(open("config/phase2.json").read())
    cfg["paths"] = {k: str(d / k) for k in cfg["paths"]}
    cfg["paths"].update(games=str(d / "games.csv"), odds=None, book="draftkings", db=str(d / "p.db"),
                        dataset=str(d / "ds.csv"), final_test_log=str(d / "ft.jsonl"),
                        upcoming=str(d / "up.csv"), dashboard=str(d / "dash.html"))
    cfg["walk_forward"].update(retrain="season", calibration_dates=15, final_test_start="2022-08-01")
    cfg["evaluation"]["n_boot"] = 50
    (d / "cfg.json").write_text(json.dumps(cfg))
    return d, cfg


def run(env, *args):
    d, _ = env
    return cli.main(["--config", str(d / "cfg.json"), *args])


def test_full_workflow(env, capsys):
    d, cfg = env
    assert run(env, "build-dataset") == 0
    assert pd.read_csv(cfg["paths"]["dataset"])["game_key"].is_unique

    assert run(env, "walk-forward") == 0
    rep = json.loads((d / "walkforward_dir" / "report.json").read_text())
    assert rep["champion"]["champion"] in rep["models"]
    assert rep["leakage_check"]["passed"] and rep["config"]["retrain"] == "season"
    for fig in rep["figures"].values():
        assert (d / "walkforward_dir" / fig.split("/")[-1]).exists()
    # Synthetic quotes are stamped 20:00Z, before the 22:30Z decision time, so they are usable.
    assert rep["paper_trading"]["status"] == "ok"

    assert run(env, "final-test") == 0
    ft = json.loads((d / "final_test_dir" / "report.json").read_text())
    assert ft["final_test_runs"] == 1 and ft["final_test_warnings"] == []
    assert run(env, "final-test") == 0
    ft = json.loads((d / "final_test_dir" / "report.json").read_text())
    assert ft["final_test_runs"] == 2 and "previously run" in ft["final_test_warnings"][0]

    assert run(env, "train") == 0
    version = registry.active_version(cfg["paths"]["artifacts"])
    _, meta = registry.load_model(version, cfg["paths"]["artifacts"])
    assert meta["reference_metrics"]["log_loss"] > 0 and meta["features"]

    capsys.readouterr()
    first_sched = pd.read_csv(cfg["paths"]["games"]).query("home_score != home_score")["date"].min()
    assert run(env, "predict", "--days", "1", "--as-of", f"{first_sched}T12:00:00") == 0
    up = pd.read_csv(cfg["paths"]["upcoming"])
    assert len(up) > 0 and up["p_home"].between(0, 1).all()
    assert (up["p_home"] + up["p_away"]).round(9).eq(1).all()
    assert "fair" in capsys.readouterr().out.lower()

    # Simulate the games finishing: give the scheduled rows scores and reconcile.
    raw = pd.read_csv(cfg["paths"]["games"])
    sched = raw["home_score"].isna()
    raw.loc[sched, "home_score"], raw.loc[sched, "away_score"] = 105, 99
    raw.to_csv(cfg["paths"]["games"], index=False)
    assert run(env, "update-results") == 0
    with store.connect(cfg["paths"]["db"]) as con:
        n_scored = con.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        n_elo = con.execute("SELECT COUNT(*) FROM elo_ratings").fetchone()[0]
    assert n_scored == len(up) and n_elo == 30

    assert run(env, "dashboard") == 0
    page = open(cfg["paths"]["dashboard"]).read()
    assert "SIMULATED" in page and "Model overview" in page and "data:image/png;base64" in page


def test_bad_config_reports_error(tmp_path, capsys):
    assert cli.main(["--config", str(tmp_path / "missing.json"), "build-dataset"]) == 2
    assert "config" in capsys.readouterr().err
