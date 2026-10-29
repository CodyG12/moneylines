"""Phase 2 command line: `python -m nbamodel <command>`.

  build-dataset    write the leakage-checked historical training dataset
  walk-forward     development walk-forward backtest: metrics, plots, champion choice
  final-test       one-time evaluation on the untouched final period (logged)
  evaluate         re-score any predictions CSV
  train            fit the champion on all completed games, save a versioned artifact
  predict          predict upcoming games with the active model and log them
  update-results   reconcile results, update metrics/Elo, detect degradation, retrain if due
  dashboard        build the static HTML dashboard

Phase 3 (odds, market benchmark, blending, paper trading):
  collect-odds     store a timestamped odds snapshot when a scheduled target is due (run from cron)
  odds-plan        API credit estimate for the snapshot schedule
  odds-status      collection runs and per-game snapshot coverage
  import-odds      import an odds CSV into the snapshot archive
  blend-study      early-season Elo blending, weights chosen on earlier seasons only
  market-benchmark models vs the no-vig market at matching prediction times
  paper-trade      paper decisions and the simulated ledger (never real bets)
  today            game-day workflow: refresh stale odds, predict a date, log and report
  weekly-report    Markdown weekly report: accuracy, Brier, log loss, calibration, market, paper

Settings come from config/phase2.json (override with --config); paths can be
overridden per command. Secrets (ODDS_API_KEY) come from the environment or .env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from . import blend, clv, espn, injuries, market, odds_collect, odds_schedule, odds_store, paper, plots, registry, store
from .backtest import BettingConfig, paper_trade
from .data import DataError, prepare_games
from .dataset import DatasetConfig, build_training_dataset
from .elo import EloConfig, compute_elo
from .evaluation import compare_log_loss, expected_calibration_error, per_game_log_loss
from .features import FEATURES_V2, build_features
from .leakage import find_leaking_features, suspicious_label_correlation
from .odds_join import attach_odds_with_close
from .production import predict_upcoming, train_production
from .report import (champion_spec, choose_champion, evaluate_predictions, prob_columns, summary_text, tables)
from .walkforward import WalkForwardConfig, run_walk_forward

log = logging.getLogger("nbamodel")
DEFAULT_CONFIG = "config/phase2.json"


# ------------------------------------------------------------------ config
def load_config(path) -> dict:
    p = Path(path)
    if not p.exists():
        raise DataError(f"config file not found: {path}")
    return json.loads(p.read_text())


def ds_config(cfg: dict) -> DatasetConfig:
    d = cfg["dataset"]
    return DatasetConfig(elo=EloConfig(**d["elo"]), window=d["window"], lead_minutes=d["lead_minutes"],
                         min_history=d["min_history"])


def wf_config(cfg: dict, **overrides) -> WalkForwardConfig:
    allowed = {f.name for f in fields(WalkForwardConfig)}
    merged = {**cfg["walk_forward"], **{k: v for k, v in overrides.items() if v is not None}}
    kw = {k: v for k, v in merged.items() if k in allowed}
    return WalkForwardConfig(**kw)


def config_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_games_from(cfg: dict, games_path=None, odds_path=None, book=None):
    games_path = games_path or cfg["paths"]["games"]
    odds_path = odds_path if odds_path is not None else cfg["paths"].get("odds")
    try:
        raw = pd.read_csv(games_path, comment="#", dtype={"game_id_source": str})
    except FileNotFoundError as exc:
        raise DataError(f"games file not found: {games_path}") from exc
    if odds_path == "db":  # snapshots archived by collect-odds (pregame, matched rows only)
        with store.connect(cfg["paths"]["db"]) as con:
            snaps = odds_store.as_join_frame(odds_store.load_snapshots(con))
        if not snaps.empty:
            raw = attach_odds_with_close(raw, snaps, book=book or cfg["paths"]["book"],
                                         lead_minutes=cfg["dataset"]["lead_minutes"])
    elif odds_path:
        try:
            snaps = pd.read_csv(odds_path)
        except FileNotFoundError as exc:
            raise DataError(f"odds file not found: {odds_path}") from exc
        raw = attach_odds_with_close(raw, snaps, book=book or cfg["paths"]["book"],
                                     lead_minutes=cfg["dataset"]["lead_minutes"])
    games, report = prepare_games(raw)
    for w in report.warnings:
        log.warning(w)
    return games, report, games_path


def _write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ------------------------------------------------------------------ commands
def cmd_build_dataset(args, cfg):
    games, report, _ = load_games_from(cfg, args.games, args.odds, args.book)
    dsc = ds_config(cfg)
    ds = build_training_dataset(games, dsc)
    leaks = find_leaking_features(games, lambda g: build_features(g, dsc.elo, dsc.window), FEATURES_V2)
    suspicious = suspicious_label_correlation(ds, FEATURES_V2)
    if leaks or suspicious:
        raise DataError(f"leakage check failed: {leaks} {suspicious}")
    out = Path(args.out or cfg["paths"]["dataset"])
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(out, index=False, float_format="%.6f")
    played = ds[ds["played"]]
    print(f"wrote {len(ds)} rows ({len(played)} played, {int(ds['train_eligible'].sum())} train-eligible, "
          f"{int((~ds['played']).sum())} scheduled) to {out}")
    n_box = int(played["home_fga"].notna().sum()) if "home_fga" in played else 0
    print(f"seasons {played['season'].min()}-{played['season'].max()}; box scores for "
          f"{n_box} played games; tip-off times for "
          f"{int(ds['tipoff_utc'].notna().sum())} rows")
    print("leakage check: passed (no feature changed when same-day/future results were flipped)")
    return 0


def _evaluate_and_write(preds, outdir: Path, cfg, title: str, importance=None, periods=None, extra=None):
    ev = cfg["evaluation"]
    report = evaluate_predictions(preds, ev["n_boot"], ev["seed"])
    tabs = tables(preds)
    cols = prob_columns(preds)
    cal_by_season = pd.DataFrame([{"season": s, **{f"{c}_ece": expected_calibration_error(g["home_win"], g[c])
                                                   for c in cols}} for s, g in preds.groupby("season")])
    tabs["calibration_by_season"] = cal_by_season
    outdir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(outdir / "predictions.csv", index=False, float_format="%.6f")
    if periods is not None:
        periods.to_csv(outdir / "periods.csv", index=False)
    for name, t in tabs.items():
        t.to_csv(outdir / f"{name}.csv", index=False, float_format="%.6f")
    figs = {
        "reliability": plots.reliability_diagram(preds, outdir / "reliability.png", f"Reliability: {title}"),
        "distribution": plots.probability_distribution(preds, outdir / "probability_distribution.png"),
        "log_loss_by_season": plots.metric_by_season(tabs["by_season"], outdir / "log_loss_by_season.png"),
        "brier_by_season": plots.metric_by_season(tabs["by_season"], outdir / "brier_by_season.png", "brier"),
        "rolling": plots.rolling_log_loss(preds, outdir / "rolling_log_loss.png"),
        "scores": plots.score_comparison(report, outdir / "score_comparison.png"),
        "calibration_drift": plots.calibration_drift(cal_by_season, outdir / "calibration_drift.png"),
    }
    report.update(extra or {})
    if importance:
        report["feature_importance"] = importance
        for k, v in importance.items():
            if v:
                figs[k] = plots.feature_importance(v, outdir / f"{k}.png", k.replace("_", " ").capitalize())
    report["figures"] = figs
    return report


def _paper_trading(preds, cfg, outdir: Path, p_col: str):
    b = cfg["betting"]
    bets, summary = paper_trade(preds, p_col, BettingConfig(**b), cfg["evaluation"]["n_boot"], cfg["evaluation"]["seed"])
    if bets is not None and not bets.empty:
        bets.to_csv(outdir / "paper_bets.csv", index=False, float_format="%.6f")
        summary["figure"] = plots.bankroll_curve(bets, outdir / "bankroll.png", b["initial_bankroll"])
    return summary


def cmd_walk_forward(args, cfg):
    games, report_data, games_path = load_games_from(cfg, args.games, args.odds, args.book)
    dsc = ds_config(cfg)
    wf = wf_config(cfg, retrain=args.retrain, window=args.window, calibration=args.calibration)
    outdir = Path(args.out or cfg["paths"]["walkforward_dir"])
    leaks = find_leaking_features(games, lambda g: build_features(g, dsc.elo, dsc.window), wf.features)
    if leaks:
        raise DataError(f"leakage check failed: {leaks}")
    result = run_walk_forward(games, wf, dsc, phase="development")
    rep = _evaluate_and_write(result.predictions, outdir, cfg, "development walk-forward", result.importance,
                              result.periods)
    rep["champion"] = choose_champion(rep)
    champ = rep["champion"]["champion"]
    ll = per_game_log_loss(result.predictions["home_win"], result.predictions[champ])
    rep["champion"]["reference"] = {"log_loss": float(ll.mean()), "log_loss_sd": float(ll.std(ddof=1)),
                                    "brier": rep["models"][champ]["brier"], "n": int(len(ll))}
    rep["champion"]["spec"] = champion_spec(champ)
    p = result.predictions
    rep["champion"]["vs_others_log_loss"] = {
        other: compare_log_loss(p["home_win"], p[champ], p[other], p["date"], cfg["evaluation"]["n_boot"],
                                cfg["evaluation"]["seed"])
        for other in prob_columns(p) if other != champ}
    rep["paper_trading"] = _paper_trading(result.predictions, cfg, outdir, champ)
    rep["config"] = result.config
    rep["config_hash"] = config_hash(result.config)
    rep["leakage_check"] = {"passed": True, "method": "flip all results on/after 3 cutoffs; features on each "
                                                      "cutoff date must not change"}
    rep["data"] = {"games_file": games_path, "games_sha256": registry.file_sha256(games_path),
                   "warnings": report_data.warnings}
    rep["periods_summary"] = {"n_periods": int(len(result.periods)), "retrain": wf.retrain, "window": wf.window}
    _write_json(outdir / "report.json", rep)
    text = summary_text(rep, "Development walk-forward (out-of-sample, chronological)")
    text += f"\n  champion (by development log loss): {champ}"
    if rep["champion"].get("note"):
        text += f"  [{rep['champion']['note']}]"
    pt = rep["paper_trading"]
    text += f"\n  paper trading: {pt.get('status')} (decisions {pt.get('decisions')})"
    (outdir / "summary.txt").write_text(text + "\n")
    print(text)
    print(f"\nWrote {outdir}/ (report.json, predictions.csv, periods.csv, tables, plots)")
    return 0


def cmd_final_test(args, cfg):
    dev_dir = Path(cfg["paths"]["walkforward_dir"])
    dev_report = dev_dir / "report.json"
    if not dev_report.exists():
        raise DataError("run `walk-forward` first: the final test uses the champion chosen on development data")
    dev = json.loads(dev_report.read_text())
    games, _, games_path = load_games_from(cfg, args.games, args.odds, args.book)
    dsc = ds_config(cfg)
    wf = wf_config(cfg)
    result = run_walk_forward(games, wf, dsc, phase="final_test")
    h = config_hash(result.config)
    log_path = Path(cfg["paths"]["final_test_log"])
    prior = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    entry = {"run_at": pd.Timestamp.now(tz="UTC").isoformat(), "config_hash": h, "dev_config_hash": dev["config_hash"],
             "champion": dev["champion"]["champion"], "games_sha256": registry.file_sha256(games_path)}
    warnings = []
    if dev["config_hash"] != config_hash({**result.config, "phase": "development"}):
        warnings.append("final-test configuration differs from the development configuration")
    if any(p["config_hash"] != h or p["champion"] != entry["champion"] for p in prior):
        warnings.append(f"the final test was already run {len(prior)} time(s) with a different configuration or "
                        "champion: it is no longer an untouched holdout, and its results are optimistic")
    elif prior:
        warnings.append(f"final test previously run {len(prior)} time(s) with the same configuration "
                        "(re-runs with new games are fine; re-tuning is not)")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    outdir = Path(args.out or cfg["paths"]["final_test_dir"])
    champ = dev["champion"]["champion"]
    rep = _evaluate_and_write(result.predictions, outdir, cfg, "final holdout", None, result.periods,
                              {"champion_from_development": champ, "final_test_warnings": warnings,
                               "final_test_runs": len(prior) + 1, "config": result.config, "config_hash": h})
    rep["paper_trading"] = _paper_trading(result.predictions, cfg, outdir, champ)
    _write_json(outdir / "report.json", rep)
    text = summary_text(rep, "FINAL HOLDOUT (untouched chronological test)")
    text += f"\n  champion fixed in advance from development: {champ}"
    for w in warnings:
        text += f"\n  WARNING: {w}"
    (outdir / "summary.txt").write_text(text + "\n")
    print(text)
    print(f"\nWrote {outdir}/")
    return 0


def cmd_evaluate(args, cfg):
    preds = pd.read_csv(args.predictions, parse_dates=["date"])
    rep = _evaluate_and_write(preds, Path(args.out), cfg, Path(args.predictions).stem)
    _write_json(Path(args.out) / "report.json", rep)
    print(summary_text(rep, f"Evaluation of {args.predictions}"))
    return 0


def _train(cfg, games, games_path, model_override=None, reason="manual"):
    dev_path = Path(cfg["paths"]["walkforward_dir"]) / "report.json"
    reference = {}
    if dev_path.exists():
        dev = json.loads(dev_path.read_text())
        spec = dev["champion"]["spec"]
        reference = {**dev["champion"]["reference"], "source": str(dev_path), "champion": dev["champion"]["champion"]}
    else:
        spec = {"model": "logistic", "calibrated": True}
        log.warning("no walk-forward report found; defaulting to calibrated logistic without reference metrics")
    if model_override:
        name = model_override.replace("_raw", "")
        spec = {"model": name, "calibrated": not model_override.endswith("_raw")}
        if reference.get("champion") != f"p_{model_override}":
            reference = {}
    model, meta = train_production(games, spec, wf_config(cfg), ds_config(cfg), reference, games_path)
    meta["experimental"] = model_override is not None and not reference
    d = registry.save_model(model, meta, cfg["paths"]["artifacts"])
    with store.connect(cfg["paths"]["db"]) as con:
        store.record_training_run(con, meta, reason)
    return meta, d


def cmd_train(args, cfg):
    games, _, games_path = load_games_from(cfg, args.games, args.odds, args.book)
    meta, d = _train(cfg, games, games_path, args.model, "manual")
    t = meta["training"]
    print(f"trained {meta['version']} on {t['n_rows']} games ({t['start']} -> {t['end']}); "
          f"calibration window from {t['calibration_start']} ({t['n_calibration']} games)")
    print(f"saved {d} and set it active")
    return 0


def _uses_odds_db(args, cfg) -> bool:
    return (args.odds or cfg["paths"].get("odds")) == "db"


def _add_shadow_blend(preds: pd.DataFrame, games: pd.DataFrame, cfg) -> pd.DataFrame:
    """p_blend from the stored blend study (shadow: logged for evaluation, published only if enabled)."""
    path = Path(cfg["paths"].get("blend", "artifacts/blend.json"))
    if preds.empty or not path.exists():
        return preds
    b = json.loads(path.read_text())
    if b["base_col"] not in preds or "p_elo" not in preds:
        return preds
    first = games.groupby("season")["date"].min()
    preds = preds.assign(season_day=(pd.to_datetime(preds["date"]) - preds["season"].map(first)).dt.days)
    preds["p_blend"] = blend.apply(preds, b["final_params"], b["base_col"])
    if b.get("enabled"):
        log.warning("blend %s is enabled; publishing p_blend as p_home", b["blend_id"])
        preds["p_home"], preds["p_away"] = preds["p_blend"], 1 - preds["p_blend"]
        preds["model_version"] = preds["model_version"] + "+" + b["blend_id"]
    return preds.drop(columns="season_day")


def cmd_predict(args, cfg):
    use_db = _uses_odds_db(args, cfg)
    games, _, _ = load_games_from(cfg, args.games, None if use_db else args.odds, args.book)
    model, meta = registry.load_model(args.version, cfg["paths"]["artifacts"])
    now = pd.Timestamp(args.as_of, tz="UTC") if args.as_of else None
    preds = predict_upcoming(games, model, meta, args.days or cfg["predict"]["horizon_days"], now)
    if preds.empty:
        print("no upcoming scheduled games in the data (run scripts/update_schedule.py)")
        return 0
    preds = _add_shadow_blend(preds, games, cfg)
    book = args.book or cfg["paths"]["book"]
    quotes = None
    with store.connect(cfg["paths"]["db"]) as con:
        if use_db:  # prices as of this prediction's own timestamp, never later
            preds, quotes = market.attach_market_asof(preds, odds_store.load_snapshots(con), book,
                                                      cfg.get("odds", {}).get("max_age_minutes", 120))
        out = Path(args.out or cfg["paths"]["upcoming"])
        out.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out, index=False, float_format="%.6f")
        n = store.log_predictions(con, preds)
        n_q = store.log_prediction_odds(con, preds, quotes)
        decided = pd.DataFrame()
        pcfg = cfg.get("paper", {})
        if pcfg.get("enabled") and not args.as_of:
            decided = paper.decide(con, paper.PaperConfig.from_dict(pcfg), model_version=preds["model_version"].iloc[0])
    print(f"model {meta['version']}; data through {preds['latest_result_in_data'].iloc[0]}")
    print(f"{'date':<11} {'game':<11} {'p(home)':>7} {'fair':>11} {'spread':>6}  market / notes")
    for _, r in preds.iterrows():
        line = (f"{r['date'].date()!s:<11} {r['away_team']:>3} @ {r['home_team']:<3}  {r['p_home']:>7.3f} "
                f"{r['fair_home_american']:+5d}/{r['fair_away_american']:+5d} {r['model_spread']:>6.3f}")
        if pd.notna(r["home_odds"]):
            line += (f"  {r['odds_book']} {r['home_odds']:.2f}/{r['away_odds']:.2f} no-vig "
                     f"{r['market_home_prob_novig']:.3f} edge {r['edge_home']:+.3f}")
        elif use_db:
            line += "  [no price at prediction time]"
        if r["low_history"]:
            line += "  [low history]"
        print(line)
    print(f"\nlogged {n} new predictions ({n_q} book prices recorded with them) to {cfg['paths']['db']}; wrote {out}")
    if not decided.empty:
        print("paper decisions (SIMULATED, no real bets): " + ", ".join(
            f"{k} {v}" for k, v in decided["decision"].value_counts().items()))
    print("Fair odds are the model's zero-margin prices, not bookmaker prices. A positive estimated edge is "
          "not a guarantee of profit.")
    return 0


def data_quality(con, games: pd.DataFrame, cfg, now: pd.Timestamp, days: int = 3) -> list:
    """Missing-data checks over games that tipped off in the last `days` days."""
    book = cfg.get("paper", {}).get("book", cfg["paths"]["book"])
    tip = pd.to_datetime(games["tipoff_utc"], utc=True, errors="coerce")
    recent = games[(tip <= now) & (tip >= now - pd.Timedelta(days=days))]
    issues = []
    if recent.empty:
        return issues
    keys = set(recent["game_key"])
    snap = odds_store.load_snapshots(con, book=book)
    no_odds = sorted(keys - set(snap["game_key"]))
    if no_odds:
        issues.append(f"{len(no_odds)} of {len(keys)} recent games have no pregame {book} snapshot")
    preds = pd.read_sql("SELECT DISTINCT game_key FROM predictions", con)
    no_pred = sorted(keys - set(preds["game_key"]))
    if no_pred:
        issues.append(f"{len(no_pred)} of {len(keys)} recent games were never predicted")
    dec = pd.read_sql("SELECT game_key FROM paper_decisions", con)
    if cfg.get("paper", {}).get("enabled"):
        no_dec = sorted((keys - set(no_pred)) - set(dec["game_key"]))
        if no_dec:
            issues.append(f"{len(no_dec)} predicted recent games have no paper decision (no prediction inside "
                          "the decision window)")
    errs = con.execute("SELECT COUNT(*) FROM odds_collection_runs WHERE status='error' AND started_at >= ?",
                       ((now - pd.Timedelta(days=1)).isoformat(),)).fetchone()[0]
    if errs:
        issues.append(f"{errs} odds collection run(s) failed in the last 24h")
    for msg in issues:
        seen = con.execute("SELECT 1 FROM alerts WHERE kind='data_quality' AND message=? AND created_at >= ?",
                           (msg, (now - pd.Timedelta(hours=20)).isoformat())).fetchone()
        if not seen:
            con.execute("INSERT INTO alerts (created_at, kind, model_version, message, details_json) "
                        "VALUES (?,?,?,?,?)", (now.isoformat(), "data_quality", None, msg, None))
    con.commit()
    return issues


def cmd_update_results(args, cfg):
    games, _, games_path = load_games_from(cfg, args.games, args.odds, args.book)
    fb = cfg["feedback"]
    now = pd.Timestamp.now(tz="UTC")
    with store.connect(cfg["paths"]["db"]) as con:
        remapped = odds_store.remap_unmatched(con, games)
        rec = store.reconcile(con, games)
        metrics = store.snapshot_metrics(con)
        _, ratings = compute_elo(games, ds_config(cfg).elo)
        store.snapshot_elo(con, ratings)
        print(f"new results recorded for {rec['results_recorded']} predicted games; scored {rec['predictions_scored']} "
              f"predictions ({rec['late_predictions']} made after tip-off, excluded from metrics)")
        if remapped:
            print(f"matched {remapped} previously unmatched odds snapshots to games")
        if not metrics.empty:
            print(metrics[["model_version", "window", "n", "log_loss", "brier", "accuracy", "ece"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        pcfg = paper.PaperConfig.from_dict(cfg.get("paper"))
        settled = paper.resolve(con, now)
        n_clv = clv.record(con, pcfg.close_window_minutes, now)
        mkt = market.snapshot_market_metrics(con, pcfg.book, now)
        print(f"paper ledger: settled {settled} decisions; CLV computed for {n_clv}"
              + (f"; market ({pcfg.book}) log loss {mkt['log_loss'].iloc[0]:.4f} on {int(mkt['n'].iloc[0])} "
                 "aligned games" if not mkt.empty else ""))
        for msg in data_quality(con, games, cfg, now):
            print(f"data check: {msg}")
        version = registry.active_version(cfg["paths"]["artifacts"])
        degradation, reason = None, None
        if version:
            _, meta = registry.load_model(version, cfg["paths"]["artifacts"])
            degradation = store.detect_degradation(con, version, meta.get("reference_metrics", {}),
                                                   fb["degradation_min_n"], fb["degradation_window"],
                                                   fb["degradation_z"])
            if degradation:
                print(f"degradation check: recent {degradation['recent_log_loss']:.4f} vs reference "
                      f"{degradation['reference_log_loss']:.4f} (n={degradation['n']}) -> "
                      f"{'DEGRADED' if degradation['degraded'] else 'ok'}")
            else:
                print(f"degradation check: waiting for {fb['degradation_min_n']} scored pregame predictions")
            new = int((games["played"] & (games["date"] > pd.Timestamp(meta["training"]["end"]))).sum())
            reason = store.retrain_due(con, fb["retrain_cadence_days"], degradation, new)
    if reason and args.retrain_if_due:
        meta, d = _train(cfg, games, games_path, None, reason)
        print(f"retrained ({reason}): {meta['version']} is now active. Its weights came from a full refit "
              "with the same validated configuration, not from adjustments to individual results.")
    elif reason:
        print(f"retraining is due ({reason}); rerun with --retrain-if-due or run `train`")
    return 0


# ------------------------------------------------------------------ Phase 3 commands
def _schedule(cfg) -> odds_schedule.ScheduleConfig:
    return odds_schedule.ScheduleConfig.from_dict(cfg.get("odds", {}).get("schedule"))


def _provider(args, cfg):
    if getattr(args, "from_file", None):
        if not args.observed_at:
            raise DataError("--from-file needs --observed-at (the time the saved response was observed)")
        return odds_collect.FileProvider(args.from_file, args.observed_at, args.file_format)
    from .odds_api import OddsApiError, load_api_key
    o = cfg.get("odds", {})
    try:
        key = load_api_key()
    except OddsApiError as exc:
        raise DataError(f"{exc}. Get a free key at https://the-odds-api.com and put ODDS_API_KEY=... in .env") from exc
    return odds_collect.TheOddsApi(key, o.get("regions", "us"), o.get("bookmakers"), o.get("retries", 4))


def cmd_collect_odds(args, cfg):
    games, _, _ = load_games_from(cfg, args.games, None)
    provider = _provider(args, cfg)
    now = pd.Timestamp(args.observed_at) if getattr(args, "from_file", None) else None
    with store.connect(cfg["paths"]["db"]) as con:
        res = odds_collect.collect(con, provider, games, _schedule(cfg), now=now, force=args.force,
                                   budget_guard=not args.no_budget_guard)
    if res["status"] == "skipped":
        print(f"no snapshot due (offsets in use: {res['offsets_in_use']}); nothing fetched, no credits used")
        return 0
    if res["status"] == "error":
        print(f"error: odds collection failed and was logged (run {res['run_id']}): {res['error']}", file=sys.stderr)
        return 3
    usage = res.get("usage") or {}
    print(f"run {res['run_id']}: {res['n_events']} events, {res['n_rows']} book prices -> {res['n_inserted']} stored, "
          f"{res['n_duplicates']} duplicates, {res['n_invalid']} invalid, {res['n_unmatched']} unmatched, "
          f"{res['n_live']} in-game (kept, never used); {res['targets_due']} target(s) due; "
          f"credits used {usage.get('x-requests-last')}, remaining {usage.get('x-requests-remaining')}")
    if res.get("due_games_without_prices"):
        print(f"warning: no prices for {len(res['due_games_without_prices'])} game(s) with due targets")
    if args.then_predict and res["n_inserted"]:
        ns = argparse.Namespace(games=args.games, odds=None, book=None, days=None, version=None, out=None,
                                as_of=None)
        return cmd_predict(ns, cfg)
    return 0


def cmd_odds_plan(args, cfg):
    games, _, _ = load_games_from(cfg, args.games, None)
    sc = _schedule(cfg)
    up = games[~games["played"]]
    start = pd.Timestamp(args.start, tz="UTC") if args.start else pd.Timestamp.now(tz="UTC")
    tip = pd.to_datetime(up["tipoff_utc"], utc=True, errors="coerce")
    if tip.notna().sum() == 0:
        print("no upcoming games with tip-off times (run scripts/update_schedule.py); nothing to plan")
        return 0
    end = pd.Timestamp(args.end, tz="UTC") if args.end else tip.max()
    print(f"schedule: {up['tipoff_utc'].notna().sum()} upcoming games with tip-off times; window {start:%Y-%m-%d} to "
          f"{end:%Y-%m-%d}; poll every {sc.poll_minutes} min; {sc.credits_per_call} credit/call; budget "
          f"{sc.monthly_credit_budget}/month (reserve {sc.reserve_credits})")
    offsets = list(sc.offsets_minutes)
    for k in range(len(offsets), 0, -1):
        keep = tuple(o for o in sc.offsets_minutes if o in sc.priority[:k])
        p = odds_schedule.plan(up, sc, start, end, keep)
        fits = p["max_calls_per_month"] * sc.credits_per_call <= sc.monthly_credit_budget - sc.reserve_credits
        print(f"  offsets {str(list(keep)):<22} calls {p['n_calls']:>5}  per 30 days {p['credits_per_30_days']:>6}  "
              f"busiest month {p['max_calls_per_month']:>4}  {'fits' if fits else 'OVER BUDGET'}")
    p = odds_schedule.plan(up, sc, start, end)
    print("calls by month (all offsets): " + ", ".join(f"{k} {v}" for k, v in p["calls_by_month"].items()))
    return 0


def cmd_odds_next(args, cfg):
    """Exit 0 when an unsatisfied snapshot target opens within --within-hours (the hosted collector keeps
    polling), 1 when none does (it can stop until the next scheduled start)."""
    games, _, _ = load_games_from(cfg, args.games, None)
    sc = _schedule(cfg)
    now = pd.Timestamp.now(tz="UTC")
    with store.connect(cfg["paths"]["db"]) as con:
        calls = odds_collect.call_times(con, cfg.get("odds", {}).get("provider", "the-odds-api"))
    tg = odds_schedule.targets_for(games[~games["played"]], sc)
    nxt = odds_schedule.next_target(tg, calls, now)
    if nxt is None:
        print("no unsatisfied snapshot target in the schedule")
        return 1
    hours = max(0.0, (nxt - now) / pd.Timedelta(hours=1))
    print(f"next snapshot target {nxt:%Y-%m-%d %H:%M} UTC ({hours:.1f} h)")
    return 0 if hours <= args.within_hours else 1


def cmd_results_update(args, cfg):
    """Results, tip-off times and player box scores from ESPN (see nbamodel/espn.py)."""
    path = args.games or cfg["paths"]["games"]
    raw = pd.read_csv(path, comment="#", dtype={"game_id_source": str})
    today = pd.Timestamp.now(tz=espn.ET).normalize().tz_localize(None)
    start = pd.Timestamp(args.start) if args.start else today - pd.Timedelta(days=args.days_back)
    end = today + pd.Timedelta(days=args.days_ahead)
    try:
        with store.connect(args.db or cfg["paths"]["db"]) as con:
            out, st = espn.update(raw, con, start, end)
    except espn.EspnError as exc:
        print(f"results update failed: {exc}", file=sys.stderr)
        return 1
    if st["updated"] or st["tipoff_changes"] or st["added"]:
        out = out[raw.columns.tolist() + [c for c in out.columns if c not in raw.columns]]
        out.sort_values(["date", "home_team"]).to_csv(path, index=False)
    print(f"ESPN {start.date()}..{end.date()}: {st['updated']} game(s) scored, {st['players']} player box rows, "
          f"{st['tipoff_changes']} tip-off change(s), {st['added']} game(s) added")
    for msg in st["conflicts"]:
        print(f"score conflict (file kept): {msg}")
    for msg in st["invalid"]:
        print(f"skipped, failed validation: {msg}")
    return 0


def cmd_players_backfill(args, cfg):
    raw = pd.read_csv(args.games or cfg["paths"]["games"], comment="#", dtype={"game_id_source": str})
    end = args.end or (pd.Timestamp.now(tz=espn.ET) - pd.Timedelta(days=1)).date().isoformat()
    try:
        with store.connect(args.db) as con:
            r = espn.backfill(con, raw, args.start, end, args.max_seconds, args.pause)
            n = con.execute("SELECT COUNT(*), COUNT(DISTINCT espn_event_id) FROM player_games").fetchone()
            bad = con.execute("SELECT COUNT(*) FROM espn_invalid").fetchone()[0]
    except espn.EspnError as exc:
        print(f"player backfill stopped: {exc}", file=sys.stderr)
        return 12
    state = "complete" if r["complete"] else f"paused at {r['next_day']}"
    print(f"player backfill {args.start}..{end}: {state}; this run stored {r['games']} game(s), {r['rows']} rows; "
          f"{args.db} holds {n[0]} rows over {n[1]} games ({bad} failed validation)")
    return 0 if r["complete"] else 10


def cmd_injuries_refresh(args, cfg):
    games, _, _ = load_games_from(cfg, args.games, None)
    with store.connect(args.db or cfg["paths"]["db"]) as con:
        try:
            res = injuries.refresh(con, games)
        except OSError as exc:  # season page unreachable: nothing recorded, retried next pass
            print(f"injury reports: refresh failed ({exc})")
            return 1
    counts = pd.Series([r["status"] for r in res]).value_counts().to_dict() if res else {}
    changes = sum(r.get("n_changes", 0) for r in res)
    print(f"injury reports: {len(res)} new file(s) {counts}; {changes} status change(s) recorded"
          if res else "injury reports: no new reports")
    return 0


def cmd_injuries_backfill(args, cfg):
    games, _, _ = load_games_from(cfg, args.games, None)
    end = args.end or (pd.Timestamp.now(tz=injuries.ET) - pd.Timedelta(days=1)).date().isoformat()
    with store.connect(args.db) as con:
        r = injuries.backfill(con, games, args.start, end, args.max_seconds, args.pause)
        n = con.execute("SELECT COUNT(*), COUNT(DISTINCT game_date) FROM injury_reports").fetchone()
    print(f"injury backfill {args.start}..{end}: {'complete' if r['complete'] else 'paused at ' + str(r['next_day'])}"
          f"; this run fetched {r['fetched']} URL(s), {r['found']} report(s) found; {args.db} now holds {n[0]} "
          f"change rows over {n[1]} game dates" + (" (stopped on a network error)" if r.get("error") else ""))
    return 0 if r["complete"] else (12 if r.get("error") else 10)   # 10: paused on time, resumable


def cmd_injuries_status(args, cfg):
    day = args.date or pd.Timestamp.now(tz=injuries.ET).date().isoformat()
    now = pd.Timestamp.now(tz="UTC")
    with store.connect(args.db or cfg["paths"]["db"]) as con:
        st = injuries.statuses_asof(con, now, [day])
        cov = injuries.report_coverage(con, now, day)
    if st.empty:
        print(f"no injury report ingested for {day} (run injuries-refresh)")
        return 0
    age = cov["report_age_minutes"]
    print(f"{day}: latest report {cov['report_published_at']} ({age:.0f} min old); not yet submitted: "
          f"{', '.join(cov['not_submitted']) or 'none'}")
    shown = st[st["status"] != injuries.NOT_SUBMITTED].sort_values(["matchup", "team", "status", "player_name"])
    print(shown[["matchup", "team", "player_name", "status", "reason"]].to_string(index=False))
    return 0


def cmd_odds_status(args, cfg):
    games, _, _ = load_games_from(cfg, args.games, None)
    now = pd.Timestamp.now(tz="UTC")
    with store.connect(cfg["paths"]["db"]) as con:
        runs = pd.read_sql("SELECT id, started_at, status, reason, n_events, n_inserted, n_invalid, n_unmatched, "
                           "n_live, credits_remaining, error FROM odds_collection_runs ORDER BY id DESC LIMIT 15", con)
        cov = odds_collect.coverage(con, games, _schedule(cfg), cfg.get("odds", {}).get("provider", "the-odds-api"),
                                    now - pd.Timedelta(days=args.days), now + pd.Timedelta(days=2))
        n = con.execute("SELECT COUNT(*), COUNT(DISTINCT game_key), MIN(observed_at), MAX(observed_at) "
                        "FROM odds_snapshots").fetchone()
    print(f"snapshot rows {n[0]} covering {n[1]} games; observed {n[2]} to {n[3]}")
    print("\nrecent collection runs:")
    print(runs.to_string(index=False) if not runs.empty else "  none yet")
    print("\ngame coverage:")
    print(cov.to_string(index=False) if not cov.empty else "  no games in range")
    return 0


def cmd_import_odds(args, cfg):
    """Import an odds CSV in the scripts/collect_odds.py layout into the snapshot archive."""
    rows = pd.read_csv(args.csv, dtype={"event_id": str})
    need = {"observed_at", "event_id", "commence_time", "home_team", "away_team", "book", "home_price", "away_price"}
    if not need <= set(rows.columns):
        raise DataError(f"odds CSV needs columns {sorted(need)}")
    games, _, _ = load_games_from(cfg, args.games, None)
    frame = odds_store.from_provider_frame(rows, args.provider, f"import:{Path(args.csv).name}",
                                           pd.Timestamp.now(tz="UTC").isoformat())
    with store.connect(cfg["paths"]["db"]) as con:
        run = odds_store.start_run(con, args.provider, "import")
        counts = odds_store.store_snapshots(con, frame, games, run)
        odds_store.finish_run(con, run, "ok", counts, reason=f"import {args.csv}")
    print(f"imported {counts['n_inserted']} of {counts['n_rows']} rows ({counts['n_duplicates']} duplicates, "
          f"{counts['n_invalid']} invalid {counts['invalid_reasons']}, {counts['n_unmatched']} unmatched, "
          f"{counts['n_live']} in-game)")
    return 0


def cmd_blend_study(args, cfg):
    ev = cfg["evaluation"]
    base = args.base or cfg.get("blend", {}).get("base_col", "p_logistic_raw")
    dev_path = Path(cfg["paths"]["walkforward_dir"]) / "predictions.csv"
    if not dev_path.exists():
        raise DataError("run `walk-forward` first: the blend study uses development walk-forward predictions")
    st = blend.study(pd.read_csv(dev_path), base, ev["n_boot"], ev["seed"])
    rep = st.report
    hold = Path(cfg["paths"]["final_test_dir"]) / "predictions.csv"
    if args.confirm_holdout and hold.exists():
        rep["holdout_confirmation"] = blend.confirm_on_holdout(pd.read_csv(hold), rep["final_params"], base,
                                                               ev["n_boot"], ev["seed"])
        log_path = Path(cfg["paths"]["final_test_log"])
        with open(log_path, "a") as fh:
            fh.write(json.dumps({"run_at": pd.Timestamp.now(tz="UTC").isoformat(), "kind": "blend_confirmation",
                                 "blend_id": rep["blend_id"], "params": rep["final_params"]}) + "\n")
    outdir = Path(args.out or cfg["paths"].get("blend_dir", "out/blend"))
    outdir.mkdir(parents=True, exist_ok=True)
    st.oos.to_csv(outdir / "oos_predictions.csv", index=False, float_format="%.6f")
    rep["figure"] = plots.blend_by_segment(rep, outdir / "blend_by_segment.png")
    _write_json(outdir / "report.json", rep)
    _write_json(cfg["paths"].get("blend", "artifacts/blend.json"),
                {k: rep[k] for k in ("blend_id", "base_col", "best_method", "final_params", "enabled", "decision_rule",
                                     "evaluated_seasons")})
    print(f"Elo blend study (base {base}); weights chosen on earlier seasons only, scored on seasons "
          f"{rep['evaluated_seasons'][0]}-{rep['evaluated_seasons'][-1]} ({rep['n']} games)")
    print(f"{'model':<24} {'log loss':>9} {'diff vs base [95% CI]':>32}   early-season diff [95% CI]")
    early = rep["by_segment"].get("early (<15 games)", {})
    for c, v in rep["overall"].items():
        d = v.get("log_loss_diff_vs_base")
        e = early.get(c, {}).get("log_loss_diff_vs_base")
        print(f"{c:<24} {v['log_loss']:>9.4f} {fmt_ci_short(d):>32}   {fmt_ci_short(e)}")
    print(f"best method: {rep['best_method']} {rep['final_params']}; ENABLED: {rep['enabled']}")
    print(f"rule: {rep['decision_rule']}. {rep['multiple_testing_note']}.")
    if rep.get("holdout_confirmation"):
        h = rep["holdout_confirmation"]["overall"]["p_blend"]["log_loss_diff_vs_base"]
        print(f"holdout confirmation (one look, nothing refitted): diff {fmt_ci_short(h)}")
    print(f"wrote {outdir}/ and {cfg['paths'].get('blend', 'artifacts/blend.json')}")
    return 0


def fmt_ci_short(d) -> str:
    if not d:
        return "-"
    return f"{d['estimate']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"


def cmd_market_benchmark(args, cfg):
    ev = cfg["evaluation"]
    book = args.book or cfg.get("paper", {}).get("book", cfg["paths"]["book"])
    with store.connect(cfg["paths"]["db"]) as con:
        live = market.live_benchmark(con, book, ev["n_boot"], ev["seed"])
    out = {"live": live}
    ft = Path(cfg["paths"]["final_test_dir"]) / "predictions.csv"
    if ft.exists():
        hp = pd.read_csv(ft, parse_dates=["date"])
        if "market_home_prob_novig" in hp and hp["market_home_prob_novig"].notna().any():
            out["final_test_backtest"] = market.benchmark(hp.rename(columns={"market_home_prob_novig": "p_market"})
                                                          .assign(home_win=hp["home_win"]), ev["n_boot"], ev["seed"])
    outdir = Path(args.out or "out/market")
    _write_json(outdir / "report.json", out)
    print(f"Model vs market (no-vig, proportional), book {book}. Each model is compared with the price recorded "
          "at its own prediction time.")
    for h, r in live["by_horizon"].items():
        if r.get("status") != "ok":
            print(f"  {h:>7}: {r.get('status')} (n={r.get('n', 0)})")
            continue
        print(f"  {h:>7}: n={r['n']}{' PRELIMINARY' if r['preliminary'] else ''}  market log loss "
              f"{r['market']['log_loss']:.4f}, Brier {r['market']['brier']:.4f}, acc {r['market']['accuracy']:.3f}")
        for c, s in r["models"].items():
            print(f"           {c:<15} log loss {s['log_loss']:.4f}  minus market {fmt_ci_short(s['log_loss_minus_market'])}")
    if "final_test_backtest" in out:
        print(f"final-test backtest games with archived odds: n={out['final_test_backtest'].get('n')}")
    print(f"wrote {outdir}/report.json")
    return 0


def _validated_metrics(cfg, champion: str) -> dict:
    """Out-of-sample metrics for the champion from the stored reports (never recomputed here)."""
    out = {}
    for key, sub in (("development walk-forward", "walkforward_dir"), ("final holdout", "final_test_dir")):
        rp = Path(cfg["paths"][sub]) / "report.json"
        if rp.exists():
            rep = json.loads(rp.read_text())
            m = rep["models"].get(champion)
            if m:
                out[key] = {"log_loss": m["log_loss"], "brier": m["brier"], "n": m["n"],
                            "range": rep["date_range"]}
    return out


def cmd_today(args, cfg):
    """Game-day workflow: refresh odds only if stale, predict the target date, log, report."""
    now = pd.Timestamp.now(tz="UTC")
    target = pd.Timestamp(args.date) if args.date else pd.Timestamp(now.tz_convert("US/Eastern").date())
    games, _, _ = load_games_from(cfg, args.games, None)
    model, meta = registry.load_model(None, cfg["paths"]["artifacts"])
    champion = (meta.get("reference_metrics") or {}).get("champion", f"p_{meta['model']}")
    book = args.book or cfg.get("paper", {}).get("book", cfg["paths"]["book"])
    day = games[games["date"] == target].copy()
    if day.empty:
        print(f"no NBA games scheduled on {target.date()} in {cfg['paths']['games']} (run `make data-update`)")
        return 0
    tip = pd.to_datetime(day["tipoff_utc"], utc=True, errors="coerce")
    excluded = []
    for r, t in zip(day.itertuples(), tip):
        g = f"{r.away_team} @ {r.home_team}"
        if r.played:
            excluded.append((g, "final result already recorded"))
        elif pd.isna(t):
            excluded.append((g, "no scheduled tip-off time"))
        elif t <= now:
            excluded.append((g, f"already started (tip-off {t:%H:%M} UTC)"))
    pending = day[~day["played"] & tip.notna() & (tip > now)]

    # 1) odds: reuse the archive when fresh, otherwise spend one credit
    with store.connect(cfg["paths"]["db"]) as con:
        snap = odds_store.load_snapshots(con, book=book)
        mine = snap[snap["game_key"].isin(pending["game_key"])]
        last = mine["observed_at"].max() if len(mine) else None
        fresh = last is not None and (now - last) <= pd.Timedelta(minutes=args.max_odds_age)
        covered = set(mine["game_key"]) >= set(pending["game_key"])
        odds_note = f"reused archive (latest {last:%Y-%m-%d %H:%M:%S} UTC)" if last is not None else "none archived"
        if not pending.empty and not args.no_refresh and not (fresh and covered):
            res = odds_collect.collect(con, _provider(args, cfg), games, _schedule(cfg), force=True)
            if res["status"] == "ok":
                odds_note = (f"fetched now from The Odds API (run {res['run_id']}, {res['n_inserted']} prices, "
                             f"{(res.get('usage') or {}).get('x-requests-remaining')} credits left)")
            else:
                odds_note += f"; refresh FAILED ({res.get('error')}), using archive"
    if pending.empty:
        print(f"{target.date()}: no games left to predict before tip-off")
        for g, why in excluded:
            print(f"  excluded {g}: {why}")
        return 0

    # 2) predictions with prices as of the prediction timestamp
    horizon = int((target - games.loc[~games["played"], "date"].min()).days) + 1
    preds = predict_upcoming(games, model, meta, max(horizon, 1), now)
    preds = preds[preds["date"] == target]
    preds = _add_shadow_blend(preds, games, cfg)
    with store.connect(cfg["paths"]["db"]) as con:
        preds, quotes = market.attach_market_asof(preds, odds_store.load_snapshots(con), book,
                                                  cfg.get("odds", {}).get("max_age_minutes", 120))
        n = n_q = 0
        if not args.no_log:
            n = store.log_predictions(con, preds)
            n_q = store.log_prediction_odds(con, preds, quotes)
        decided = pd.DataFrame()
        if cfg.get("paper", {}).get("enabled") and not args.no_log:
            decided = paper.decide(con, paper.PaperConfig.from_dict(cfg.get("paper")),
                                   model_version=preds["model_version"].iloc[0])
    out = Path(args.out or f"out/today/predictions_{target.date()}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with_winner(preds).to_csv(out, index=False, float_format="%.6f")

    # 3) report
    t = meta["training"]
    print(f"NBA moneyline predictions for {target.date()} (US/Eastern game date)")
    print(f"model: {meta['model']}  version {meta['version']}  trained {meta['trained_at'][:16]} UTC")
    print(f"training games: {t['start']} to {t['end']} ({t['n_rows']} games); features use results through "
          f"{preds['latest_result_in_data'].iloc[0]}")
    for k, m in _validated_metrics(cfg, champion).items():
        print(f"validated ({k}, {m['range'][0]} to {m['range'][1]}, n={m['n']}): log loss {m['log_loss']:.4f}, "
              f"Brier {m['brier']:.4f}")
    print(f"prediction timestamp: {preds['prediction_timestamp'].iloc[0]}")
    print(f"odds: book {book}, {odds_note}")
    rows = []
    preds = preds.assign(_tip=pd.to_datetime(preds["tipoff_utc"], utc=True)).sort_values(["_tip", "game_key"])
    for r in preds.to_dict("records"):
        home = r["p_home"] >= 0.5
        winner = r["home_team"] if home else r["away_team"]
        p_w = r["p_home"] if home else r["p_away"]
        fair = r["fair_home_american"] if home else r["fair_away_american"]
        flags = []
        if pd.notna(r["home_odds"]):
            h_am = decimal_to_american_int(r["home_odds"])
            a_am = decimal_to_american_int(r["away_odds"])
            age = (pd.Timestamp(r["prediction_timestamp"]) - pd.Timestamp(r["odds_timestamp"])).total_seconds() / 60
            if age > args.max_odds_age:
                flags.append(f"odds {age:.0f} min old")
            nv_w = r["market_home_prob_novig"] if home else 1 - r["market_home_prob_novig"]
            fav = r["home_team"] if r["market_home_prob_novig"] > 0.5 else r["away_team"]
            book_s = f"{r['home_team']} {h_am:+d} / {r['away_team']} {a_am:+d}"
            edge = f"{p_w - nv_w:+.3f}"
            roi = f"{(r['ev_home'] if home else r['ev_away']):+.1%}"
            nv = f"{r['market_home_prob_novig']:.3f}/{1 - r['market_home_prob_novig']:.3f}"
            differs = "YES" if fav != winner else "no"
        else:
            book_s, edge, roi, nv, differs = "missing", "n/a", "n/a", "n/a", "n/a"
            flags.append("no sportsbook price")
        if r["low_history"]:
            flags.append("early season (<5 games)")
        rows.append({"Tip-off (ET)": r["_tip"].tz_convert("US/Eastern").strftime("%I:%M %p"),
                     "Game": f"{r['away_team']} @ {r['home_team']}", "Predicted Winner": winner,
                     "Win Probability": f"{p_w:.1%}", "Model Fair Odds": f"{fair:+d}",
                     "Sportsbook Odds": book_s, "No-vig (home/away)": nv, "Estimated Edge": edge,
                     "Expected ROI": roi, "Differs from book favorite": differs, "Flags": "; ".join(flags)})
    table = pd.DataFrame(rows)
    print()
    print(_markdown(table) if args.markdown else table.to_string(index=False))
    print("\nEstimated Edge = model minus no-vig market probability for the predicted winner; Expected ROI = "
          "model probability x book decimal odds - 1 for that side. Both assume the model is right. The most "
          "likely winner is not necessarily the best value, and neither column is a betting recommendation.")
    for g, why in excluded:
        print(f"excluded {g}: {why}")
    if args.no_log:
        print(f"not logged (--no-log): {cfg['paths']['db']} is unchanged; wrote {out}")
    else:
        print(f"saved {n} predictions ({n_q} book prices recorded with them) to {cfg['paths']['db']}; wrote {out}")
    if not decided.empty:
        print("paper decisions made now (SIMULATED): " + ", ".join(
            f"{k} {v}" for k, v in decided["decision"].value_counts().items()))
    print("Paper decisions for later tip-offs are made by the scheduled collector 30-90 min before tip-off. "
          "No real bets are placed.")
    return 0


def with_winner(preds: pd.DataFrame) -> pd.DataFrame:
    """Add predicted_winner (team with p >= 0.5, home on an exact tie) and its probability as the first
    columns after the game identifiers."""
    home = preds["p_home"] >= 0.5
    out = preds.copy()
    at = list(out.columns).index("away_team") + 1 if "away_team" in out else 0
    out.insert(at, "predicted_winner", np.where(home, out["home_team"], out["away_team"]))
    out.insert(at + 1, "predicted_winner_prob", np.where(home, out["p_home"], out["p_away"]))
    return out


def _markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def decimal_to_american_int(d) -> int:
    from .odds import decimal_to_american
    return int(round(float(decimal_to_american(float(d)))))


def cmd_weekly_report(args, cfg):
    from . import weekly
    end = (pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC").tz_convert("US/Eastern").normalize()
           .tz_localize(None) + pd.Timedelta(days=1)).tz_localize("UTC")
    ref = {}
    version = registry.active_version(cfg["paths"]["artifacts"])
    if version:
        ref = registry.load_model(version, cfg["paths"]["artifacts"])[1].get("reference_metrics", {})
    with store.connect(cfg["paths"]["db"]) as con:
        text = weekly.build(con, cfg, end, args.days, ref, cfg["evaluation"]["n_boot"])
    out = Path(args.out or f"out/reports/weekly_{(end - pd.Timedelta(days=1)).date()}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(text)
    print(f"\nwrote {out}")
    return 0


def cmd_paper_trade(args, cfg):
    pcfg = paper.PaperConfig.from_dict(cfg.get("paper"))
    with store.connect(cfg["paths"]["db"]) as con:
        now = pd.Timestamp(args.now, tz="UTC") if args.now else None
        decided = paper.decide(con, pcfg, now=now) if not args.report_only else pd.DataFrame()
        s = paper.summarize(con, pcfg, cfg["evaluation"]["n_boot"], cfg["evaluation"]["seed"])
        c = clv.summary(con, pcfg.strategy)
    if not decided.empty:
        print("new decisions: " + ", ".join(f"{k} {v}" for k, v in decided["decision"].value_counts().items()))
    print(f"PAPER TRADING (simulated, strategy {s['strategy']}, book {s['book']}): decisions {s['decisions']}")
    print(f"bets {s['n_bets']} (settled {s['n_settled']}, open {s['n_open']}); mean est. EV {s['mean_est_ev']}; "
          f"mean edge vs no-vig {s['mean_edge_vs_novig']}")
    if s["status"] == "ok":
        print(f"win rate {s['win_rate']:.3f} vs break-even {s['mean_break_even']:.3f}; flat ROI "
              f"{fmt_ci_short(s['flat']['roi'])}; flat max drawdown {s['flat']['max_drawdown_units']:.2f}u; "
              f"Kelly bankroll {s['kelly']['bankroll']:.2f} (max drawdown {s['kelly']['max_drawdown_fraction']:.1%})"
              + ("  PRELIMINARY" if s["preliminary"] else ""))
    print(f"{c.get('label')}: {c.get('bets', {})} status {c.get('status_counts', {})}")
    print(s["caveat"])
    return 0


def cmd_dashboard(args, cfg):
    from .dashboard import build_dashboard
    out = build_dashboard(cfg, args.out)
    print(f"wrote {out}")
    return 0


# ------------------------------------------------------------------ parser
def build_parser():
    p = argparse.ArgumentParser(prog="python -m nbamodel", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--games", help="games CSV (default from config)")
        sp.add_argument("--odds", help="odds snapshot CSV from scripts/collect_odds.py")
        sp.add_argument("--book", help='bookmaker key or "best"')
        return sp

    common(sub.add_parser("build-dataset", help="write the training dataset")).add_argument("--out")
    wf = common(sub.add_parser("walk-forward", help="development walk-forward backtest"))
    wf.add_argument("--out")
    wf.add_argument("--retrain", choices=["weekly", "monthly", "season"])
    wf.add_argument("--window", choices=["expanding", "rolling"])
    wf.add_argument("--calibration", choices=["none", "platt", "isotonic"])
    common(sub.add_parser("final-test", help="evaluate once on the untouched final period")).add_argument("--out")
    ev = sub.add_parser("evaluate", help="score a predictions CSV")
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--out", required=True)
    tr = common(sub.add_parser("train", help="fit and save the production model"))
    tr.add_argument("--model", choices=["elo", "logistic", "logistic_raw", "gbm", "gbm_raw", "ensemble"],
                    help="override the walk-forward champion (marked experimental)")
    pr = common(sub.add_parser("predict", help="predict upcoming games"))
    pr.add_argument("--days", type=int)
    pr.add_argument("--version", help="model version (default: active)")
    pr.add_argument("--out")
    pr.add_argument("--as-of", help="prediction time (UTC ISO) instead of now, e.g. to replay a past decision")
    up = common(sub.add_parser("update-results", help="reconcile results and monitor"))
    up.add_argument("--retrain-if-due", action="store_true")
    sub.add_parser("dashboard", help="build the HTML dashboard").add_argument("--out")

    co = sub.add_parser("collect-odds", help="store an odds snapshot if a scheduled target is due")
    co.add_argument("--games")
    co.add_argument("--force", action="store_true", help="fetch even if no target is due (uses a credit)")
    co.add_argument("--no-budget-guard", action="store_true", help="keep all offsets even if over budget")
    co.add_argument("--then-predict", action="store_true", help="run `predict` after a successful fetch")
    co.add_argument("--from-file", help="replay a saved provider JSON response instead of calling the API")
    co.add_argument("--observed-at", help="observation time (UTC ISO) of --from-file data")
    co.add_argument("--file-format", default="decimal", choices=["decimal", "american"])
    op = sub.add_parser("odds-plan", help="estimate API credit use for the collection schedule")
    op.add_argument("--games")
    op.add_argument("--start")
    op.add_argument("--end")
    os_ = sub.add_parser("odds-status", help="collection runs and per-game snapshot coverage")
    ru = sub.add_parser("results-update", help="scores, tip-off times and player box scores from ESPN")
    ru.add_argument("--games")
    ru.add_argument("--db")
    ru.add_argument("--start", help="first US/Eastern date (default: --days-back before today)")
    ru.add_argument("--days-back", type=int, default=3)
    ru.add_argument("--days-ahead", type=int, default=7, help="also refresh tip-off times this far ahead")
    pb = sub.add_parser("players-backfill", help="historical player box scores from ESPN into a separate database")
    pb.add_argument("--games")
    pb.add_argument("--db", default="data/player_history.db")
    pb.add_argument("--start", default="2014-10-28")
    pb.add_argument("--end", help="last game date (default: yesterday, US/Eastern)")
    pb.add_argument("--max-seconds", type=float, default=3600)
    pb.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    ir = sub.add_parser("injuries-refresh", help="ingest new official NBA injury reports (today's, from the season page)")
    ir.add_argument("--games")
    ir.add_argument("--db", help="database (default from config)")
    ib = sub.add_parser("injuries-backfill", help="fetch historical official injury reports into a separate database")
    ib.add_argument("--games")
    ib.add_argument("--db", default="data/injury_history.db")
    ib.add_argument("--start", default="2019-10-22", help="first game date (the archive starts in 2019-20)")
    ib.add_argument("--end", help="last game date (default: yesterday, US/Eastern)")
    ib.add_argument("--max-seconds", type=float, default=3600)
    ib.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    ist = sub.add_parser("injuries-status", help="latest official injury statuses for a game date")
    ist.add_argument("--date", help="US/Eastern game date (default: today)")
    ist.add_argument("--db")
    on = sub.add_parser("odds-next", help="exit 0 if a snapshot target opens within --within-hours, else 1")
    on.add_argument("--games")
    on.add_argument("--within-hours", type=float, default=6.0)
    os_.add_argument("--games")
    os_.add_argument("--days", type=int, default=3)
    io = sub.add_parser("import-odds", help="import an odds snapshot CSV into the archive")
    io.add_argument("--csv", required=True)
    io.add_argument("--games")
    io.add_argument("--provider", default="the-odds-api")
    bs = sub.add_parser("blend-study", help="early-season Elo blending study on development predictions")
    bs.add_argument("--base", help="ML probability column (default from config)")
    bs.add_argument("--confirm-holdout", action="store_true", help="score the chosen blend once on the final holdout")
    bs.add_argument("--out")
    mb = sub.add_parser("market-benchmark", help="models vs no-vig market at matching prediction times")
    mb.add_argument("--book")
    mb.add_argument("--out")
    wr = sub.add_parser("weekly-report", help="Markdown weekly performance report from the live log")
    wr.add_argument("--days", type=int, default=7)
    wr.add_argument("--end", help="exclusive end date (default: tomorrow, US/Eastern)")
    wr.add_argument("--out")
    td = sub.add_parser("today", help="game-day workflow: refresh stale odds, predict a date, log, report")
    td.add_argument("--date", help="US/Eastern game date (default: today)")
    td.add_argument("--games")
    td.add_argument("--book")
    td.add_argument("--max-odds-age", type=float, default=30.0,
                    help="reuse archived odds younger than this many minutes (default 30)")
    td.add_argument("--no-refresh", action="store_true", help="never call the API; use the archive only")
    td.add_argument("--no-log", action="store_true",
                    help="report only: no predictions or paper decisions written (read-only mirror of the archive)")
    td.add_argument("--markdown", action="store_true")
    td.add_argument("--out")
    pt = sub.add_parser("paper-trade", help="make due paper decisions and report the simulated ledger")
    pt.add_argument("--report-only", action="store_true")
    pt.add_argument("--now", help="decision clock (UTC ISO) for replaying logged predictions; default now")
    return p


COMMANDS = {"build-dataset": cmd_build_dataset, "walk-forward": cmd_walk_forward, "final-test": cmd_final_test,
            "evaluate": cmd_evaluate, "train": cmd_train, "predict": cmd_predict,
            "update-results": cmd_update_results, "dashboard": cmd_dashboard,
            "collect-odds": cmd_collect_odds, "odds-plan": cmd_odds_plan, "odds-next": cmd_odds_next,
            "results-update": cmd_results_update, "players-backfill": cmd_players_backfill, "injuries-refresh": cmd_injuries_refresh, "injuries-backfill": cmd_injuries_backfill,
            "injuries-status": cmd_injuries_status,
            "odds-status": cmd_odds_status,
            "import-odds": cmd_import_odds, "blend-study": cmd_blend_study, "market-benchmark": cmd_market_benchmark,
            "paper-trade": cmd_paper_trade, "today": cmd_today,
            "weekly-report": cmd_weekly_report}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = load_config(args.config)
        return COMMANDS[args.command](args, cfg)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
