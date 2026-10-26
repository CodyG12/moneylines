"""Phase 2 command line: `python -m nbamodel <command>`.

  build-dataset    write the leakage-checked historical training dataset
  walk-forward     development walk-forward backtest: metrics, plots, champion choice
  final-test       one-time evaluation on the untouched final period (logged)
  evaluate         re-score any predictions CSV
  train            fit the champion on all completed games, save a versioned artifact
  predict          predict upcoming games with the active model and log them
  update-results   reconcile results, update metrics/Elo, detect degradation, retrain if due
  dashboard        build the static HTML dashboard

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

from . import plots, registry, store
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
    if odds_path:
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


def cmd_predict(args, cfg):
    games, _, _ = load_games_from(cfg, args.games, args.odds, args.book)
    model, meta = registry.load_model(args.version, cfg["paths"]["artifacts"])
    now = pd.Timestamp(args.as_of, tz="UTC") if args.as_of else None
    preds = predict_upcoming(games, model, meta, args.days or cfg["predict"]["horizon_days"], now)
    if preds.empty:
        print("no upcoming scheduled games in the data (run scripts/update_schedule.py)")
        return 0
    out = Path(args.out or cfg["paths"]["upcoming"])
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out, index=False, float_format="%.6f")
    with store.connect(cfg["paths"]["db"]) as con:
        n = store.log_predictions(con, preds)
    print(f"model {meta['version']}; data through {preds['latest_result_in_data'].iloc[0]}")
    print(f"{'date':<11} {'game':<11} {'p(home)':>7} {'fair':>11} {'spread':>6}  market / notes")
    for _, r in preds.iterrows():
        line = (f"{r['date'].date()!s:<11} {r['away_team']:>3} @ {r['home_team']:<3}  {r['p_home']:>7.3f} "
                f"{r['fair_home_american']:+5d}/{r['fair_away_american']:+5d} {r['model_spread']:>6.3f}")
        if pd.notna(r["home_odds"]):
            line += f"  book {r['home_odds']:.2f}/{r['away_odds']:.2f} no-vig {r['market_home_prob_novig']:.3f}"
        if r["low_history"]:
            line += "  [low history]"
        print(line)
    print(f"\nlogged {n} new predictions to {cfg['paths']['db']}; wrote {out}")
    print("Fair odds are the model's zero-margin prices, not bookmaker prices. A positive estimated edge is "
          "not a guarantee of profit.")
    return 0


def cmd_update_results(args, cfg):
    games, _, games_path = load_games_from(cfg, args.games, args.odds, args.book)
    fb = cfg["feedback"]
    with store.connect(cfg["paths"]["db"]) as con:
        rec = store.reconcile(con, games)
        metrics = store.snapshot_metrics(con)
        _, ratings = compute_elo(games, ds_config(cfg).elo)
        store.snapshot_elo(con, ratings)
        print(f"results recorded for {rec['results_recorded']} predicted games; scored {rec['predictions_scored']} "
              f"predictions ({rec['late_predictions']} made after tip-off, excluded from metrics)")
        if not metrics.empty:
            print(metrics[["model_version", "window", "n", "log_loss", "brier", "accuracy", "ece"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
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
    return p


COMMANDS = {"build-dataset": cmd_build_dataset, "walk-forward": cmd_walk_forward, "final-test": cmd_final_test,
            "evaluate": cmd_evaluate, "train": cmd_train, "predict": cmd_predict,
            "update-results": cmd_update_results, "dashboard": cmd_dashboard}


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
