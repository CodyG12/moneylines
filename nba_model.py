#!/usr/bin/env python3
"""Train, backtest and score the Phase 1 NBA moneyline model.

Example:
    python nba_model.py --games data/sample/synthetic_games.csv --output out/predictions.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from nbamodel.backtest import BettingConfig
import pandas as pd

from nbamodel.data import DataError, prepare_games
from nbamodel.elo import EloConfig
from nbamodel.odds_join import attach_odds
from nbamodel.pipeline import PipelineConfig, run_pipeline

log = logging.getLogger("nba_model")

PREDICTION_COLUMNS = [
    "game_id", "date", "season", "home_team", "away_team", "home_score", "away_score", "home_win",
    "home_elo_pre", "away_elo_pre", "elo_home_prob", "model_home_prob",
    "fair_home_american", "fair_away_american", "fair_home_decimal", "fair_away_decimal",
    "home_odds", "away_odds", "market_home_prob_novig", "market_overround", "ev_home", "ev_away",
    "home_form_n", "away_form_n", "home_days_rest", "away_days_rest",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", required=True, help="CSV of historical (and optionally scheduled) games")
    p.add_argument("--output", default="predictions.csv", help="holdout predictions CSV")
    p.add_argument("--odds", help="odds snapshot CSV from scripts/collect_odds.py (replaces odds in --games)")
    p.add_argument("--book", default="draftkings", help='bookmaker key, or "best" (optimistic line shopping)')
    p.add_argument("--odds-lead-minutes", type=float, default=60.0,
                   help="use the latest quote observed at least this long before tip-off")
    p.add_argument("--upcoming-days", type=int, default=7,
                   help="predict scheduled games within this many days of the next game date")
    p.add_argument("--window", type=int, default=10, help="recent-form window in games")
    p.add_argument("--elo-k", type=float, default=20.0)
    p.add_argument("--elo-home-advantage", type=float, default=100.0)
    p.add_argument("--elo-carryover", type=float, default=0.75)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--validation-frac", type=float, default=0.2)
    p.add_argument("--ev-threshold", type=float, default=0.02)
    p.add_argument("--kelly-multiplier", type=float, default=0.25)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _fmt_ci(d):
    return f"{d['estimate']:.4f} [{d['ci_low']:.4f}, {d['ci_high']:.4f}]"


def print_summary(metrics: dict) -> None:
    split = metrics["split"]
    print("\nChronological split (games / dates):")
    for name, s in split.items():
        print(f"  {name:<11} {s['games']:>6} / {s['dates']:<5} {s['start']} -> {s['end']}")
    print(f"\nChosen C = {metrics['model_selection']['chosen_C']} (by validation log loss)")
    print("\nHoldout (n = %d games), 95%% CIs resample dates:" % metrics["holdout"]["logistic"]["n"])
    print(f"  {'model':<15} {'log loss [95% CI]':<28} {'brier':>7} {'acc':>6}")
    for name, s in metrics["holdout"].items():
        print(f"  {name:<15} {_fmt_ci(s['log_loss_ci']):<28} {s['brier']:>7.4f} {s['accuracy']:>6.3f}")
    for name, c in metrics["holdout_comparisons"].items():
        print(f"  {name}: {_fmt_ci(c)}  (negative favours logistic)")
    mc = metrics.get("market_comparison")
    if mc:
        print(f"\nMarket comparison on {mc['n_games_with_odds']} holdout games with odds "
              f"(timing verified: {mc['odds_timing_verified']}):")
        print(f"  market no-vig log loss {mc['market_novig']['log_loss']:.4f}, "
              f"logistic {mc['logistic_on_same_games']['log_loss']:.4f}, "
              f"diff {_fmt_ci(mc['logistic_minus_market_log_loss'])}")
        pt = metrics["paper_trading"]
        print(f"\nPaper trading (EV > {pt['ev_threshold']:.0%}): {pt['n_bets']} bets")
        if pt["n_bets"]:
            fl = pt["flat"]
            print(f"  flat 1u: realised {fl['realized_profit_units']:+.2f}u on {fl['total_staked_units']:.0f}u, "
                  f"ROI {_fmt_ci(fl['roi'])}, max drawdown {fl['max_drawdown_units']:.2f}u")
            k = pt["fractional_kelly"]
            print(f"  {k['settings']['kelly_multiplier']:.2f}x Kelly (capped): bankroll "
                  f"{k['initial_bankroll']:.1f} -> {k['final_bankroll']:.1f}, "
                  f"max drawdown {k['max_drawdown_fraction']:.1%}")
        print(f"  {pt['caveat']}")
    for w in metrics["data"]["warnings"]:
        print(f"\nWARNING: {w}")


def print_upcoming(upcoming) -> None:
    if upcoming.empty:
        return
    print(f"\nUpcoming games ({len(upcoming)}): model home win probability and fair odds")
    for _, r in upcoming.iterrows():
        line = (f"  {r['date'].date()} {r['away_team']:>3} @ {r['home_team']:<3} "
                f"p(home)={r['model_home_prob']:.3f}  fair {r['fair_home_american']:+d}/{r['fair_away_american']:+d}")
        if r["home_odds"] == r["home_odds"]:
            line += (f"  market {r['home_odds']:.2f}/{r['away_odds']:.2f} "
                     f"no-vig p(home)={r['market_home_prob_novig']:.3f}")
        if r["low_history"]:
            line += "  [low history]"
        print(line)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = PipelineConfig(
        elo=EloConfig(k=args.elo_k, home_advantage=args.elo_home_advantage, season_carryover=args.elo_carryover),
        betting=BettingConfig(ev_threshold=args.ev_threshold, kelly_multiplier=args.kelly_multiplier),
        window=args.window, train_frac=args.train_frac, validation_frac=args.validation_frac,
        n_boot=args.n_boot, seed=args.seed, upcoming_days=args.upcoming_days,
    )
    try:
        try:
            raw = pd.read_csv(args.games, comment="#")
        except FileNotFoundError as exc:
            raise DataError(f"games file not found: {args.games}") from exc
        if args.odds:
            try:
                snapshots = pd.read_csv(args.odds)
            except FileNotFoundError as exc:
                raise DataError(f"odds file not found: {args.odds}") from exc
            raw = attach_odds(raw, snapshots, book=args.book, lead_minutes=args.odds_lead_minutes)
        games, report = prepare_games(raw)
        result = run_pipeline(games, report, config)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    result.holdout[PREDICTION_COLUMNS].to_csv(out, index=False, float_format="%.6f")
    result.calibration.to_csv(f"{stem}_calibration.csv", index=False, float_format="%.6f")
    with open(f"{stem}_metrics.json", "w") as fh:
        json.dump(result.metrics, fh, indent=2, default=str)
    written = [str(out), f"{stem}_calibration.csv", f"{stem}_metrics.json"]
    if result.bets is not None:
        result.bets.to_csv(f"{stem}_paper_bets.csv", index=False, float_format="%.6f")
        written.append(f"{stem}_paper_bets.csv")
    if not result.upcoming.empty:
        cols = [c for c in PREDICTION_COLUMNS if c not in ("home_score", "away_score", "home_win")]
        cols += [c for c in ("tipoff_utc", "odds_timestamp", "odds_book") if c in result.upcoming.columns]
        cols += ["low_history"]
        result.upcoming[cols].to_csv(f"{stem}_upcoming.csv", index=False, float_format="%.6f")
        written.append(f"{stem}_upcoming.csv")

    print_summary(result.metrics)
    print_upcoming(result.upcoming)
    print("\nWrote: " + ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
