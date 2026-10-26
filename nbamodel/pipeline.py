"""End-to-end Phase 1 pipeline: features -> chronological split -> model
selection on validation -> single holdout evaluation -> optional paper trading
-> predictions for scheduled games.

Data roles (strictly chronological, split by date so no date straddles two sets):
  train       fits candidate models
  validation  chooses the regularisation strength (never used for final scoring)
  holdout     touched once, by the model refit on train+validation
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import BettingConfig, select_bets, simulate
from .data import DataError, DataReport
from .elo import EloConfig
from .evaluation import calibration_table, cluster_bootstrap_ratio, compare_log_loss, per_game_log_loss, score
from .features import FEATURES, build_features
from .odds import fair_american_odds, fair_decimal_odds, overround, remove_vig_two_way

log = logging.getLogger(__name__)

MIN_GAMES = 100
MIN_DATES = 30
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


@dataclass
class PipelineConfig:
    elo: EloConfig = field(default_factory=EloConfig)
    betting: BettingConfig = field(default_factory=BettingConfig)
    window: int = 10
    train_frac: float = 0.6
    validation_frac: float = 0.2
    min_history: int = 5          # training rows need both teams to have >= this many prior games
    n_boot: int = 1000
    seed: int = 0
    upcoming_days: int = 7


@dataclass
class PipelineResult:
    holdout: pd.DataFrame
    metrics: dict
    calibration: pd.DataFrame
    bets: pd.DataFrame | None
    upcoming: pd.DataFrame


def chronological_split(dates: pd.Series, train_frac: float, validation_frac: float) -> np.ndarray:
    if not (0 < train_frac < 1 and 0 < validation_frac < 1 and train_frac + validation_frac < 1):
        raise ValueError("need 0 < train_frac, validation_frac and train_frac + validation_frac < 1")
    unique = np.sort(dates.unique())
    n_train = int(len(unique) * train_frac)
    n_val = int(len(unique) * validation_frac)
    if n_train < 1 or n_val < 1 or n_train + n_val >= len(unique):
        raise DataError("not enough distinct dates for a train/validation/holdout split")
    train_end, val_end = unique[n_train - 1], unique[n_train + n_val - 1]
    return np.where(dates <= train_end, "train", np.where(dates <= val_end, "validation", "holdout"))


def _fit(X: pd.DataFrame, y: pd.Series, C: float):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    return model.fit(X, y)


def _add_odds_columns(df: pd.DataFrame, p_col: str) -> pd.DataFrame:
    df = df.copy()
    p = df[p_col].to_numpy()
    df["fair_home_decimal"] = fair_decimal_odds(p)
    df["fair_away_decimal"] = fair_decimal_odds(1 - p)
    df["fair_home_american"] = np.round(fair_american_odds(p)).astype(int)
    df["fair_away_american"] = np.round(fair_american_odds(1 - p)).astype(int)
    priced = df["home_odds"].notna() & df["away_odds"].notna()
    for col in ("market_home_prob_novig", "market_overround", "ev_home", "ev_away"):
        df[col] = np.nan
    if priced.any():
        h, a = df.loc[priced, "home_odds"], df.loc[priced, "away_odds"]
        df.loc[priced, "market_home_prob_novig"] = remove_vig_two_way(h, a)[0]
        df.loc[priced, "market_overround"] = overround(h, a)
        df.loc[priced, "ev_home"] = p[priced.to_numpy()] * h - 1
        df.loc[priced, "ev_away"] = (1 - p[priced.to_numpy()]) * a - 1
    return df


def run_pipeline(games: pd.DataFrame, report: DataReport, config: PipelineConfig = PipelineConfig()) -> PipelineResult:
    feats = build_features(games, config.elo, config.window)
    played = feats[feats["played"]].copy()
    if len(played) < MIN_GAMES or played["date"].nunique() < MIN_DATES:
        raise DataError(f"need at least {MIN_GAMES} played games across {MIN_DATES} dates; got "
                        f"{len(played)} games across {played['date'].nunique()} dates")

    played["split"] = chronological_split(played["date"], config.train_frac, config.validation_frac)
    enough_history = (played["home_history"] >= config.min_history) & (played["away_history"] >= config.min_history)
    trainable = played["split"].isin(["train", "validation"]) & enough_history
    train = played[(played["split"] == "train") & enough_history]
    val = played[(played["split"] == "validation") & enough_history]
    holdout = played[played["split"] == "holdout"].copy()
    if train["home_win"].nunique() < 2 or val.empty or holdout.empty:
        raise DataError("split produced an empty or single-class partition")

    # Model selection uses validation only.
    val_scores = {}
    for C in C_GRID:
        p_val = _fit(train[FEATURES], train["home_win"], C).predict_proba(val[FEATURES])[:, 1]
        val_scores[C] = score(val["home_win"], p_val)["log_loss"]
    best_C = min(val_scores, key=val_scores.get)
    log.info("validation log loss by C: %s -> chose C=%s", val_scores, best_C)

    fit_rows = played[trainable]
    final = _fit(fit_rows[FEATURES], fit_rows["home_win"], best_C)
    holdout["model_home_prob"] = final.predict_proba(holdout[FEATURES])[:, 1]
    holdout["base_rate_home_prob"] = float(fit_rows["home_win"].mean())

    y = holdout["home_win"].to_numpy()
    clusters = holdout["date"]
    probs = {"logistic": holdout["model_home_prob"], "elo": holdout["elo_home_prob"],
             "home_base_rate": holdout["base_rate_home_prob"]}
    metrics = {
        "data": asdict(report),
        "split": {name: {"games": int((played["split"] == name).sum()),
                         "dates": int(played.loc[played["split"] == name, "date"].nunique()),
                         "start": str(played.loc[played["split"] == name, "date"].min().date()),
                         "end": str(played.loc[played["split"] == name, "date"].max().date())}
                  for name in ("train", "validation", "holdout")},
        "training_rows_after_min_history": int(len(fit_rows)),
        "model_selection": {"validation_log_loss_by_C": {str(k): v for k, v in val_scores.items()},
                            "chosen_C": best_C, "features": FEATURES,
                            "coefficients_standardized": dict(zip(
                                FEATURES, final[-1].coef_[0].round(4).tolist()))},
        "holdout": {},
        "holdout_comparisons": {},
        "notes": [
            "Holdout metrics are from a single, untouched chronological period; CIs resample whole dates.",
            "Lower log loss / Brier is better. Accuracy is reported but is not the optimisation target.",
        ],
    }
    for name, p in probs.items():
        s = score(y, p)
        s["log_loss_ci"] = cluster_bootstrap_ratio(per_game_log_loss(y, p), np.ones(len(y)), clusters,
                                                   config.n_boot, config.seed)
        metrics["holdout"][name] = s
    for baseline in ("elo", "home_base_rate"):
        metrics["holdout_comparisons"][f"logistic_minus_{baseline}_log_loss"] = compare_log_loss(
            y, probs["logistic"], probs[baseline], clusters, config.n_boot, config.seed)

    holdout = _add_odds_columns(holdout, "model_home_prob")
    calib = calibration_table(y, holdout["model_home_prob"])

    bets_df = None
    priced = holdout["market_home_prob_novig"].notna()
    if priced.sum() > 0:
        hp = holdout[priced]
        metrics["market_comparison"] = {
            "n_games_with_odds": int(priced.sum()),
            "n_holdout_games_without_odds": int((~priced).sum()),
            "odds_timing_verified": report.odds_timing_verified,
            "market_novig": score(hp["home_win"], hp["market_home_prob_novig"]),
            "logistic_on_same_games": score(hp["home_win"], hp["model_home_prob"]),
            "logistic_minus_market_log_loss": compare_log_loss(
                hp["home_win"], hp["model_home_prob"], hp["market_home_prob_novig"], hp["date"],
                config.n_boot, config.seed),
            "mean_overround": float(hp["market_overround"].mean()),
        }
        bets = select_bets(holdout, "model_home_prob", config.betting.ev_threshold)
        bets_df, sim = simulate(bets, config.betting, config.n_boot, config.seed)
        sim["ev_threshold"] = config.betting.ev_threshold
        sim["caveat"] = ("Paper trading only. Estimated EV assumes the model's probabilities are correct; "
                         "realised profit in a single holdout sample is noisy, see the ROI CI.")
        metrics["paper_trading"] = sim

    upcoming = feats[~feats["played"] & (feats["date"] > played["date"].max())].copy()
    if not upcoming.empty:
        horizon = upcoming["date"].min() + pd.Timedelta(days=config.upcoming_days - 1)
        upcoming = upcoming[upcoming["date"] <= horizon].copy()
    if not upcoming.empty:
        # Refit on every played game (with enough history) for live use; this model
        # has no out-of-sample score of its own beyond the holdout result above.
        all_rows = played[enough_history]
        live = _fit(all_rows[FEATURES], all_rows["home_win"], best_C)
        upcoming["model_home_prob"] = live.predict_proba(upcoming[FEATURES])[:, 1]
        upcoming = _add_odds_columns(upcoming, "model_home_prob")
        upcoming["low_history"] = (upcoming["home_form_n"] < config.min_history) | (
            upcoming["away_form_n"] < config.min_history)

    return PipelineResult(holdout=holdout, metrics=metrics, calibration=calib, bets=bets_df, upcoming=upcoming)
