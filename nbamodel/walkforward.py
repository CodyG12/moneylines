"""Walk-forward (chronological) backtest.

For each evaluation period (week, month or season), in date order:
  1. training rows = completed, train-eligible games dated strictly before the period
     (expanding: all of history; rolling: the last `rolling_seasons` seasons)
  2. the latest `calibration_dates` game dates of that window are held back as the
     calibration window; models fit (and tune hyperparameters on an inner validation
     slice) on the earlier "fit" window only, so preprocessing is fitted on it too
  3. calibrators are fitted on the calibration window
  4. the period's games are predicted from inputs with every outcome column removed,
     and the predictions are frozen
  5. only then are outcomes joined and scored; the next period retrains with the
     newly completed games

Within a period, features (Elo, form, ratings) still update daily from completed
games, as they would live; only model weights wait for the next retrain.

`phase="development"` evaluates games before `final_test_start`; that is where
model and setting choices are made. `phase="final_test"` evaluates games from
`final_test_start` onward with the configuration fixed in advance. The final test
should be run once; see cli.final_test, which logs every run.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .data import DataError
from .dataset import DatasetConfig, build_training_dataset
from .features import FEATURES_V2
from .leakage import outcome_columns
from .models import (Calibrator, CalibratedModel, EloModel, GBMModel, LogisticModel, elo_grid_probabilities,
                     ensemble_mean)
from .odds import overround, remove_vig_two_way

log = logging.getLogger(__name__)

PERIOD_FREQ = {"weekly": "W-MON", "monthly": "M"}
MODEL_NAMES = ("elo", "logistic", "gbm")


@dataclass
class WalkForwardConfig:
    retrain: str = "monthly"          # weekly | monthly | season
    window: str = "expanding"         # expanding | rolling
    rolling_seasons: int = 4
    min_train_seasons: int = 2
    calibration_dates: int = 40       # most recent training dates held back for calibration
    calibration: str = "platt"        # none | platt | isotonic
    validation_frac: float = 0.2      # inner validation share of the fit window (hyperparameters)
    features: list = field(default_factory=lambda: list(FEATURES_V2))
    models: tuple = MODEL_NAMES
    gbm_backend: str = "sklearn"
    seed: int = 0
    final_test_start: str = "2025-10-01"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["models"] = list(self.models)
        return d


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    periods: pd.DataFrame
    importance: dict
    config: dict


def assign_periods(dates: pd.Series, seasons: pd.Series, retrain: str) -> pd.Series:
    if retrain == "season":
        return seasons.astype(str)
    if retrain not in PERIOD_FREQ:
        raise ValueError(f"retrain must be weekly, monthly or season, not {retrain!r}")
    return dates.dt.to_period(PERIOD_FREQ[retrain]).astype(str)


def _evaluation_rows(ds: pd.DataFrame, cfg: WalkForwardConfig, phase: str) -> pd.DataFrame:
    played = ds[ds["played"]]
    seasons = np.sort(played["season"].unique())
    if len(seasons) <= cfg.min_train_seasons:
        raise DataError(f"need more than {cfg.min_train_seasons} seasons for walk-forward validation")
    final_start = pd.Timestamp(cfg.final_test_start)
    if phase == "development":
        first_eval_season = seasons[cfg.min_train_seasons]
        rows = played[(played["season"] >= first_eval_season) & (played["date"] < final_start)]
    elif phase == "final_test":
        rows = played[played["date"] >= final_start]
    else:
        raise ValueError("phase must be 'development' or 'final_test'")
    if rows.empty:
        raise DataError(f"no games to evaluate in the {phase} phase")
    return rows


def _attach_market(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["market_home_prob_novig"] = np.nan
    df["market_overround"] = np.nan
    if {"home_odds", "away_odds"} <= set(df.columns):
        priced = df["home_odds"].notna() & df["away_odds"].notna()
        if priced.any():
            h, a = df.loc[priced, "home_odds"], df.loc[priced, "away_odds"]
            df.loc[priced, "market_home_prob_novig"] = remove_vig_two_way(h, a)[0]
            df.loc[priced, "market_overround"] = overround(h, a)
    return df


def fit_models(train: pd.DataFrame, cfg: WalkForwardConfig, elo_probs: dict) -> dict:
    """Fit every configured model on `train` (fit window + calibration window split inside)."""
    dates = np.sort(train["date"].unique())
    if len(dates) <= cfg.calibration_dates + 10:
        raise DataError("training window too short for the calibration window")
    cal_start = dates[-cfg.calibration_dates]
    fit_rows, cal_rows = train[train["date"] < cal_start], train[train["date"] >= cal_start]
    fitted = {}
    if "elo" in cfg.models:
        fitted["elo"] = EloModel(elo_probs).fit(train)
    if "logistic" in cfg.models:
        fitted["logistic"] = CalibratedModel(
            LogisticModel(cfg.features, validation_frac=cfg.validation_frac),
            Calibrator(cfg.calibration)).fit(fit_rows, cal_rows)
    if "gbm" in cfg.models:
        fitted["gbm"] = CalibratedModel(
            GBMModel(cfg.features, validation_frac=cfg.validation_frac, seed=cfg.seed, backend=cfg.gbm_backend),
            Calibrator(cfg.calibration)).fit(fit_rows, cal_rows)
    fitted["_windows"] = {"fit_rows": fit_rows, "cal_rows": cal_rows, "cal_start": pd.Timestamp(cal_start)}
    return fitted


def predict_frame(fitted: dict, X: pd.DataFrame, base_rate: float) -> pd.DataFrame:
    """Model probabilities for `X`, which must not contain outcome columns."""
    leaked = set(outcome_columns(X))
    if leaked:
        raise DataError(f"prediction inputs contain outcome columns: {sorted(leaked)}")
    out = pd.DataFrame(index=X.index)
    out["p_home_base_rate"] = base_rate
    if "elo" in fitted:
        out["p_elo"] = fitted["elo"].predict(X)
    for name in ("logistic", "gbm"):
        if name in fitted:
            out[f"p_{name}_raw"] = fitted[name].predict_raw(X)
            out[f"p_{name}"] = fitted[name].predict(X)
    if "p_logistic" in out and "p_gbm" in out:
        out["p_ensemble"] = ensemble_mean(out["p_logistic"], out["p_gbm"])
    return out


def run_walk_forward(games: pd.DataFrame, cfg: WalkForwardConfig = WalkForwardConfig(),
                     ds_config: DatasetConfig = DatasetConfig(), phase: str = "development",
                     dataset: pd.DataFrame | None = None) -> WalkForwardResult:
    ds = dataset if dataset is not None else build_training_dataset(games, ds_config)
    elo_probs = elo_grid_probabilities(games) if "elo" in cfg.models else {}
    eval_rows = _evaluation_rows(ds, cfg, phase)
    period_of = assign_periods(eval_rows["date"], eval_rows["season"], cfg.retrain)
    trainable = ds[ds["train_eligible"]]
    outcome_cols = outcome_columns(ds)

    frozen, periods = [], []
    for period in sorted(period_of.unique(), key=lambda p: eval_rows.loc[period_of == p, "date"].min()):
        test = eval_rows[period_of == period]
        start = test["date"].min()
        train = trainable[trainable["date"] < start]
        if cfg.window == "rolling":
            train = train[train["season"] > test["season"].min() - cfg.rolling_seasons - 1]
        assert train["date"].max() < start  # strict chronology
        fitted = fit_models(train, cfg, elo_probs)
        w = fitted.pop("_windows")
        X = test.drop(columns=outcome_cols)
        preds = predict_frame(fitted, X, float(w["fit_rows"]["home_win"].mean()))
        preds.insert(0, "game_key", test["game_key"].to_numpy())
        preds.insert(1, "period", period)
        frozen.append(preds)
        info = {"period": period, "test_start": start, "test_end": test["date"].max(), "n_test": len(test),
                "train_start": train["date"].min(), "train_end": train["date"].max(), "n_train": len(train),
                "n_fit": len(w["fit_rows"]), "cal_start": w["cal_start"], "n_cal": len(w["cal_rows"])}
        for name, m in fitted.items():
            info[f"{name}_params"] = m.params
        if "logistic" in fitted:
            info["logistic_coefficients"] = fitted["logistic"].base.importance()
        periods.append(info)
        log.info("period %s: train %d rows to %s, test %d games", period, len(train),
                 train["date"].max().date(), len(test))
        last = (fitted, w)

    importance = {}
    fitted, w = last
    if "logistic" in fitted:
        importance["logistic_standardized_coefficients"] = fitted["logistic"].base.importance()
    if "gbm" in fitted:
        importance["gbm_permutation_log_loss_increase"] = fitted["gbm"].base.importance(w["cal_rows"])

    # Outcomes are revealed only after every prediction has been frozen.
    predictions = pd.concat(frozen, ignore_index=True)
    keep = ["game_key", "game_id_source", "season", "date", "tipoff_utc", "prediction_timestamp", "home_team",
            "away_team", "neutral_site", "home_season_games", "away_season_games"]
    odds_cols = [c for c in ("home_odds", "away_odds", "odds_timestamp", "odds_book",
                             "close_home_odds", "close_away_odds") if c in ds.columns]
    revealed = ds.set_index("game_key").loc[predictions["game_key"], [c for c in keep[1:] + odds_cols]
                                             + ["home_score", "away_score", "home_win"]].reset_index()
    predictions = revealed.merge(predictions, on="game_key", validate="one_to_one")
    predictions = _attach_market(predictions)
    predictions["phase"] = phase
    periods_df = pd.DataFrame(periods)
    return WalkForwardResult(predictions=predictions, periods=periods_df, importance=importance,
                             config={**cfg.to_dict(), "phase": phase,
                                     "dataset": {"elo": asdict(ds_config.elo), "window": ds_config.window,
                                                 "lead_minutes": ds_config.lead_minutes,
                                                 "min_history": ds_config.min_history}})
