"""Train the production model on all completed games and predict upcoming ones.

The champion (chosen on development walk-forward results, see report.choose_champion)
determines which probability is published. Elo, logistic and boosting are all fitted
anyway: their spread is reported as a model-disagreement indicator of uncertainty.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd
import sklearn

from .data import DataError
from .dataset import DatasetConfig, build_training_dataset
from .elo import EloConfig, compute_elo
from .features import FEATURE_DEFINITIONS
from .leakage import outcome_columns
from .models import clip_prob, elo_grid_probabilities, ensemble_mean
from .odds import fair_american_odds, fair_decimal_odds, overround, remove_vig_two_way
from .registry import frame_sha256, make_version
from .walkforward import WalkForwardConfig, fit_models

log = logging.getLogger(__name__)


@dataclass
class ProductionModel:
    spec: dict
    features: list
    ds_config: DatasetConfig
    elo_config: EloConfig
    fitted: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        m = self.spec["model"]
        return m if m in ("elo", "ensemble") or self.spec.get("calibrated", True) else f"{m}_raw"

    def component_probs(self, rows: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
        X = rows.drop(columns=[c for c in outcome_columns(rows)])
        elo, _ = compute_elo(games, self.elo_config)
        elo_p = pd.Series(elo["elo_home_prob"].to_numpy(), index=games["game_key"].to_numpy())
        out = pd.DataFrame({"p_elo": clip_prob(elo_p.loc[X["game_key"]].to_numpy())}, index=X.index)
        for name in ("logistic", "gbm"):
            out[f"p_{name}"] = self.fitted[name].predict(X)
            out[f"p_{name}_raw"] = self.fitted[name].predict_raw(X)
        out["p_ensemble"] = ensemble_mean(out["p_logistic"], out["p_gbm"])
        return out

    def predict(self, rows: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
        comp = self.component_probs(rows, games)
        comp["p_home"] = comp[f"p_{self.name}"]
        comp["model_spread"] = comp[["p_elo", "p_logistic", "p_gbm"]].max(axis=1) - comp[
            ["p_elo", "p_logistic", "p_gbm"]].min(axis=1)
        return comp


def train_production(games: pd.DataFrame, spec: dict, wf: WalkForwardConfig, ds_config: DatasetConfig,
                     reference: dict | None = None, source_path: str | None = None,
                     now: pd.Timestamp | None = None):
    """Fit on every train-eligible completed game. Returns (ProductionModel, metadata)."""
    now = now or pd.Timestamp.now(tz="UTC")
    ds = build_training_dataset(games, ds_config)
    train = ds[ds["train_eligible"]]
    if train.empty:
        raise DataError("no train-eligible games")
    cfg = replace(wf, models=("elo", "logistic", "gbm"))
    fitted = fit_models(train, cfg, elo_grid_probabilities(games))
    w = fitted.pop("_windows")
    model = ProductionModel(spec=spec, features=list(cfg.features), ds_config=ds_config,
                            elo_config=fitted["elo"].config, fitted={k: fitted[k] for k in ("logistic", "gbm")})
    fingerprint = {"spec": spec, "wf": cfg.to_dict(), "data": frame_sha256(train[["game_key", "home_win"]])}
    meta = {
        "version": make_version(model.name, now, fingerprint),
        "model": model.name, "spec": spec, "trained_at": now.isoformat(),
        "features": model.features, "feature_definitions": {f: FEATURE_DEFINITIONS.get(f) for f in model.features},
        "training": {"n_rows": int(len(train)), "start": str(train["date"].min().date()),
                     "end": str(train["date"].max().date()), "n_fit": int(len(w["fit_rows"])),
                     "n_calibration": int(len(w["cal_rows"])), "calibration_start": str(w["cal_start"].date()),
                     "data_sha256": fingerprint["data"], "source_file": source_path},
        "params": {"elo": fitted["elo"].params, "logistic": fitted["logistic"].params, "gbm": fitted["gbm"].params},
        "logistic_coefficients": fitted["logistic"].base.importance(),
        "walk_forward_config": cfg.to_dict(),
        "dataset_config": {"elo": asdict(ds_config.elo), "window": ds_config.window,
                           "lead_minutes": ds_config.lead_minutes, "min_history": ds_config.min_history},
        "reference_metrics": reference or {},
        "seed": cfg.seed, "sklearn_version": sklearn.__version__,
    }
    return model, meta


def _key_factors(model: ProductionModel, rows: pd.DataFrame, top: int = 3) -> list:
    """Largest standardized logistic contributions (coef x z-score), as readable strings."""
    pipe = model.fitted["logistic"].base.model
    X = pipe[:-1].transform(rows[model.features])
    contrib = X * pipe[-1].coef_[0]
    out = []
    for r in contrib:
        order = np.argsort(-np.abs(r))[:top]
        out.append("; ".join(f"{model.features[j]} {'+' if r[j] > 0 else '-'}{abs(r[j]):.2f}" for j in order))
    return out


def upcoming_rows(ds: pd.DataFrame, horizon_days: int, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Scheduled games still to be played, from the latest played date on, within the horizon.

    Unplayed games dated before the latest played date are missing results (postponed or not yet
    updated), not upcoming games. Games whose tip-off has passed are excluded: no pregame call.
    """
    played_dates = ds.loc[ds["played"], "date"]
    floor = played_dates.max() if not played_dates.empty else ds["date"].min()
    up = ds[~ds["played"] & (ds["date"] >= floor)]
    if now is not None:
        tip = pd.to_datetime(up["tipoff_utc"], utc=True, errors="coerce")
        up = up[tip.isna() | (tip > now)]
    if up.empty:
        return up
    horizon = up["date"].min() + pd.Timedelta(days=horizon_days - 1)
    return up[up["date"] <= horizon]


def predict_upcoming(games: pd.DataFrame, model: ProductionModel, meta: dict, horizon_days: int = 3,
                     now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = now or pd.Timestamp.now(tz="UTC")
    ds = build_training_dataset(games, model.ds_config)
    rows = upcoming_rows(ds, horizon_days, now)
    if rows.empty:
        return rows
    probs = model.predict(rows, games)
    out = rows[["game_key", "game_id_source", "season", "date", "tipoff_utc", "home_team", "away_team",
                "neutral_site", "home_season_games", "away_season_games"]].copy()
    out = pd.concat([out, probs], axis=1)
    out["p_away"] = 1 - out["p_home"]
    out["fair_home_decimal"] = fair_decimal_odds(out["p_home"])
    out["fair_away_decimal"] = fair_decimal_odds(out["p_away"])
    out["fair_home_american"] = np.round(fair_american_odds(out["p_home"])).astype(int)
    out["fair_away_american"] = np.round(fair_american_odds(out["p_away"])).astype(int)
    for c in ("home_odds", "away_odds", "odds_timestamp", "odds_book"):
        out[c] = rows[c] if c in rows else np.nan
    priced = out["home_odds"].notna() & out["away_odds"].notna()
    for c in ("market_home_prob_novig", "market_overround", "edge_home", "ev_home", "ev_away"):
        out[c] = np.nan
    if priced.any():
        h, a = out.loc[priced, "home_odds"], out.loc[priced, "away_odds"]
        nv = remove_vig_two_way(h, a)[0]
        out.loc[priced, "market_home_prob_novig"] = nv
        out.loc[priced, "market_overround"] = overround(h, a)
        out.loc[priced, "edge_home"] = out.loc[priced, "p_home"] - nv
        out.loc[priced, "ev_home"] = out.loc[priced, "p_home"] * h - 1
        out.loc[priced, "ev_away"] = out.loc[priced, "p_away"] * a - 1
    out["low_history"] = (rows["home_season_games"] < model.ds_config.min_history) | (
        rows["away_season_games"] < model.ds_config.min_history)
    out["key_factors"] = _key_factors(model, rows)
    out["prediction_timestamp"] = now.isoformat()
    out["model_version"] = meta["version"]
    out["features_asof"] = rows["features_asof"].dt.date.astype(str)
    out["latest_result_in_data"] = str(ds.loc[ds["played"], "date"].max().date())
    out["source_fetched_at"] = rows["source_fetched_at"]
    out["feature_snapshot"] = [json.dumps({f: (None if pd.isna(v) else round(float(v), 6)) for f, v in r.items()})
                               for r in rows[model.features].to_dict("records")]
    return out.reset_index(drop=True)
