"""Phase 2 probability models with a common fit / predict interface.

Every model is fitted on a frame of training rows and predicts P(home win) for
another frame. `predict` never needs outcome columns, so callers can (and the
walk-forward does) pass frames with results stripped out.

Hyperparameters are chosen *inside* the training rows: the last
`validation_frac` of training dates acts as an inner validation set, the best
setting is refitted on all training rows. Nothing here sees later games.

  EloModel        Model A: Elo probability; K / home advantage / carryover chosen on
                  the training window's own one-step-ahead (already out-of-sample) predictions
  LogisticModel   Model B: median imputation + standardisation + L2 logistic regression
  GBMModel        Model C: histogram gradient boosting (scikit-learn; LightGBM optional)
  Calibrator      Platt (sigmoid on logit) or isotonic, fitted on a separate window
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .elo import EloConfig, compute_elo
from .evaluation import per_game_log_loss

log = logging.getLogger(__name__)

P_CLIP = 1e-4
LABEL = "home_win"


def clip_prob(p):
    return np.clip(np.asarray(p, dtype=float), P_CLIP, 1 - P_CLIP)


def logit(p):
    p = clip_prob(p)
    return np.log(p / (1 - p))


def inner_split(dates: pd.Series, validation_frac: float):
    """Boolean masks (fit, validate) splitting by date: the last fraction of dates validates."""
    unique = np.sort(dates.unique())
    n_val = max(1, int(round(len(unique) * validation_frac)))
    if len(unique) - n_val < 1:
        raise ValueError("not enough training dates for an inner validation split")
    cut = unique[-n_val]
    return (dates < cut).to_numpy(), (dates >= cut).to_numpy()


def _mean_log_loss(y, p) -> float:
    return float(per_game_log_loss(y, clip_prob(p)).mean())


# ---------------------------------------------------------------- Model A: Elo
ELO_GRID = {"k": (15.0, 20.0, 25.0), "home_advantage": (70.0, 100.0), "season_carryover": (0.6, 0.75)}


def elo_grid_probabilities(games: pd.DataFrame, grid: dict = ELO_GRID, base: EloConfig = EloConfig()) -> dict:
    """Online Elo P(home win) for every game under each grid config.

    Each probability uses only earlier dates' results, so selecting a config by its
    log loss over a training window uses no information from after that window.
    """
    out = {}
    for k, hca, carry in itertools.product(grid["k"], grid["home_advantage"], grid["season_carryover"]):
        cfg = EloConfig(k=k, home_advantage=hca, initial_rating=base.initial_rating,
                        season_carryover=carry, use_margin=base.use_margin)
        elo, _ = compute_elo(games, cfg)
        out[cfg] = pd.Series(elo["elo_home_prob"].to_numpy(), index=games["game_key"].to_numpy())
    return out


@dataclass
class EloModel:
    """Selects the Elo config with the lowest training-window log loss; predicts by game_key lookup."""
    grid_probs: dict
    name: str = "elo"
    config: EloConfig | None = None
    selection: dict = field(default_factory=dict)

    def fit(self, train: pd.DataFrame):
        y = train[LABEL].to_numpy()
        scores = {cfg: _mean_log_loss(y, probs.loc[train["game_key"]].to_numpy())
                  for cfg, probs in self.grid_probs.items()}
        self.config = min(scores, key=scores.get)
        self.selection = {"train_log_loss": scores[self.config]}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return clip_prob(self.grid_probs[self.config].loc[df["game_key"]].to_numpy())

    @property
    def params(self) -> dict:
        c = self.config
        return {"k": c.k, "home_advantage": c.home_advantage, "season_carryover": c.season_carryover}


# ---------------------------------------------------------- Model B: logistic
@dataclass
class LogisticModel:
    features: list
    C_grid: tuple = (0.001, 0.003, 0.01, 0.03, 0.1, 1.0)
    validation_frac: float = 0.2
    name: str = "logistic"
    C: float | None = None
    model: object = None
    selection: dict = field(default_factory=dict)

    def _make(self, C):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=C, max_iter=2000))

    def fit(self, train: pd.DataFrame):
        X, y = train[self.features], train[LABEL].to_numpy()
        fit_m, val_m = inner_split(train["date"], self.validation_frac)
        scores = {}
        for C in self.C_grid:
            m = self._make(C).fit(X[fit_m], y[fit_m])
            scores[C] = _mean_log_loss(y[val_m], m.predict_proba(X[val_m])[:, 1])
        self.C = min(scores, key=scores.get)
        self.selection = {"inner_validation_log_loss": {str(k): v for k, v in scores.items()}}
        self.model = self._make(self.C).fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return clip_prob(self.model.predict_proba(df[self.features])[:, 1])

    @property
    def params(self) -> dict:
        return {"C": self.C}

    def importance(self) -> dict:
        coef = self.model[-1].coef_[0]
        return dict(zip(self.features, np.round(coef, 5).tolist()))


# ------------------------------------------------------- Model C: boosting
GBM_GRID = (
    {"learning_rate": 0.03, "max_iter": 300, "max_leaf_nodes": 8, "min_samples_leaf": 200, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 4, "min_samples_leaf": 100, "l2_regularization": 1.0},
    {"learning_rate": 0.02, "max_iter": 400, "max_leaf_nodes": 16, "min_samples_leaf": 300, "l2_regularization": 5.0},
)
# Constrain features whose direction is not in doubt (higher = better for the home team).
MONOTONE_INCREASING = ("elo_diff", "adj_margin_diff", "home_off_vs_away_def")
MONOTONE_DECREASING = ("away_off_vs_home_def",)


@dataclass
class GBMModel:
    features: list
    grid: tuple = GBM_GRID
    validation_frac: float = 0.2
    seed: int = 0
    backend: str = "sklearn"
    name: str = "gbm"
    best: dict | None = None
    model: object = None
    selection: dict = field(default_factory=dict)

    def _monotone(self):
        return [1 if f in MONOTONE_INCREASING else -1 if f in MONOTONE_DECREASING else 0 for f in self.features]

    def _make(self, params):
        if self.backend == "lightgbm":
            try:
                import lightgbm as lgb
            except (ImportError, OSError) as exc:
                raise RuntimeError("lightgbm backend requested but lightgbm cannot be imported "
                                   "(on macOS it needs `brew install libomp`)") from exc
            return lgb.LGBMClassifier(
                learning_rate=params["learning_rate"], n_estimators=params["max_iter"],
                num_leaves=params["max_leaf_nodes"], min_child_samples=params["min_samples_leaf"],
                reg_lambda=params["l2_regularization"], monotone_constraints=self._monotone(),
                random_state=self.seed, deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1)
        return HistGradientBoostingClassifier(**params, early_stopping=False, monotonic_cst=self._monotone(),
                                              random_state=self.seed)

    def fit(self, train: pd.DataFrame):
        X, y = train[self.features], train[LABEL].to_numpy()
        fit_m, val_m = inner_split(train["date"], self.validation_frac)
        scores = []
        for params in self.grid:
            m = self._make(params).fit(X[fit_m], y[fit_m])
            scores.append(_mean_log_loss(y[val_m], m.predict_proba(X[val_m])[:, 1]))
        best_i = int(np.argmin(scores))
        self.best = dict(self.grid[best_i])
        self.selection = {"inner_validation_log_loss": scores, "chosen_index": best_i}
        self.model = self._make(self.best).fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return clip_prob(self.model.predict_proba(df[self.features])[:, 1])

    @property
    def params(self) -> dict:
        return dict(self.best or {}, backend=self.backend)

    def importance(self, df: pd.DataFrame | None = None, n_repeats: int = 3) -> dict:
        """Permutation importance (increase in log loss) on `df`, which should be held-out rows."""
        if df is None or df.empty:
            return {}
        from sklearn.inspection import permutation_importance
        r = permutation_importance(self.model, df[self.features], df[LABEL], scoring="neg_log_loss",
                                   n_repeats=n_repeats, random_state=self.seed)
        return dict(zip(self.features, np.round(r.importances_mean, 6).tolist()))


# -------------------------------------------------------------- calibration
@dataclass
class Calibrator:
    method: str = "platt"   # "none" | "platt" | "isotonic"
    model: object = None

    def fit(self, p, y):
        if self.method == "platt":
            self.model = LogisticRegression(C=1e4, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=P_CLIP, y_max=1 - P_CLIP).fit(p, y)
        elif self.method != "none":
            raise ValueError(f"unknown calibration method {self.method!r}")
        return self

    def transform(self, p):
        p = clip_prob(p)
        if self.method == "platt":
            return clip_prob(self.model.predict_proba(logit(p).reshape(-1, 1))[:, 1])
        if self.method == "isotonic":
            return clip_prob(self.model.predict(p))
        return p

    @property
    def params(self) -> dict:
        if self.method == "platt":
            return {"method": "platt", "slope": float(self.model.coef_[0][0]),
                    "intercept": float(self.model.intercept_[0])}
        return {"method": self.method}


@dataclass
class CalibratedModel:
    """Base model fitted on the fit window, calibrator on the later calibration window."""
    base: object
    calibrator: Calibrator

    @property
    def name(self):
        return self.base.name

    def fit(self, fit_rows: pd.DataFrame, cal_rows: pd.DataFrame):
        self.base.fit(fit_rows)
        self.calibrator.fit(self.base.predict(cal_rows), cal_rows[LABEL].to_numpy())
        return self

    def predict_raw(self, df):
        return self.base.predict(df)

    def predict(self, df):
        return self.calibrator.transform(self.base.predict(df))

    @property
    def params(self):
        return {"base": self.base.params, "calibration": self.calibrator.params}


def ensemble_mean(*probs) -> np.ndarray:
    """Equal-weight average in logit space; no weights are fitted, so nothing can overfit here."""
    return clip_prob(1 / (1 + np.exp(-np.mean([logit(p) for p in probs], axis=0))))
