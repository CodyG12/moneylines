"""Early-season Elo blending: P = w * P_elo + (1 - w) * P_ml, with w fixed or varying by season stage.

Weights are always chosen on earlier seasons and applied to a later one (forward-chaining by
season over development walk-forward predictions, which are themselves out-of-sample). So every
blended probability scored here comes from weights that never saw that season's outcomes. The
final holdout is not used to choose anything; `confirm_on_holdout` scores the pre-chosen blend
there once, for the record.

Methods
  fixed          one w for all games
  games_played   one w per bucket of min(home, away) games played this season
  season_phase   one w per bucket of days since the season's first game
  decay          w(g) = w0 * exp(-g / tau), g = min games played
  early_only     one w while g < EARLY_GAMES, pure ML after (the "early season" hypothesis itself)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataError
from .evaluation import compare_brier, compare_log_loss, per_game_log_loss, score

W_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
GP_EDGES = (0, 5, 10, 20, 40, 10_000)
PHASE_EDGES = (0, 21, 45, 90, 10_000)          # days since the season's first game
DECAY_W0 = np.round(np.arange(0.0, 1.0001, 0.1), 2)
DECAY_TAU = (3.0, 5.0, 10.0, 20.0, 40.0, 80.0)
SEGMENTS = {"early (<15 games)": (0, 15), "mid (15-49)": (15, 50), "late (50+)": (50, 10_000)}
EARLY_GAMES = 15
METHODS = ("fixed", "games_played", "season_phase", "decay", "early_only")


def games_played(df: pd.DataFrame) -> np.ndarray:
    return np.minimum(df["home_season_games"].to_numpy(float), df["away_season_games"].to_numpy(float))


def season_day(df: pd.DataFrame) -> np.ndarray:
    """Days since the season's first game; live callers pass `season_day` computed from the full
    schedule, since a few upcoming rows do not show when the season started."""
    if "season_day" in df:
        return df["season_day"].to_numpy(float)
    d = pd.to_datetime(df["date"])
    first = d.groupby(df["season"]).transform("min")
    return (d - first).dt.days.to_numpy(float)


def _bucket(x: np.ndarray, edges) -> np.ndarray:
    return np.clip(np.searchsorted(np.asarray(edges[1:-1], float), x, side="right"), 0, len(edges) - 2)


def blend(p_elo, p_ml, w) -> np.ndarray:
    w = np.clip(np.asarray(w, float), 0.0, 1.0)
    return w * np.asarray(p_elo, float) + (1 - w) * np.asarray(p_ml, float)


def _best_w(y, pe, pm) -> float:
    losses = [per_game_log_loss(y, blend(pe, pm, w)).mean() for w in W_GRID]
    return float(W_GRID[int(np.argmin(losses))])


def fit(df: pd.DataFrame, method: str, base_col: str) -> dict:
    """Choose blend parameters by minimum log loss on `df` (training seasons only)."""
    y, pe, pm = df["home_win"].to_numpy(float), df["p_elo"].to_numpy(float), df[base_col].to_numpy(float)
    if method == "fixed":
        return {"method": method, "w": _best_w(y, pe, pm)}
    if method in ("games_played", "season_phase"):
        edges = GP_EDGES if method == "games_played" else PHASE_EDGES
        x = games_played(df) if method == "games_played" else season_day(df)
        b = _bucket(x, edges)
        ws = [(_best_w(y[b == k], pe[b == k], pm[b == k]) if (b == k).sum() >= 30 else 0.0)
              for k in range(len(edges) - 1)]
        return {"method": method, "edges": list(edges), "w": ws}
    if method == "decay":
        g = games_played(df)
        best = min(((w0, tau) for w0 in DECAY_W0 for tau in DECAY_TAU),
                   key=lambda wt: per_game_log_loss(y, blend(pe, pm, wt[0] * np.exp(-g / wt[1]))).mean())
        return {"method": method, "w0": float(best[0]), "tau": float(best[1])}
    if method == "early_only":
        e = games_played(df) < EARLY_GAMES
        return {"method": method, "games": EARLY_GAMES, "w": _best_w(y[e], pe[e], pm[e]) if e.sum() >= 30 else 0.0}
    raise ValueError(f"unknown blend method {method!r}")


def weights(df: pd.DataFrame, params: dict) -> np.ndarray:
    m = params["method"]
    if m == "fixed":
        return np.full(len(df), params["w"])
    if m in ("games_played", "season_phase"):
        x = games_played(df) if m == "games_played" else season_day(df)
        return np.asarray(params["w"], float)[_bucket(x, params["edges"])]
    if m == "decay":
        return params["w0"] * np.exp(-games_played(df) / params["tau"])
    if m == "early_only":
        return np.where(games_played(df) < params["games"], params["w"], 0.0)
    raise ValueError(f"unknown blend method {m!r}")


def apply(df: pd.DataFrame, params: dict, base_col: str) -> np.ndarray:
    return blend(df["p_elo"], df[base_col], weights(df, params))


def params_id(params: dict, base_col: str) -> str:
    return "blend-" + hashlib.sha256(json.dumps({"p": params, "base": base_col}, sort_keys=True).encode()
                                     ).hexdigest()[:8]


def forward_chain(preds: pd.DataFrame, base_col: str, methods=METHODS, min_train_seasons: int = 1):
    """Out-of-sample blended probabilities: season s uses parameters fitted on seasons < s.
    Returns (frame with p_blend_<method> columns for the evaluated seasons, per-season params)."""
    seasons = sorted(preds["season"].unique())
    if len(seasons) <= min_train_seasons:
        raise DataError(f"blend study needs more than {min_train_seasons} season(s) of development predictions, "
                        f"got {len(seasons)}")
    out, chosen = [], []
    for s in seasons[min_train_seasons:]:
        train, test = preds[preds["season"] < s], preds[preds["season"] == s].copy()
        for m in methods:
            p = fit(train, m, base_col)
            test[f"p_blend_{m}"] = apply(test, p, base_col)
            chosen.append({"season": int(s), **p})
        out.append(test)
    return pd.concat(out, ignore_index=True), chosen


def segment_of(df: pd.DataFrame) -> pd.Series:
    g = games_played(df)
    lab = pd.Series("", index=df.index)
    for name, (lo, hi) in SEGMENTS.items():
        lab[(g >= lo) & (g < hi)] = name
    return lab


def compare(df: pd.DataFrame, cols: list, base_col: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """Scores and paired differences vs the ML base, overall and by season segment."""
    y = df["home_win"].to_numpy(float)

    def block(d):
        yy = d["home_win"].to_numpy(float)
        res = {c: score(yy, d[c]) for c in cols}
        for c in cols:
            if c != base_col and len(d) > 1:
                res[c]["log_loss_diff_vs_base"] = compare_log_loss(yy, d[c], d[base_col], d["date"], n_boot, seed)
                res[c]["brier_diff_vs_base"] = compare_brier(yy, d[c], d[base_col], d["date"], n_boot, seed)
        return res

    seg = segment_of(df)
    return {"n": int(len(y)), "overall": block(df),
            "by_segment": {name: block(df[seg == name]) for name in SEGMENTS if (seg == name).any()}}


@dataclass
class BlendStudy:
    report: dict
    oos: pd.DataFrame


def study(dev_preds: pd.DataFrame, base_col: str, n_boot: int = 1000, seed: int = 0,
          alpha_methods: int | None = None) -> BlendStudy:
    """Forward-chained blend study on development predictions plus the enable decision.

    Decision rule, fixed before looking at results: enable the method with the lowest
    out-of-sample log loss only if (a) its paired log-loss difference vs the ML base has a 95%
    CI entirely below 0 and (b) it does not lose to the base in the early-season segment.
    Several methods are tried, so the winner's estimate is optimistic; that is reported."""
    df = dev_preds.dropna(subset=["p_elo", base_col, "home_win"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    oos, chosen = forward_chain(df, base_col)
    cols = [base_col, "p_elo"] + [f"p_blend_{m}" for m in METHODS]
    rep = compare(oos, cols, base_col, n_boot, seed)
    ranked = sorted(METHODS, key=lambda m: rep["overall"][f"p_blend_{m}"]["log_loss"])
    best = ranked[0]
    d = rep["overall"][f"p_blend_{best}"]["log_loss_diff_vs_base"]
    early = rep["by_segment"].get("early (<15 games)", {}).get(f"p_blend_{best}", {})
    early_diff = early.get("log_loss_diff_vs_base", {}).get("estimate")
    enable = bool(d["ci_high"] < 0 and (early_diff is None or early_diff <= 0))
    final_params = fit(df, best, base_col)  # production parameters: all development seasons
    rep.update({
        "base_col": base_col, "methods_ranked": ranked, "best_method": best,
        "evaluated_seasons": sorted(int(s) for s in oos["season"].unique()),
        "per_season_params": chosen, "final_params": final_params,
        "blend_id": params_id(final_params, base_col), "enabled": enable,
        "decision_rule": "enable the lowest-OOS-log-loss method only if its log-loss difference vs the base has "
                         "a 95% CI below 0 and it is not worse in the early-season segment",
        "multiple_testing_note": f"{len(METHODS)} methods compared on the same out-of-sample games; the best "
                                 "one's improvement is optimistic by selection",
    })
    return BlendStudy(rep, oos)


def confirm_on_holdout(holdout_preds: pd.DataFrame, params: dict, base_col: str, n_boot: int = 1000,
                       seed: int = 0) -> dict:
    """Score the already-chosen blend on the final holdout once. Nothing is re-fitted."""
    df = holdout_preds.dropna(subset=["p_elo", base_col, "home_win"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["p_blend"] = apply(df, params, base_col)
    return compare(df, [base_col, "p_elo", "p_blend"], base_col, n_boot, seed)
