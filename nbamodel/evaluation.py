"""Probability scoring, cluster-bootstrap uncertainty, and calibration tables.

Games on the same date share conditions (and in betting, bankroll), so
confidence intervals resample whole dates rather than individual games.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-15


def per_game_log_loss(y, p) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def per_game_brier(y, p) -> np.ndarray:
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def score(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "n": int(len(y)),
        "log_loss": float(per_game_log_loss(y, p).mean()),
        "brier": float(per_game_brier(y, p).mean()),
        "accuracy": float(((p >= 0.5) == (y == 1)).mean()),
    }


def cluster_bootstrap_ratio(numerator, denominator, clusters, n_boot: int = 1000,
                            seed: int = 0, alpha: float = 0.05) -> dict:
    """CI for sum(numerator)/sum(denominator), resampling whole clusters.

    With denominator == 1 this is a cluster-bootstrapped mean.
    """
    frame = pd.DataFrame({"num": numerator, "den": denominator, "c": clusters})
    agg = frame.groupby("c")[["num", "den"]].sum()
    nums, dens = agg["num"].to_numpy(), agg["den"].to_numpy()
    k = len(agg)
    point = float(nums.sum() / dens.sum())
    if k < 2:
        return {"estimate": point, "ci_low": float("nan"), "ci_high": float("nan"), "n_clusters": k}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(n_boot, k))
    stats = nums[draws].sum(axis=1) / dens[draws].sum(axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi), "n_clusters": k}


def compare_log_loss(y, p_model, p_baseline, clusters, n_boot=1000, seed=0) -> dict:
    """Mean log-loss difference (model - baseline); negative favours the model."""
    diff = per_game_log_loss(y, p_model) - per_game_log_loss(y, p_baseline)
    return cluster_bootstrap_ratio(diff, np.ones_like(diff), clusters, n_boot, seed)


def wilson_interval(successes, n, z: float = 1.96):
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    phat = successes / n
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table. Wilson intervals treat games as independent, so read
    them as approximate."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            continue
        wins = y[mask].sum()
        lo, hi = wilson_interval(wins, n)
        rows.append({
            "bin_low": edges[b], "bin_high": edges[b + 1], "n": n,
            "mean_predicted": float(p[mask].mean()), "observed_rate": float(wins / n),
            "observed_ci_low": float(lo), "observed_ci_high": float(hi),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Phase 2
def roc_auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=float)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Count-weighted mean |observed rate - mean predicted| over equal-width bins."""
    t = calibration_table(y, p, n_bins)
    if t.empty:
        return float("nan")
    return float((t["n"] * (t["observed_rate"] - t["mean_predicted"]).abs()).sum() / t["n"].sum())


def calibration_slope_intercept(y, p) -> dict:
    """Logistic recalibration fit y ~ a + b * logit(p). Perfect calibration: a = 0, b = 1.
    b < 1 means predictions are too extreme; b > 1, too timid."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        return {"slope": float("nan"), "intercept": float("nan")}
    pc = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lp = np.log(pc / (1 - pc)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y)
    return {"slope": float(m.coef_[0][0]), "intercept": float(m.intercept_[0])}


def full_score(y, p, clusters=None, n_boot: int = 1000, seed: int = 0) -> dict:
    """Every headline metric for one model; CIs resample whole dates when `clusters` is given."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    s = score(y, p)
    s["roc_auc"] = roc_auc(y, p)
    s["ece"] = expected_calibration_error(y, p)
    s.update({f"calibration_{k}": v for k, v in calibration_slope_intercept(y, p).items()})
    s["mean_predicted"] = float(p.mean())
    s["observed_rate"] = float(y.mean())
    if clusters is not None and len(y) > 1:
        ones = np.ones(len(y))
        s["log_loss_ci"] = cluster_bootstrap_ratio(per_game_log_loss(y, p), ones, clusters, n_boot, seed)
        s["brier_ci"] = cluster_bootstrap_ratio(per_game_brier(y, p), ones, clusters, n_boot, seed)
        s["accuracy_ci"] = cluster_bootstrap_ratio(((p >= 0.5) == (y == 1)).astype(float), ones, clusters,
                                                   n_boot, seed)
    return s


def compare_brier(y, p_model, p_baseline, clusters, n_boot=1000, seed=0) -> dict:
    """Mean Brier difference (model - baseline); negative favours the model."""
    diff = per_game_brier(y, p_model) - per_game_brier(y, p_baseline)
    return cluster_bootstrap_ratio(diff, np.ones_like(diff), clusters, n_boot, seed)


def confusion_matrix(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y, dtype=float) == 1
    pred = np.asarray(p, dtype=float) >= threshold
    return {"tp_home_win": int((pred & y).sum()), "fp_home_win": int((pred & ~y).sum()),
            "tn_away_win": int((~pred & ~y).sum()), "fn_away_win": int((~pred & y).sum())}


def by_confidence(y, p, edges=(0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0001)) -> pd.DataFrame:
    """Accuracy of the favoured side by model confidence max(p, 1-p)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    conf = np.maximum(p, 1 - p)
    fav_won = np.where(p >= 0.5, y, 1 - y)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        wlo, whi = wilson_interval(fav_won[m].sum(), m.sum())
        rows.append({"conf_low": lo, "conf_high": min(hi, 1.0), "n": int(m.sum()),
                     "mean_confidence": float(conf[m].mean()), "favourite_win_rate": float(fav_won[m].mean()),
                     "ci_low": float(wlo), "ci_high": float(whi),
                     "log_loss": float(per_game_log_loss(y[m], p[m]).mean())})
    return pd.DataFrame(rows)


def by_group(df: pd.DataFrame, group_col: str, prob_cols: list, y_col: str = "home_win") -> pd.DataFrame:
    """Log loss / Brier / accuracy per group (season, month, team, ...) for each model."""
    rows = []
    for g, sub in df.groupby(group_col, sort=True):
        row = {group_col: g, "n": len(sub)}
        for c in prob_cols:
            s = score(sub[y_col], sub[c])
            row.update({f"{c}_log_loss": s["log_loss"], f"{c}_brier": s["brier"], f"{c}_accuracy": s["accuracy"]})
        rows.append(row)
    return pd.DataFrame(rows)


def by_side(df: pd.DataFrame, p_col: str, y_col: str = "home_win") -> pd.DataFrame:
    """Performance when the model favours the home team vs the away team."""
    rows = []
    for side, m in (("home_favoured", df[p_col] >= 0.5), ("away_favoured", df[p_col] < 0.5)):
        sub = df[m]
        if sub.empty:
            continue
        p_fav = np.where(sub[p_col] >= 0.5, sub[p_col], 1 - sub[p_col])
        won = np.where(sub[p_col] >= 0.5, sub[y_col], 1 - sub[y_col])
        rows.append({"side": side, "n": len(sub), "mean_favourite_prob": float(p_fav.mean()),
                     "favourite_win_rate": float(won.mean()),
                     "log_loss": float(per_game_log_loss(sub[y_col], sub[p_col]).mean())})
    return pd.DataFrame(rows)


def by_team(df: pd.DataFrame, p_col: str, y_col: str = "home_win") -> pd.DataFrame:
    """Per-team log loss over every game the team played, home or away."""
    ll = per_game_log_loss(df[y_col], df[p_col])
    long = pd.concat([pd.DataFrame({"team": df["home_team"].to_numpy(), "venue": "home", "ll": ll}),
                      pd.DataFrame({"team": df["away_team"].to_numpy(), "venue": "away", "ll": ll})])
    t = long.pivot_table(index="team", columns="venue", values="ll", aggfunc=["mean", "size"])
    t.columns = [f"{venue}_{stat}" for stat, venue in t.columns]
    t = t.rename(columns={"home_mean": "home_log_loss", "away_mean": "away_log_loss",
                          "home_size": "home_n", "away_size": "away_n"})
    return t.reset_index().sort_values("team")
