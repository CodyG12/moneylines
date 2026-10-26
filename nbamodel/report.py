"""Scoring a set of walk-forward predictions and choosing a champion model.

Champion rule (applied to *development* results only, never the final test):
lowest log loss among candidate models. The ensemble is eligible only if its
log loss is below both of its components'; otherwise the better single model wins.
Every comparison carries a date-clustered bootstrap CI, so "better" can be read
as "better beyond noise" only when the CI excludes zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import (by_confidence, by_group, by_side, by_team, calibration_table, compare_brier,
                         compare_log_loss, confusion_matrix, full_score, score)

MODEL_LABELS = {
    "p_home_base_rate": "Home base rate", "p_elo": "Elo", "p_logistic": "Logistic (calibrated)",
    "p_logistic_raw": "Logistic (uncalibrated)", "p_gbm": "Gradient boosting (calibrated)",
    "p_gbm_raw": "Gradient boosting (uncalibrated)", "p_ensemble": "Ensemble (logistic+GBM)",
    "market_home_prob_novig": "Market (no-vig)",
}
CANDIDATES = ("p_elo", "p_logistic", "p_logistic_raw", "p_gbm", "p_gbm_raw", "p_ensemble")
BASELINES = ("p_home_base_rate", "p_elo")


def prob_columns(preds: pd.DataFrame) -> list:
    return [c for c in MODEL_LABELS if c in preds.columns and c != "market_home_prob_novig"]


def evaluate_predictions(preds: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    y = preds["home_win"].to_numpy()
    clusters = preds["date"]
    cols = prob_columns(preds)
    report = {
        "n_games": int(len(preds)), "n_dates": int(preds["date"].nunique()),
        "date_range": [str(preds["date"].min().date()), str(preds["date"].max().date())],
        "seasons": sorted(int(s) for s in preds["season"].unique()),
        "models": {c: full_score(y, preds[c], clusters, n_boot, seed) for c in cols},
        "comparisons": {},
        "by_side": {c: by_side(preds, c).to_dict("records") for c in cols},
        "by_confidence": {c: by_confidence(y, preds[c]).to_dict("records") for c in cols},
        "confusion_matrix": {c: confusion_matrix(y, preds[c]) for c in cols},
        "calibration": {c: calibration_table(y, preds[c]).to_dict("records") for c in cols},
    }
    for c in cols:
        for b in BASELINES:
            if b in preds.columns and b != c:
                report["comparisons"][f"{c}_minus_{b}"] = {
                    "log_loss": compare_log_loss(y, preds[c], preds[b], clusters, n_boot, seed),
                    "brier": compare_brier(y, preds[c], preds[b], clusters, n_boot, seed)}
    priced = preds["market_home_prob_novig"].notna() if "market_home_prob_novig" in preds else pd.Series(False)
    if priced.any():
        hp = preds[priced]
        report["market"] = {
            "n_games_with_odds": int(priced.sum()),
            "odds_timestamps_complete": bool(hp["odds_timestamp"].notna().all()) if "odds_timestamp" in hp else False,
            "market_novig": full_score(hp["home_win"], hp["market_home_prob_novig"], hp["date"], n_boot, seed),
            "models_on_same_games": {c: score(hp["home_win"], hp[c]) for c in cols},
            "model_minus_market_log_loss": {
                c: compare_log_loss(hp["home_win"], hp[c], hp["market_home_prob_novig"], hp["date"], n_boot, seed)
                for c in cols},
        }
    else:
        report["market"] = None
    return report


def tables(preds: pd.DataFrame) -> dict:
    """Breakdown tables written as CSVs next to the report."""
    cols = prob_columns(preds)
    p = preds.assign(month=preds["date"].dt.to_period("M").astype(str))
    out = {"by_season": by_group(p, "season", cols), "by_month": by_group(p, "month", cols)}
    for c in ("p_logistic", "p_gbm", "p_elo"):
        if c in preds:
            out[f"by_team_{c[2:]}"] = by_team(preds, c)
    return out


def choose_champion(report: dict) -> dict:
    m = report["models"]
    ll = {c: m[c]["log_loss"] for c in CANDIDATES if c in m}
    pool = dict(ll)
    note = None
    if "p_ensemble" in pool:
        parts = [ll[c] for c in ("p_logistic", "p_gbm") if c in ll]
        if not parts or pool["p_ensemble"] >= min(parts):
            pool.pop("p_ensemble")
            note = "ensemble did not beat both components out of sample; not eligible"
    champion = min(pool, key=pool.get)
    ranked = sorted(pool.items(), key=lambda kv: kv[1])
    runner_up = ranked[1][0] if len(ranked) > 1 else None
    out = {"champion": champion, "log_loss": pool[champion], "ranking": ranked, "rule": __doc__.split("\n\n")[1]}
    if note:
        out["note"] = note
    if runner_up:
        key = f"{champion}_minus_{runner_up}"
        out["vs_runner_up"] = {"runner_up": runner_up, "log_loss_diff": ll[champion] - ll[runner_up]}
        out["vs_runner_up"]["comparison_ci"] = report["comparisons"].get(key)
    return out


def champion_spec(champion_col: str) -> dict:
    """Translate a prediction column into what the production trainer should fit."""
    name = champion_col[2:]
    calibrated = not name.endswith("_raw")
    base = name.replace("_raw", "")
    if base == "ensemble":
        return {"model": "ensemble", "components": ["logistic", "gbm"], "calibrated": True}
    return {"model": base, "calibrated": calibrated and base != "elo"}


def fmt_ci(d: dict | None) -> str:
    if not d:
        return "n/a"
    if np.isnan(d.get("ci_low", np.nan)):
        return f"{d['estimate']:.4f}"
    return f"{d['estimate']:.4f} [{d['ci_low']:.4f}, {d['ci_high']:.4f}]"


def summary_text(report: dict, title: str) -> str:
    lines = [f"{title}: {report['n_games']} games, {report['n_dates']} dates, "
             f"{report['date_range'][0]} -> {report['date_range'][1]}",
             f"  {'model':<34} {'log loss [95% CI]':<28} {'brier':>7} {'acc':>6} {'auc':>6} {'ece':>6} {'slope':>6}"]
    for c, s in sorted(report["models"].items(), key=lambda kv: kv[1]["log_loss"]):
        lines.append(f"  {MODEL_LABELS.get(c, c):<34} {fmt_ci(s.get('log_loss_ci')):<28} {s['brier']:>7.4f} "
                     f"{s['accuracy']:>6.3f} {s['roc_auc']:>6.3f} {s['ece']:>6.3f} {s['calibration_slope']:>6.2f}")
    for key in [k for k in report["comparisons"] if k.endswith("_minus_p_elo")]:
        if key in report["comparisons"]:
            lines.append(f"  {key.replace('p_', '')} log loss: {fmt_ci(report['comparisons'][key]['log_loss'])}"
                         "  (negative favours the first model)")
    if report.get("market"):
        mk = report["market"]
        lines.append(f"  market (no-vig) on {mk['n_games_with_odds']} priced games: "
                     f"log loss {mk['market_novig']['log_loss']:.4f}")
    else:
        lines.append("  market comparison: no historical odds supplied")
    return "\n".join(lines)
