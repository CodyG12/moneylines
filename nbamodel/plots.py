"""Static evaluation charts (PNG, matplotlib, headless).

Each model keeps one colour everywhere (colour follows the entity). Charts show
at most four model series, plus the reference line or baseline in grey.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import calibration_table, per_game_log_loss
from .report import MODEL_LABELS

COLORS = {"p_elo": "#2a78d6", "p_logistic_raw": "#eb6834", "p_gbm_raw": "#1baf7a", "p_logistic": "#eda100",
          "p_ensemble": "#6250d6", "p_gbm": "#008300", "market_home_prob_novig": "#e87ba4",
          "p_home_base_rate": "#8a8984"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MAIN = ("p_elo", "p_logistic_raw", "p_logistic", "p_gbm_raw")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2})
    return plt


def _legend_below(ax):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)


def _models(preds, cols=MAIN):
    return [c for c in cols if c in preds.columns and preds[c].notna().any()]


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(path)


def reliability_diagram(preds, path, title="Reliability diagram"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.plot([0, 1], [0, 1], color=COLORS["p_home_base_rate"], lw=1, ls="--", label="Perfect calibration")
    for c in _models(preds, MAIN + ("market_home_prob_novig",)):
        sub = preds[preds[c].notna()]
        t = calibration_table(sub["home_win"], sub[c])
        ax.plot(t["mean_predicted"], t["observed_rate"], marker="o", ms=5, color=COLORS[c],
                label=MODEL_LABELS[c])
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted P(home win)", ylabel="Observed home win rate", title=title)
    ax.legend(loc="upper left")
    return _save(fig, path)


def probability_distribution(preds, path):
    plt = _plt()
    cols = _models(preds)
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        ax.hist(preds[c], bins=np.linspace(0, 1, 26), color=COLORS[c], edgecolor=SURFACE, linewidth=1)
        ax.set(title=MODEL_LABELS[c], xlabel="Predicted P(home win)")
    axes[0].set_ylabel("Games")
    return _save(fig, path)


def metric_by_season(by_season: pd.DataFrame, path, metric="log_loss"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in ("p_home_base_rate",) + MAIN:
        col = f"{c}_{metric}"
        if col in by_season:
            ax.plot(by_season["season"], by_season[col], marker="o", ms=5, color=COLORS[c],
                    ls="--" if c == "p_home_base_rate" else "-", lw=1 if c == "p_home_base_rate" else 2,
                    label=MODEL_LABELS[c])
    ax.set(xlabel="Season (start year)", ylabel=metric.replace("_", " ").title(),
           title=f"{metric.replace('_', ' ').title()} by season (lower is better)")
    _legend_below(ax)
    return _save(fig, path)


def rolling_log_loss(preds, path, window=None):
    """Monthly mean log loss over time; lines break across off-seasons. (`window` kept for compatibility.)"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    p = preds.assign(month=preds["date"].dt.to_period("M").dt.to_timestamp())
    for c in _models(p):
        m = p.assign(ll=per_game_log_loss(p["home_win"], p[c])).groupby("month")["ll"].mean()
        gap = m.index.to_series().diff() > pd.Timedelta(days=62)
        m = pd.concat([m, pd.Series(np.nan, index=m.index[gap.to_numpy()] - pd.Timedelta(days=1))]).sort_index()
        ax.plot(m.index, m.to_numpy(), color=COLORS[c], label=MODEL_LABELS[c], lw=1.5, marker="o", ms=3)
    ax.set(ylabel="Log loss, monthly mean", title="Performance over time (each month's games)")
    _legend_below(ax)
    return _save(fig, path)


def score_comparison(report: dict, path):
    """Log loss and Brier as dots with 95% date-clustered CIs (no truncated bars); one axis per panel."""
    plt = _plt()
    order = ("p_home_base_rate", "p_elo", "p_logistic_raw", "p_logistic", "p_gbm_raw", "p_gbm", "p_ensemble")
    cols = [c for c in order if c in report["models"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(cols) + 1.4))
    for ax, metric in zip(axes, ("log_loss", "brier")):
        y = np.arange(len(cols))
        for yi, c in zip(y, cols):
            s = report["models"][c]
            ci = s.get(f"{metric}_ci", {})
            ax.plot([ci.get("ci_low", s[metric]), ci.get("ci_high", s[metric])], [yi, yi], color=COLORS[c], lw=2)
            ax.plot(s[metric], yi, "o", ms=8, color=COLORS[c], markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(f"{s[metric]:.4f}", (ci.get("ci_high", s[metric]), yi), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y, [MODEL_LABELS[c] for c in cols])
        ax.invert_yaxis()
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + (hi - lo) * 0.15)
        ax.set_title(f"{metric.replace('_', ' ').title()} (95% CI, lower is better)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    return _save(fig, path)


def confusion_plot(cm: dict, path, label: str):
    plt = _plt()
    m = np.array([[cm["tp_home_win"], cm["fp_home_win"]], [cm["fn_away_win"], cm["tn_away_win"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(m, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{v}", ha="center", va="center", color=INK if v < m.max() * 0.6 else SURFACE, fontsize=12)
    ax.set_xticks([0, 1], ["Home won", "Away won"])
    ax.set_yticks([0, 1], ["Picked home", "Picked away"])
    ax.set_title(f"Confusion matrix at 0.5: {label}")
    return _save(fig, path)


def calibration_drift(by_season_cal: pd.DataFrame, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for c in MAIN:
        col = f"{c}_ece"
        if col in by_season_cal:
            ax.plot(by_season_cal["season"], by_season_cal[col], marker="o", ms=5, color=COLORS[c],
                    label=MODEL_LABELS[c])
    ax.set(xlabel="Season", ylabel="Expected calibration error", title="Calibration drift by season")
    _legend_below(ax)
    return _save(fig, path)


def feature_importance(importance: dict, path, title):
    plt = _plt()
    items = sorted(importance.items(), key=lambda kv: abs(kv[1]))
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(items) + 1))
    ax.barh([k for k, _ in items], [v for _, v in items], color=COLORS["p_logistic"] if "coef" in title.lower()
            else COLORS["p_gbm"], height=0.6)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(title)
    return _save(fig, path)


def bankroll_curve(bets: pd.DataFrame, path, initial: float):
    """Equity and drawdown: two stacked panels, each with its own single axis."""
    plt = _plt()
    b = bets.sort_values(["date", "game_key"])
    flat = b["flat_pnl"].cumsum().to_numpy()
    kelly = initial + b["kelly_pnl"].cumsum().to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    axes[0].plot(b["date"], flat, color=COLORS["p_elo"], label="Flat 1-unit stakes (units)")
    axes[0].axhline(0, color=INK2, lw=0.8)
    axes[0].set(title="SIMULATED paper-trading equity (not real bets)", ylabel="Cumulative profit, units")
    axes[0].legend(loc="best")
    peak = np.maximum.accumulate(kelly)
    axes[1].fill_between(b["date"], -(peak - kelly) / peak * 100, 0, color=COLORS["p_logistic"], alpha=0.6,
                         label="Fractional Kelly drawdown")
    axes[1].set(ylabel="Drawdown, %", title="Fractional Kelly drawdown from peak")
    return _save(fig, path)
