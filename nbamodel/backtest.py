"""Paper-trading simulation on holdout games with archived odds.

Paper trading only. Nothing here places real bets. Estimated EV is the
model's opinion; realised P&L is what actually happened in the sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import cluster_bootstrap_ratio
from .odds import closing_line_value, expected_value, kelly_fraction, overround, remove_vig_two_way


@dataclass(frozen=True)
class BettingConfig:
    ev_threshold: float = 0.02         # minimum estimated EV per unit to take a bet
    flat_stake: float = 1.0            # units per bet in the flat-stake scenario
    kelly_multiplier: float = 0.25     # fraction of full Kelly
    max_bet_fraction: float = 0.02     # cap per bet, as share of bankroll
    max_daily_exposure: float = 0.10   # cap on total stakes per date, share of bankroll
    initial_bankroll: float = 100.0


def select_bets(df: pd.DataFrame, p_col: str, threshold: float) -> pd.DataFrame:
    """Pick at most one side per game: the side with the higher estimated EV,
    if it clears `threshold`. Games without complete odds are abstentions."""
    priced = df[df["home_odds"].notna() & df["away_odds"].notna()].copy()
    if priced.empty:
        return priced.assign(side=[], odds=[], p_side=[], est_ev=[], won=[])
    p = priced[p_col].to_numpy()
    ev_home = expected_value(p, priced["home_odds"].to_numpy())
    ev_away = expected_value(1 - p, priced["away_odds"].to_numpy())
    home_better = ev_home >= ev_away
    priced["side"] = np.where(home_better, "home", "away")
    priced["odds"] = np.where(home_better, priced["home_odds"], priced["away_odds"])
    priced["p_side"] = np.where(home_better, p, 1 - p)
    priced["est_ev"] = np.where(home_better, ev_home, ev_away)
    priced["won"] = np.where(home_better, priced["home_win"], 1 - priced["home_win"])
    return priced[priced["est_ev"] > threshold].reset_index(drop=True)


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    return float((peaks - equity).max())


def simulate(bets: pd.DataFrame, config: BettingConfig = BettingConfig(), n_boot=1000, seed=0):
    """Run flat-stake and capped fractional-Kelly scenarios.

    Kelly stakes for all games on a date are sized from the bankroll at the
    start of that date, since those games are decided concurrently.
    """
    bets = bets.sort_values(["date", "game_id"]).reset_index(drop=True)
    won = bets["won"].to_numpy(dtype=float)
    odds = bets["odds"].to_numpy(dtype=float)

    flat_stake = np.full(len(bets), config.flat_stake)
    flat_pnl = np.where(won == 1, flat_stake * (odds - 1), -flat_stake)
    bets["flat_stake"] = flat_stake
    bets["flat_pnl"] = flat_pnl
    flat_equity = np.concatenate([[0.0], np.cumsum(flat_pnl)])

    kelly_stake = np.zeros(len(bets))
    kelly_pnl = np.zeros(len(bets))
    bankroll = config.initial_bankroll
    for _, idx in bets.groupby("date", sort=True).indices.items():
        f = kelly_fraction(bets.loc[idx, "p_side"], bets.loc[idx, "odds"]) * config.kelly_multiplier
        f = np.minimum(f, config.max_bet_fraction)
        if f.sum() > config.max_daily_exposure:
            f = f * config.max_daily_exposure / f.sum()
        stakes = f * bankroll
        pnl = np.where(won[idx] == 1, stakes * (odds[idx] - 1), -stakes)
        kelly_stake[idx] = stakes
        kelly_pnl[idx] = pnl
        bankroll += pnl.sum()
    bets["kelly_stake"] = kelly_stake
    bets["kelly_pnl"] = kelly_pnl
    kelly_equity = config.initial_bankroll + np.concatenate([[0.0], np.cumsum(kelly_pnl)])
    kelly_peaks = np.maximum.accumulate(kelly_equity)

    summary = {
        "n_bets": int(len(bets)),
        "n_bet_dates": int(bets["date"].nunique()) if len(bets) else 0,
        "mean_estimated_ev": float(bets["est_ev"].mean()) if len(bets) else None,
        "hit_rate": float(won.mean()) if len(bets) else None,
        "mean_break_even_rate": float((1 / odds).mean()) if len(bets) else None,
        "flat": {
            "total_staked_units": float(flat_stake.sum()),
            "realized_profit_units": float(flat_pnl.sum()),
            "max_drawdown_units": _max_drawdown(flat_equity),
            "pnl_std_per_bet": float(flat_pnl.std(ddof=1)) if len(bets) > 1 else None,
            "roi": (cluster_bootstrap_ratio(flat_pnl, flat_stake, bets["date"], n_boot, seed)
                    if len(bets) else None),
        },
        "fractional_kelly": {
            "initial_bankroll": config.initial_bankroll,
            "final_bankroll": float(kelly_equity[-1]),
            "max_drawdown_fraction": float(((kelly_peaks - kelly_equity) / kelly_peaks).max()),
            "settings": {"kelly_multiplier": config.kelly_multiplier,
                         "max_bet_fraction": config.max_bet_fraction,
                         "max_daily_exposure": config.max_daily_exposure},
        },
    }
    return bets, summary


# ------------------------------------------------------------------ Phase 2
DECISIONS = ("bet", "below_threshold", "no_odds", "no_odds_timestamp", "odds_after_decision")
SWEEP_THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)


def decision_table(df: pd.DataFrame, p_col: str, threshold: float, require_timestamps: bool = True) -> pd.DataFrame:
    """Classify every game: bet, or why not.

    A price is usable only if it was observed at or before the game's decision time
    (`prediction_timestamp`). Untimestamped prices are refused unless require_timestamps=False,
    because they may be closing or post-game lines.
    """
    out = df.copy()
    has_odds = out["home_odds"].notna() & out["away_odds"].notna() if "home_odds" in out else pd.Series(
        False, index=out.index)
    ts = (pd.to_datetime(out["odds_timestamp"], utc=True, errors="coerce") if "odds_timestamp" in out
          else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"))
    decision_at = pd.to_datetime(out["prediction_timestamp"], utc=True, errors="coerce")
    reason = pd.Series("bet", index=out.index, dtype=object)
    reason[~has_odds] = "no_odds"
    if require_timestamps:
        reason[has_odds & ts.isna()] = "no_odds_timestamp"
    reason[has_odds & ts.notna() & decision_at.notna() & (ts > decision_at)] = "odds_after_decision"
    usable = reason == "bet"
    out["decision"] = reason
    out["est_ev"] = np.nan
    if usable.any():
        chosen = select_bets(out[usable].reset_index().rename(columns={"index": "_row"}), p_col, -np.inf)
        idx = chosen["_row"].to_numpy()
        for col in ("side", "odds", "p_side", "est_ev", "won"):
            out.loc[idx, col] = chosen[col].to_numpy()
        out.loc[usable & ~(out["est_ev"] > threshold), "decision"] = "below_threshold"
    return out


def _bet_extras(bets: pd.DataFrame) -> pd.DataFrame:
    bets = bets.copy()
    home = bets["side"] == "home"
    nv_home = remove_vig_two_way(bets["home_odds"], bets["away_odds"])[0]
    bets["market_prob_side_novig"] = np.where(home, nv_home, 1 - nv_home)
    bets["edge"] = bets["p_side"] - bets["market_prob_side_novig"]
    if {"close_home_odds", "close_away_odds"} <= set(bets.columns):
        closed = bets["close_home_odds"].notna() & bets["close_away_odds"].notna()
        bets["clv"] = np.nan
        if closed.any():
            c = bets[closed]
            side_close = np.where(c["side"] == "home", c["close_home_odds"], c["close_away_odds"])
            other_close = np.where(c["side"] == "home", c["close_away_odds"], c["close_home_odds"])
            bets.loc[closed, "clv"] = closing_line_value(c["odds"], side_close, other_close)
    return bets


def paper_trade(preds: pd.DataFrame, p_col: str, config: BettingConfig = BettingConfig(), n_boot: int = 1000,
                seed: int = 0, require_timestamps: bool = True, thresholds=SWEEP_THRESHOLDS):
    """Simulated betting on walk-forward predictions with archived, timestamped prices.

    Returns (bets, summary). summary["status"] is "no_valid_odds" when nothing could be simulated;
    no synthetic prices are ever substituted.
    """
    decided = decision_table(preds, p_col, config.ev_threshold, require_timestamps)
    counts = decided["decision"].value_counts().reindex(DECISIONS, fill_value=0).astype(int).to_dict()
    summary = {"model": p_col, "decisions": counts, "ev_threshold": config.ev_threshold,
               "require_timestamps": require_timestamps,
               "caveat": "SIMULATED paper trading on archived prices. Estimated EV assumes the model is right; "
                         "realised profit in one historical sample is noisy, see the ROI CI. Not a claim of "
                         "future profitability."}
    priced = decided[decided["decision"].isin(["bet", "below_threshold"])]
    if priced.empty:
        summary["status"] = "no_valid_odds"
        return None, summary
    bets = _bet_extras(decided[decided["decision"] == "bet"].assign(game_id=lambda d: d["game_key"]))
    if bets.empty:
        summary["status"] = "no_bets_above_threshold"
        return bets, summary
    bets, sim = simulate(bets, config, n_boot, seed)
    summary.update(sim)
    summary["status"] = "ok"
    summary["mean_edge_vs_novig"] = float(bets["edge"].mean())
    summary["mean_overround"] = float(overround(bets["home_odds"], bets["away_odds"]).mean())
    if "clv" in bets and bets["clv"].notna().any():
        summary["clv"] = {"n": int(bets["clv"].notna().sum()), "mean": float(bets["clv"].mean()),
                          "share_positive": float((bets["clv"] > 0).mean())}
    summary["overfitting_check"] = threshold_sweep(priced.assign(game_id=lambda d: d["game_key"]), p_col,
                                                   thresholds, n_boot, seed)
    return bets, summary


def threshold_sweep(priced: pd.DataFrame, p_col: str, thresholds=SWEEP_THRESHOLDS, n_boot=1000, seed=0) -> dict:
    """Flat-stake ROI at several EV thresholds. Picking the best of k thresholds after the fact
    inflates its ROI, so the best one also gets a Bonferroni-widened (alpha / k) interval."""
    rows = []
    for t in thresholds:
        b = priced[priced["est_ev"] > t]
        if b.empty:
            rows.append({"threshold": t, "n_bets": 0})
            continue
        pnl = np.where(b["won"] == 1, b["odds"] - 1, -1.0)
        roi = cluster_bootstrap_ratio(pnl, np.ones(len(b)), b["date"], n_boot, seed)
        rows.append({"threshold": t, "n_bets": int(len(b)), "roi": roi["estimate"], "ci_low": roi["ci_low"],
                     "ci_high": roi["ci_high"]})
    valid = [r for r in rows if r["n_bets"]]
    out = {"by_threshold": rows, "n_thresholds_tried": len(thresholds),
           "n_positive_roi": int(sum(r["roi"] > 0 for r in valid))}
    if valid:
        best = max(valid, key=lambda r: r["roi"])
        b = priced[priced["est_ev"] > best["threshold"]]
        pnl = np.where(b["won"] == 1, b["odds"] - 1, -1.0)
        adj = cluster_bootstrap_ratio(pnl, np.ones(len(b)), b["date"], n_boot, seed,
                                      alpha=0.05 / len(thresholds))
        out["best_threshold"] = best["threshold"]
        out["best_roi_bonferroni_ci"] = [adj["ci_low"], adj["ci_high"]]
        out["interpretation"] = ("best-threshold ROI interval excludes 0" if adj["ci_low"] > 0
                                 else "no threshold shows ROI distinguishable from 0 after multiple-testing "
                                      "adjustment")
    return out
