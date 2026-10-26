"""Paper-trading simulation on holdout games with archived odds.

Paper trading only. Nothing here places real bets. Estimated EV is the
model's opinion; realised P&L is what actually happened in the sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import cluster_bootstrap_ratio
from .odds import expected_value, kelly_fraction


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
