"""Deterministic Phase 3 fixtures: two NBA games and hand-built odds snapshots with known values."""
import json

import numpy as np
import pandas as pd

from nbamodel import odds_store, store

# CHA (home) vs WAS tips 23:00Z; DEN (home) vs GSW tips 02:00Z next day (same US date).
GAMES = pd.DataFrame({
    "game_key": ["g_cha", "g_den"], "date": pd.to_datetime(["2026-10-26", "2026-10-26"]),
    "season": [2026, 2026], "home_team": ["CHA", "DEN"], "away_team": ["WAS", "GSW"],
    "tipoff_utc": ["2026-10-26T23:00:00Z", "2026-10-27T02:00:00Z"], "played": [False, False]})


def snap_row(obs, event="ev1", home="CHA", away="WAS", book="draftkings", h=-150.0, a=130.0,
             commence="2026-10-26T23:00:00Z", provider_ts=None):
    return {"provider": "test", "source": "test:current", "provider_event_id": event, "home_team": home,
            "away_team": away, "provider_home_name": home, "provider_away_name": away, "book": book, "market": "h2h",
            "home_american": h, "away_american": a, "home_decimal": np.nan, "away_decimal": np.nan,
            "commence_time": commence, "observed_at": obs, "provider_ts": provider_ts, "retrieved_at": obs}


def frame(*rows):
    return pd.DataFrame(list(rows))


def pred(game_key, ts, p, tip, version="m1", home="CHA", away="WAS", spread=0.05, low=False, date="2026-10-26"):
    return {"game_key": game_key, "date": pd.Timestamp(date), "season": 2026, "home_team": home, "away_team": away,
            "tipoff_utc": tip, "prediction_timestamp": ts, "model_version": version, "p_home": p, "p_away": 1 - p,
            "fair_home_american": -150, "fair_away_american": 150, "fair_home_decimal": 1 / p,
            "fair_away_decimal": 1 / (1 - p), "home_odds": np.nan, "away_odds": np.nan, "odds_timestamp": None,
            "odds_book": None, "market_home_prob_novig": np.nan, "model_spread": spread, "low_history": low,
            "features_asof": "2026-10-25", "feature_snapshot": json.dumps({}), "key_factors": "x",
            "p_elo": p - 0.01, "p_logistic_raw": p, "p_logistic": p, "p_gbm": p + 0.01, "p_gbm_raw": p + 0.01,
            "p_ensemble": p + 0.005}


def seeded_db(path, snaps):
    con = store.connect(path)
    odds_store.store_snapshots(con, snaps, GAMES)
    return con
