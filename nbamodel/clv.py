"""Price-based closing-line value (CLV) for paper decisions.

    price CLV = decimal odds at decision / decimal closing odds - 1

"Closing odds" are the final pregame snapshot of the SAME sportsbook and the same side, observed
after the decision price and within `close_window_minutes` of tip-off. Post-tip-off prices are
never used (odds_store only returns pregame snapshots). When the decision used the best price
across books, the close is the best price across books at their final pregame snapshots, and the
comparison is labelled "best_vs_best_close". No other cross-book substitution is made.

Also stored: the no-vig probability of the chosen side at decision and at close, and its move.

Positive CLV says the decision beat the later market price. It is evidence about price
timing, not proof of profit.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import odds_store
from .odds import remove_vig_two_way

STATUSES = ("ok", "no_decision_price", "no_close_snapshot", "no_later_snapshot", "close_too_early")


def price_clv(decision_decimal, close_decimal):
    d, c = np.asarray(decision_decimal, float), np.asarray(close_decimal, float)
    if np.any(d <= 1) or np.any(c <= 1):
        raise ValueError("decimal odds must be greater than 1.0")
    return d / c - 1.0


def _close_for(snap: pd.DataFrame, game_key: str, book: str) -> pd.Series | None:
    s = snap[snap["game_key"] == game_key]
    if s.empty:
        return None
    if book == "best":
        last = odds_store.closing_quotes(s)
        if last.empty:
            return None
        return pd.Series({"home_decimal": last["home_decimal"].max(), "away_decimal": last["away_decimal"].max(),
                          "observed_at": last["observed_at"].min(), "book": "best"})
    s = s[s["book"] == book]
    if s.empty:
        return None
    return s.sort_values("observed_at").iloc[-1]


def compute(decisions: pd.DataFrame, snap: pd.DataFrame, close_window_minutes: float = 30.0) -> pd.DataFrame:
    """One CLV row per decision (status explains any missing value)."""
    rows = []
    for d in decisions.to_dict("records"):
        r = {"decision_id": d["id"], "book": d.get("book"), "status": "ok",
             "comparison": "best_vs_best_close" if d.get("book") == "best" else "same_book",
             "decision_decimal": d.get("decimal_odds"), "close_decimal": None, "close_observed_at": None,
             "price_clv": None, "open_novig": d.get("market_p_side_novig"), "close_novig": None,
             "novig_prob_move": None}
        if not d.get("side") or pd.isna(d.get("decimal_odds")):
            rows.append({**r, "status": "no_decision_price"})
            continue
        close = _close_for(snap, d["game_key"], d["book"])
        if close is None:
            rows.append({**r, "status": "no_close_snapshot"})
            continue
        obs = pd.Timestamp(close["observed_at"])
        if obs <= pd.Timestamp(d["odds_observed_at"]):
            rows.append({**r, "status": "no_later_snapshot"})
            continue
        tip = pd.Timestamp(d["tipoff_utc"])
        if (tip - obs) > pd.Timedelta(minutes=close_window_minutes):
            rows.append({**r, "status": "close_too_early", "close_observed_at": obs.isoformat()})
            continue
        home = d["side"] == "home"
        side_c = close["home_decimal"] if home else close["away_decimal"]
        nv_h = float(remove_vig_two_way(close["home_decimal"], close["away_decimal"])[0])
        close_nv = nv_h if home else 1 - nv_h
        r.update(close_decimal=float(side_c), close_observed_at=obs.isoformat(),
                 price_clv=float(price_clv(d["decimal_odds"], side_c)), close_novig=close_nv,
                 novig_prob_move=(close_nv - d["market_p_side_novig"]) if pd.notna(d.get("market_p_side_novig"))
                 else None)
        rows.append(r)
    return pd.DataFrame(rows)


def record(con: sqlite3.Connection, close_window_minutes: float = 30.0, now: pd.Timestamp | None = None) -> int:
    """Compute CLV for decisions whose games have tipped off and have no CLV row yet."""
    now = now or pd.Timestamp.now(tz="UTC")
    dec = pd.read_sql("""SELECT d.* FROM paper_decisions d LEFT JOIN clv_results c ON c.decision_id = d.id
                         WHERE c.decision_id IS NULL""", con)
    if dec.empty:
        return 0
    dec = dec[pd.to_datetime(dec["tipoff_utc"], utc=True) <= now]
    if dec.empty:
        return 0
    rows = compute(dec, odds_store.load_snapshots(con), close_window_minutes)
    rows["computed_at"] = now.isoformat()
    cols = ["decision_id", "status", "book", "comparison", "decision_decimal", "close_decimal", "close_observed_at",
            "price_clv", "open_novig", "close_novig", "novig_prob_move", "computed_at"]
    before = con.total_changes
    con.executemany(f"INSERT OR IGNORE INTO clv_results ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [tuple(odds_store._py(r[c]) for c in cols) for r in rows.to_dict("records")])
    con.commit()
    return con.total_changes - before


def _stats(x: pd.Series) -> dict:
    x = x.dropna()
    if x.empty:
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()),
            "share_positive": float((x > 0).mean())}


def summary(con: sqlite3.Connection, strategy: str | None = None) -> dict:
    df = pd.read_sql("""SELECT c.*, d.decision, d.model_version, d.strategy, d.game_date, d.side
                        FROM clv_results c JOIN paper_decisions d ON d.id = c.decision_id""", con)
    if strategy:
        df = df[df["strategy"] == strategy]
    if df.empty:
        return {"n_decisions": 0, "label": "price-based CLV"}
    ok = df[df["status"] == "ok"]
    ok = ok.assign(season=pd.to_datetime(ok["game_date"]).map(lambda d: d.year if d.month >= 7 else d.year - 1))
    bets = ok[ok["decision"] == "bet"]
    return {"label": "price-based CLV = decision decimal / closing decimal (same book) - 1",
            "n_decisions": int(len(df)), "status_counts": df["status"].value_counts().to_dict(),
            "all_priced_decisions": _stats(ok["price_clv"]), "bets": _stats(bets["price_clv"]),
            "bets_novig_prob_move": _stats(bets["novig_prob_move"]),
            "by_model": {k: _stats(g["price_clv"]) for k, g in bets.groupby("model_version")},
            "by_book": {k: _stats(g["price_clv"]) for k, g in bets.groupby("book")},
            "by_season": {int(k): _stats(g["price_clv"]) for k, g in bets.groupby("season")},
            "caveat": "Positive CLV alone is not proof of profitability."}
