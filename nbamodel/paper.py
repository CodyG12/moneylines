"""Live paper-trading ledger. PAPER ONLY: nothing here can place a real bet.

For each upcoming game, one decision per strategy is written once, at the first logged
prediction made inside the decision window (default 30-90 minutes before tip-off), using only
the sportsbook price recorded with that prediction (observed at or before it). Decisions are:

  bet                 estimated EV above the threshold, within stake and exposure limits
  below_threshold     priced, but estimated EV too small
  no_odds             no price for the book at prediction time
  stale_odds          a price existed but was older than max_odds_age_minutes
  odds_after_decision guard: a price observed after the decision (should never happen)
  low_history         abstained: early-season game and skip_low_history is on
  model_disagreement  abstained: Elo/logistic/boosting spread above max_model_spread
  edge_too_large      abstained: model and no-vig market differ by more than max_abs_edge; at that size
                      the model missing information (injuries, roster news) is likelier than value
  daily_exposure_cap  abstained: the day's simulated stakes already reach the cap

At most one side per game (no correlated/overlapping selections on the same game). Kelly
stakes use the bankroll from results resolved before the decision. Stakes are capped per bet
(max_bet_fraction, max_stake_units as a stand-in for book limits) and per date
(max_daily_exposure). Decisions and results are append-only; a decision is never revised.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import odds_store
from .evaluation import cluster_bootstrap_ratio
from .odds import decimal_to_american, expected_value, kelly_fraction, remove_vig_two_way

DECISIONS = ("bet", "below_threshold", "no_odds", "stale_odds", "odds_after_decision", "low_history",
             "model_disagreement", "edge_too_large", "daily_exposure_cap")


@dataclass(frozen=True)
class PaperConfig:
    strategy: str = "live-default"
    book: str = "draftkings"
    ev_threshold: float = 0.02
    decision_min_lead_minutes: float = 30.0
    decision_max_lead_minutes: float = 90.0
    max_odds_age_minutes: float = 120.0
    flat_stake: float = 1.0
    kelly_multiplier: float = 0.25
    max_bet_fraction: float = 0.02
    max_stake_units: float = 5.0
    max_daily_exposure: float = 0.10
    initial_bankroll: float = 100.0
    skip_low_history: bool = False
    max_model_spread: float | None = 0.20
    max_abs_edge: float | None = 0.10
    close_window_minutes: float = 30.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "PaperConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in allowed})


def bankroll_at(con: sqlite3.Connection, strategy: str, initial: float, at: pd.Timestamp) -> float:
    r = con.execute("""SELECT COALESCE(SUM(r.pnl_kelly), 0) FROM paper_results r
                       JOIN paper_decisions d ON d.id = r.decision_id
                       WHERE d.strategy = ? AND r.resolved_at <= ?""", (strategy, at.isoformat())).fetchone()
    return float(initial + r[0])


def _candidates(con: sqlite3.Connection, cfg: PaperConfig, now: pd.Timestamp, model_version: str | None):
    p = pd.read_sql("""SELECT p.* FROM predictions p
                       LEFT JOIN paper_decisions d ON d.game_key = p.game_key AND d.strategy = ?
                       WHERE d.id IS NULL AND p.tipoff_utc IS NOT NULL""", con, params=[cfg.strategy])
    if model_version:
        p = p[p["model_version"] == model_version]
    if p.empty:
        return p
    ts, tip = pd.to_datetime(p["prediction_ts"], utc=True), pd.to_datetime(p["tipoff_utc"], utc=True)
    lead = (tip - ts).dt.total_seconds() / 60
    # decisions are only made before tip-off: never on a game that has started or finished
    p = p[(lead >= cfg.decision_min_lead_minutes) & (lead <= cfg.decision_max_lead_minutes) & (ts <= now)
          & (tip > now)]
    # the first qualifying prediction is the decision point (later ones cannot revise it)
    return p.sort_values("prediction_ts").groupby("game_key", as_index=False).head(1)


def _quote(con: sqlite3.Connection, pred_id: int, book: str):
    q = pd.read_sql("SELECT * FROM prediction_odds WHERE prediction_id = ?", con, params=[int(pred_id)])
    if q.empty:
        return None
    if book == "best":
        return pd.Series({"book": "best", "home_decimal": q["home_decimal"].max(), "away_decimal": q["away_decimal"].max(),
                          "observed_at": q["observed_at"].max(), "home_american": np.nan, "away_american": np.nan})
    q = q[q["book"] == book]
    return None if q.empty else q.iloc[0]


def decide(con: sqlite3.Connection, cfg: PaperConfig, now: pd.Timestamp | None = None,
           model_version: str | None = None) -> pd.DataFrame:
    """Write decisions for games whose decision prediction is available. Returns the new rows."""
    now = now or pd.Timestamp.now(tz="UTC")
    cand = _candidates(con, cfg, now, model_version)
    if cand.empty:
        return pd.DataFrame()
    snap = None
    new = []
    for p in cand.sort_values(["tipoff_utc", "game_key"]).to_dict("records"):
        decided_at = pd.Timestamp(p["prediction_ts"])
        row = {"strategy": cfg.strategy, "prediction_id": int(p["id"]), "game_key": p["game_key"],
               "game_date": p["game_date"], "tipoff_utc": p["tipoff_utc"], "model_version": p["model_version"],
               "p_col": "p_home", "decided_at": decided_at.isoformat(), "book": cfg.book, "side": None,
               "p_side": None, "decimal_odds": None, "american_odds": None, "odds_observed_at": None,
               "market_p_side_novig": None, "edge": None, "est_ev": None, "stake_flat": 0.0, "stake_kelly": 0.0,
               "bankroll_at_decision": None}
        details = {"config": asdict(cfg), "model_spread": p.get("model_spread"), "low_history": p.get("low_history")}
        q = _quote(con, p["id"], cfg.book)
        if q is None:
            if snap is None:
                snap = odds_store.load_snapshots(con, book=None if cfg.book == "best" else cfg.book)
            older = snap[(snap["game_key"] == p["game_key"]) & (snap["observed_at"] <= decided_at)] if len(snap) else snap
            row["decision"] = "stale_odds" if len(older) else "no_odds"
            new.append({**row, "details_json": json.dumps(details, default=str)})
            continue
        obs = pd.Timestamp(q["observed_at"])
        h, a, ph = float(q["home_decimal"]), float(q["away_decimal"]), float(p["p_home"])
        ev_h, ev_a = float(expected_value(ph, h)), float(expected_value(1 - ph, a))
        home = ev_h >= ev_a
        nv_h = float(remove_vig_two_way(h, a)[0])
        dec = h if home else a
        am = q["home_american"] if home else q["away_american"]
        row.update(side="home" if home else "away", p_side=ph if home else 1 - ph, decimal_odds=dec,
                   american_odds=float(am) if pd.notna(am) else float(np.round(decimal_to_american(dec))),
                   odds_observed_at=obs.isoformat(), market_p_side_novig=nv_h if home else 1 - nv_h,
                   est_ev=ev_h if home else ev_a)
        row["edge"] = row["p_side"] - row["market_p_side_novig"]
        if obs > decided_at:
            row["decision"] = "odds_after_decision"
        elif row["est_ev"] <= cfg.ev_threshold:
            row["decision"] = "below_threshold"
        elif cfg.skip_low_history and p.get("low_history"):
            row["decision"] = "low_history"
        elif cfg.max_model_spread is not None and pd.notna(p.get("model_spread")) and p["model_spread"] > cfg.max_model_spread:
            row["decision"] = "model_disagreement"
        elif cfg.max_abs_edge is not None and abs(row["edge"]) > cfg.max_abs_edge:
            row["decision"] = "edge_too_large"
        else:
            bank = bankroll_at(con, cfg.strategy, cfg.initial_bankroll, decided_at)
            used = con.execute("SELECT COALESCE(SUM(stake_kelly), 0) FROM paper_decisions WHERE strategy=? AND "
                               "game_date=? AND decision='bet'", (cfg.strategy, p["game_date"])).fetchone()[0]
            used += sum(n["stake_kelly"] for n in new if n["game_date"] == p["game_date"] and n["decision"] == "bet")
            room = cfg.max_daily_exposure * bank - used
            f = min(float(kelly_fraction(row["p_side"], dec)) * cfg.kelly_multiplier, cfg.max_bet_fraction)
            stake_k = min(f * bank, cfg.max_stake_units, max(room, 0.0))
            row["bankroll_at_decision"] = bank
            if room <= 1e-9:
                row["decision"] = "daily_exposure_cap"
            else:
                row.update(decision="bet", stake_flat=min(cfg.flat_stake, cfg.max_stake_units), stake_kelly=stake_k)
        new.append({**row, "details_json": json.dumps(details, default=str)})
    df = pd.DataFrame(new)
    cols = [c for c in df.columns]
    con.executemany(f"INSERT OR IGNORE INTO paper_decisions ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [tuple(odds_store._py(r[c]) for c in cols) for r in df.to_dict("records")])
    con.commit()
    return df


def resolve(con: sqlite3.Connection, now: pd.Timestamp | None = None) -> int:
    """Settle decisions whose game has a recorded result. Non-bets get pnl 0 (kept for what-if)."""
    now = (now or pd.Timestamp.now(tz="UTC")).isoformat()
    pend = pd.read_sql("""SELECT d.id, d.decision, d.side, d.decimal_odds, d.stake_flat, d.stake_kelly, r.home_win
                          FROM paper_decisions d JOIN results r USING (game_key)
                          LEFT JOIN paper_results pr ON pr.decision_id = d.id WHERE pr.decision_id IS NULL""", con)
    rows = []
    for r in pend.itertuples():
        won = None if r.side is None else int((r.side == "home") == (r.home_win == 1))
        bet = r.decision == "bet"
        pf = (r.stake_flat * (r.decimal_odds - 1) if won else -r.stake_flat) if bet else 0.0
        pk = (r.stake_kelly * (r.decimal_odds - 1) if won else -r.stake_kelly) if bet else 0.0
        rows.append((int(r.id), int(r.home_win), won, float(pf), float(pk), now))
    con.executemany("INSERT OR IGNORE INTO paper_results VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    return len(rows)


def ledger(con: sqlite3.Connection, strategy: str | None = None) -> pd.DataFrame:
    q = """SELECT d.*, r.won, r.pnl_flat, r.pnl_kelly, r.resolved_at, c.price_clv, c.status AS clv_status
           FROM paper_decisions d LEFT JOIN paper_results r ON r.decision_id = d.id
           LEFT JOIN clv_results c ON c.decision_id = d.id"""
    df = pd.read_sql(q, con)
    if strategy:
        df = df[df["strategy"] == strategy]
    return df.sort_values(["tipoff_utc", "game_key"]).reset_index(drop=True)


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    return float((peaks - equity).max())


def summarize(con: sqlite3.Connection, cfg: PaperConfig, n_boot: int = 1000, seed: int = 0) -> dict:
    led = ledger(con, cfg.strategy)
    out = {"strategy": cfg.strategy, "book": cfg.book, "simulated": True,
           "decisions": led["decision"].value_counts().reindex(DECISIONS, fill_value=0).astype(int).to_dict()
           if len(led) else {d: 0 for d in DECISIONS},
           "caveat": "SIMULATED paper trading on recorded sportsbook prices. No real bets. Small samples are noise."}
    bets = led[led["decision"] == "bet"]
    settled = bets[bets["won"].notna()]
    out.update({"n_bets": int(len(bets)), "n_settled": int(len(settled)), "n_open": int(len(bets) - len(settled)),
                "mean_est_ev": float(bets["est_ev"].mean()) if len(bets) else None,
                "mean_edge_vs_novig": float(bets["edge"].mean()) if len(bets) else None,
                "mean_price_clv": float(bets["price_clv"].mean()) if bets["price_clv"].notna().any() else None})
    if settled.empty:
        out["status"] = "no_settled_bets"
        out["bankroll_history"] = []
        return out
    pf, sf = settled["pnl_flat"].to_numpy(float), settled["stake_flat"].to_numpy(float)
    pk = settled["pnl_kelly"].to_numpy(float)
    eq_flat = np.concatenate([[0.0], np.cumsum(pf)])
    eq_k = cfg.initial_bankroll + np.concatenate([[0.0], np.cumsum(pk)])
    out.update({
        "status": "ok", "preliminary": len(settled) < 300,
        "win_rate": float(settled["won"].mean()), "mean_break_even": float((1 / settled["decimal_odds"]).mean()),
        "flat": {"staked": float(sf.sum()), "profit": float(pf.sum()), "roi": cluster_bootstrap_ratio(
            pf, sf, settled["game_date"], n_boot, seed), "max_drawdown_units": _max_drawdown(eq_flat)},
        "kelly": {"initial_bankroll": cfg.initial_bankroll, "bankroll": float(eq_k[-1]),
                  "staked": float(settled["stake_kelly"].sum()), "profit": float(pk.sum()),
                  "max_drawdown_fraction": float(((np.maximum.accumulate(eq_k) - eq_k) / np.maximum.accumulate(eq_k)).max())},
        "bankroll_history": [{"tipoff_utc": t, "game_key": g, "flat_cum": float(f), "kelly_bankroll": float(k)}
                             for t, g, f, k in zip(settled["tipoff_utc"], settled["game_key"], eq_flat[1:], eq_k[1:])],
    })
    return out
