"""Append-only archive of sportsbook moneyline snapshots (SQLite tables in store.PHASE3_SCHEMA).

Every observation is kept: a later price never overwrites an earlier one. Three times are stored:
  observed_at   when the price was known to be on offer (our retrieval time for current odds;
                the provider's snapshot time for historical odds)
  provider_ts   the provider's own "last updated" time for that market, if given
  retrieved_at  when our collector received the data

Provider events are matched to NBA `game_key`s by team pair and scheduled start; prices are
stored oriented to the NBA's home team (`flipped` = 1 when the provider listed the sides the other
way round). Snapshots observed at or after tip-off are kept but marked is_pregame = 0, and every
reader here filters them out: an in-game price is never a decision or closing price.
"""
from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd

from .odds import american_to_decimal, decimal_to_american

log = logging.getLogger(__name__)

MATCH_WINDOW = pd.Timedelta(hours=18)      # provider start vs NBA tip-off, for the same game
MAX_OVERROUND = 0.25                       # anything above is a data error, not a real NBA moneyline
MIN_OVERROUND = -0.02                      # small negative = stale/arbitrage quote; below = bad data
FUTURE_SKEW = pd.Timedelta(minutes=5)      # provider_ts later than observed_at by more is invalid


def _utc(s) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


# ------------------------------------------------------------------ validation
def validate_snapshots(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (valid, invalid). Invalid rows carry an `invalid_reason`.

    Decimal prices are derived from American ones where only those are present."""
    df = df.copy()
    for side in ("home", "away"):
        if f"{side}_decimal" not in df:
            df[f"{side}_decimal"] = np.nan
        if f"{side}_american" not in df:
            df[f"{side}_american"] = np.nan
        am = pd.to_numeric(df[f"{side}_american"], errors="coerce")
        dec = pd.to_numeric(df[f"{side}_decimal"], errors="coerce")
        ok_am = am.notna() & np.isfinite(am) & (am.abs() >= 100)
        fill = dec.isna() & ok_am
        dec[fill] = american_to_decimal(am[fill]) if fill.any() else dec[fill]
        df[f"{side}_decimal"], df[f"{side}_american"] = dec, am
    reason = pd.Series("", index=df.index, dtype=object)

    def flag(mask, why):
        reason[mask & (reason == "")] = why

    for side in ("home", "away"):
        am = df[f"{side}_american"]
        flag(am.notna() & (am.abs() < 100), f"{side}_american_out_of_range")
        d = df[f"{side}_decimal"]
        flag(d.isna() | ~np.isfinite(d.astype(float)), f"missing_{side}_price")
        flag(d <= 1.0, f"{side}_decimal_not_above_1")
    obs = _utc(df["observed_at"])
    flag(obs.isna(), "bad_observed_at")
    flag(df["home_team"].isna() | df["away_team"].isna(), "unknown_team")
    flag(df["home_team"] == df["away_team"], "same_team_both_sides")
    ok = reason == ""
    over = pd.Series(np.nan, index=df.index)
    over[ok] = 1 / df.loc[ok, "home_decimal"] + 1 / df.loc[ok, "away_decimal"] - 1
    flag(ok & ((over > MAX_OVERROUND) | (over < MIN_OVERROUND)), "implausible_overround")
    if "provider_ts" in df:
        pts = _utc(df["provider_ts"])
        flag(pts.notna() & obs.notna() & (pts > obs + FUTURE_SKEW), "provider_ts_after_observation")
    df["invalid_reason"] = reason
    return df[reason == ""].drop(columns="invalid_reason"), df[reason != ""]


# ------------------------------------------------------------------ matching
def _game_tips(games: pd.DataFrame) -> pd.Series:
    """NBA tip-off; when unknown, 19:00 US/Eastern on the game date (typical tip)."""
    tip = _utc(games["tipoff_utc"]) if "tipoff_utc" in games else pd.Series(pd.NaT, index=games.index)
    nominal = (pd.to_datetime(games["date"]).dt.tz_localize(None) + pd.Timedelta(hours=19)
               ).dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
    return tip.fillna(nominal)


def map_to_games(snap: pd.DataFrame, games: pd.DataFrame, window: pd.Timedelta = MATCH_WINDOW) -> pd.DataFrame:
    """Attach game_key / tipoff_utc / flipped / match_status to provider rows.

    A provider event matches an NBA game with the same two teams (either home/away order) whose
    tip-off is within `window` of the provider's start time. Two candidates -> "ambiguous";
    none -> "unmatched". Matched rows are re-oriented to the NBA home team."""
    out = snap.copy()
    out["game_key"], out["tipoff_utc"], out["flipped"], out["match_status"] = None, None, 0, "unmatched"
    if games is None or games.empty or out.empty:
        return out
    g = games[["game_key", "home_team", "away_team"]].copy()
    g["tip"] = _game_tips(games).to_numpy()
    by_pair = {}
    for r in g.itertuples(index=False):
        by_pair.setdefault(frozenset((r.home_team, r.away_team)), []).append(r)
    start = _utc(out["commence_time"])
    for i in out.index:
        cands = [c for c in by_pair.get(frozenset((out.at[i, "home_team"], out.at[i, "away_team"])), [])
                 if pd.notna(start[i]) and abs(c.tip - start[i]) <= window]
        if len(cands) > 1:
            out.at[i, "match_status"] = "ambiguous"
            continue
        if not cands:
            continue
        c = cands[0]
        out.at[i, "game_key"], out.at[i, "match_status"] = c.game_key, "matched"
        out.at[i, "tipoff_utc"] = c.tip.isoformat()
        if c.home_team != out.at[i, "home_team"]:
            out.at[i, "flipped"] = 1
            for a, b in (("home_team", "away_team"), ("home_decimal", "away_decimal"),
                         ("home_american", "away_american")):
                out.at[i, a], out.at[i, b] = out.at[i, b], out.at[i, a]
    return out


def pregame_mask(df: pd.DataFrame) -> pd.Series:
    """observed strictly before the earlier of the NBA tip-off and the provider's start time."""
    obs = _utc(df["observed_at"])
    tip = pd.concat([_utc(df.get("tipoff_utc")), _utc(df.get("commence_time"))], axis=1).min(axis=1)
    return obs.notna() & tip.notna() & (obs < tip)


# ------------------------------------------------------------------ persistence
INSERT_COLS = ["provider", "source", "provider_event_id", "game_key", "match_status", "flipped", "home_team",
               "away_team", "provider_home_name", "provider_away_name", "book", "market", "home_american",
               "away_american", "home_decimal", "away_decimal", "commence_time", "tipoff_utc", "observed_at",
               "provider_ts", "retrieved_at", "is_pregame", "run_id"]


def from_provider_frame(rows: pd.DataFrame, provider: str, source: str, retrieved_at: str) -> pd.DataFrame:
    """odds_api.parse_events output -> snapshot-table columns (before validation and matching)."""
    out = pd.DataFrame({
        "provider": provider, "source": source, "provider_event_id": rows["event_id"].astype(str),
        "home_team": rows["home_team"], "away_team": rows["away_team"],
        "provider_home_name": rows.get("home_name"), "provider_away_name": rows.get("away_name"),
        "book": rows["book"], "market": "h2h",
        "home_american": rows.get("home_american", np.nan), "away_american": rows.get("away_american", np.nan),
        "home_decimal": rows["home_price"], "away_decimal": rows["away_price"],
        "commence_time": rows["commence_time"], "observed_at": _utc(rows["observed_at"]).map(
            lambda t: t.isoformat() if pd.notna(t) else None),
        "provider_ts": rows.get("market_last_update"), "retrieved_at": retrieved_at})
    return out


def store_snapshots(con: sqlite3.Connection, rows: pd.DataFrame, games: pd.DataFrame | None,
                    run_id: int | None = None) -> dict:
    """Validate, match, de-duplicate and insert. Returns counts; nothing is ever overwritten."""
    counts = {"n_rows": int(len(rows)), "n_invalid": 0, "n_inserted": 0, "n_duplicates": 0, "n_unmatched": 0,
              "n_live": 0, "invalid_reasons": {}}
    if rows.empty:
        return counts
    valid, invalid = validate_snapshots(rows)
    counts["n_invalid"] = int(len(invalid))
    counts["invalid_reasons"] = invalid["invalid_reason"].value_counts().to_dict() if len(invalid) else {}
    valid = map_to_games(valid, games)
    valid["is_pregame"] = pregame_mask(valid).astype(int)
    valid["run_id"] = run_id
    for side in ("home", "away"):  # keep the quoted American price; derive it only if absent
        miss = valid[f"{side}_american"].isna()
        if miss.any():
            valid.loc[miss, f"{side}_american"] = np.round(decimal_to_american(valid.loc[miss, f"{side}_decimal"]))
    counts["n_unmatched"] = int((valid["match_status"] != "matched").sum())
    counts["n_live"] = int((valid["is_pregame"] == 0).sum())
    valid = valid.drop_duplicates(["provider", "provider_event_id", "book", "market", "observed_at"])
    before = con.total_changes
    con.executemany(f"INSERT OR IGNORE INTO odds_snapshots ({','.join(INSERT_COLS)}) "
                    f"VALUES ({','.join('?' * len(INSERT_COLS))})",
                    [tuple(_py(r[c]) for c in INSERT_COLS) for r in valid.to_dict("records")])
    con.commit()
    counts["n_inserted"] = con.total_changes - before
    counts["n_duplicates"] = int(len(rows) - counts["n_invalid"] - counts["n_inserted"])
    return counts


def _py(v):
    if v is None or (np.ndim(v) == 0 and pd.isna(v)):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def remap_unmatched(con: sqlite3.Connection, games: pd.DataFrame) -> int:
    """Match snapshots stored before their game appeared in the schedule. Only the match columns
    (and orientation) of unmatched rows change; observed prices are not touched."""
    snap = pd.read_sql("SELECT * FROM odds_snapshots WHERE match_status != 'matched'", con)
    if snap.empty:
        return 0
    m = map_to_games(snap.drop(columns=["game_key", "tipoff_utc", "flipped", "match_status"]), games)
    m["is_pregame"] = pregame_mask(m).astype(int)
    hit = m[m["match_status"] == "matched"]
    con.executemany("""UPDATE odds_snapshots SET game_key=?, tipoff_utc=?, flipped=?, match_status='matched',
                       home_team=?, away_team=?, home_decimal=?, away_decimal=?, home_american=?, away_american=?,
                       is_pregame=? WHERE id=? AND match_status != 'matched'""",
                    [(r.game_key, r.tipoff_utc, int(r.flipped), r.home_team, r.away_team, float(r.home_decimal),
                      float(r.away_decimal), _py(r.home_american), _py(r.away_american), int(r.is_pregame), int(r.id))
                     for r in hit.itertuples()])
    con.commit()
    return int(len(hit))


def start_run(con: sqlite3.Connection, provider: str, mode: str, targets: list | None = None,
              now: pd.Timestamp | None = None) -> int:
    import json
    cur = con.execute("INSERT INTO odds_collection_runs (started_at, provider, mode, status, targets_json) "
                      "VALUES (?,?,?,?,?)", ((now or pd.Timestamp.now(tz="UTC")).isoformat(), provider, mode,
                                             "running", json.dumps(targets or [], default=str)))
    con.commit()
    return int(cur.lastrowid)


def finish_run(con: sqlite3.Connection, run_id: int, status: str, counts: dict | None = None, usage: dict | None = None,
               error: str | None = None, reason: str | None = None, n_events: int | None = None) -> None:
    c, u = counts or {}, usage or {}

    def num(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    con.execute("""UPDATE odds_collection_runs SET finished_at=?, status=?, reason=?, n_events=?, n_rows=?,
                   n_inserted=?, n_duplicates=?, n_invalid=?, n_unmatched=?, n_live=?, credits_used=?,
                   credits_remaining=?, error=? WHERE id=?""",
                (pd.Timestamp.now(tz="UTC").isoformat(), status, reason, n_events, c.get("n_rows"),
                 c.get("n_inserted"), c.get("n_duplicates"), c.get("n_invalid"), c.get("n_unmatched"),
                 c.get("n_live"), num(u.get("x-requests-last")), num(u.get("x-requests-remaining")), error, run_id))
    con.commit()


# ------------------------------------------------------------------ reading
def load_snapshots(con: sqlite3.Connection, pregame_only: bool = True, book: str | None = None,
                   matched_only: bool = True) -> pd.DataFrame:
    q = "SELECT * FROM odds_snapshots WHERE 1=1"
    params = []
    if pregame_only:
        q += " AND is_pregame = 1"
    if matched_only:
        q += " AND match_status = 'matched'"
    if book and book != "best":
        q += " AND book = ?"
        params.append(book)
    df = pd.read_sql(q, con, params=params)
    for c in ("observed_at", "commence_time", "tipoff_utc", "retrieved_at"):
        df[c] = _utc(df[c])
    if pregame_only and not df.empty:
        df = df[pregame_mask(df)]  # belt and braces: re-check against the stored times
    return df.sort_values(["game_key", "book", "observed_at"]).reset_index(drop=True)


def as_join_frame(snap: pd.DataFrame) -> pd.DataFrame:
    """Snapshots in the CSV layout odds_join.attach_odds expects (already NBA-oriented)."""
    return pd.DataFrame({"observed_at": snap["observed_at"], "event_id": snap["provider_event_id"],
                         "commence_time": snap["tipoff_utc"].fillna(snap["commence_time"]),
                         "home_team": snap["home_team"], "away_team": snap["away_team"], "book": snap["book"],
                         "home_price": snap["home_decimal"], "away_price": snap["away_decimal"]})


def quotes_asof(snap: pd.DataFrame, asof: pd.Series, max_age: pd.Timedelta | None = None,
                book: str | None = None) -> pd.DataFrame:
    """Latest pregame quote per (game, book) observed at or before each game's `asof` time.

    asof: Series indexed by game_key (UTC timestamps). Quotes older than `max_age` before `asof`
    are treated as stale and dropped. Returns one row per (game_key, book)."""
    if snap.empty or asof.empty:
        return snap.iloc[0:0].assign(asof=pd.Series(dtype="datetime64[ns, UTC]"))
    s = snap[snap["game_key"].isin(asof.index)]
    if book and book != "best":
        s = s[s["book"] == book]
    s = s.assign(asof=_utc(s["game_key"].map(asof)))
    s = s[s["observed_at"] <= s["asof"]]
    if max_age is not None:
        s = s[s["observed_at"] >= s["asof"] - max_age]
    s = s.sort_values("observed_at").groupby(["game_key", "book"], as_index=False).tail(1)
    return s.rename(columns={"id": "snapshot_id"}).reset_index(drop=True)


def closing_quotes(snap: pd.DataFrame, game_keys=None, book: str | None = None) -> pd.DataFrame:
    """Final pregame quote per (game, book): the closing price for CLV."""
    s = snap if game_keys is None else snap[snap["game_key"].isin(list(game_keys))]
    if book and book != "best":
        s = s[s["book"] == book]
    s = s.sort_values("observed_at").groupby(["game_key", "book"], as_index=False).tail(1)
    return s.rename(columns={"id": "snapshot_id"}).reset_index(drop=True)


def best_price(quotes: pd.DataFrame) -> pd.DataFrame:
    """Best decimal price per side across books for each game (labelled book="best"; optimistic:
    it assumes an account at every book and that both quotes were simultaneously available)."""
    if quotes.empty:
        return quotes
    agg = quotes.groupby("game_key").agg(home_decimal=("home_decimal", "max"), away_decimal=("away_decimal", "max"),
                                          observed_at=("observed_at", "max"))
    return agg.reset_index().assign(book="best")
