"""One odds-collection pass: decide whether a call is due, fetch, store, log the run.

Providers are small objects with `name` and `fetch_current() -> (rows, usage)` where rows is
odds_api.parse_events output. TheOddsApi is the live one; FileProvider replays saved JSON
(tests, offline validation). Any failure is recorded in odds_collection_runs and never leaves
partial or invented prices behind.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from . import odds_api, odds_schedule, odds_store

log = logging.getLogger(__name__)


class TheOddsApi:
    name = "the-odds-api"

    def __init__(self, api_key: str, regions: str = "us", bookmakers: str | None = None, retries: int = 4):
        self.api_key, self.regions, self.bookmakers, self.retries = api_key, regions, bookmakers, retries

    def fetch_current(self):
        return odds_api.fetch_current(self.api_key, self.regions, self.bookmakers, odds_format="american",
                                      retries=self.retries)


class FileProvider:
    """Replays a saved provider response; `observed_at` must be given (it is not 'now')."""
    name = "file"

    def __init__(self, path, observed_at: str, odds_format: str = "decimal"):
        self.path, self.observed_at, self.odds_format = Path(path), observed_at, odds_format

    def fetch_current(self):
        events = json.loads(self.path.read_text())
        return odds_api.parse_events(events, self.observed_at, self.odds_format), {}


def call_times(con: sqlite3.Connection, provider: str) -> list:
    """Retrieval times of successful current-odds calls (what satisfies schedule targets)."""
    rows = con.execute("SELECT started_at FROM odds_collection_runs WHERE provider=? AND mode='current' "
                       "AND status='ok'", (provider,)).fetchall()
    return [r[0] for r in rows]


def last_credits_remaining(con: sqlite3.Connection, provider: str):
    r = con.execute("SELECT credits_remaining FROM odds_collection_runs WHERE provider=? AND "
                    "credits_remaining IS NOT NULL ORDER BY id DESC LIMIT 1", (provider,)).fetchone()
    return r[0] if r else None


def collect(con: sqlite3.Connection, provider, games: pd.DataFrame, schedule: odds_schedule.ScheduleConfig,
            now: pd.Timestamp | None = None, force: bool = False, budget_guard: bool = True) -> dict:
    """Fetch and store one snapshot if a scheduled target is due (or `force`).

    Returns a summary dict with status "ok", "skipped" or "error"."""
    now = now or pd.Timestamp.now(tz="UTC")
    upcoming = games[~games["played"]] if "played" in games else games
    offsets = schedule.offsets_minutes
    if budget_guard:
        offsets = odds_schedule.affordable_offsets(upcoming, schedule, now,
                                                   last_credits_remaining(con, provider.name))
    tg = odds_schedule.targets_for(upcoming, schedule, offsets)
    due = odds_schedule.due(tg, call_times(con, provider.name), now)
    if due.empty and not force:
        return {"status": "skipped", "reason": "no target due", "offsets_in_use": list(offsets)}
    targets = due[["game_key", "offset", "target"]].assign(target=lambda d: d["target"].astype(str)).to_dict("records")
    run_id = odds_store.start_run(con, provider.name, "current", targets, now)
    try:
        rows, usage = provider.fetch_current()
    except odds_api.OddsApiError as exc:
        odds_store.finish_run(con, run_id, "error", error=str(exc))
        log.error("odds collection failed: %s", exc)
        return {"status": "error", "error": str(exc), "run_id": run_id}
    retrieved = pd.Timestamp.now(tz="UTC").isoformat()
    frame = odds_store.from_provider_frame(rows, provider.name, f"{provider.name}:current", retrieved)
    counts = odds_store.store_snapshots(con, frame, games, run_id)
    n_events = int(rows["event_id"].nunique()) if len(rows) else 0
    reason = "forced" if force and due.empty else f"{len(due)} target(s) due"
    odds_store.finish_run(con, run_id, "ok", counts, usage, reason=reason, n_events=n_events)
    matched = set(odds_store.map_to_games(frame, games)["game_key"].dropna())
    missing = sorted(set(due["game_key"]) - matched)
    if missing:
        log.warning("no prices returned for %d game(s) with due targets: %s", len(missing), missing)
    return {"status": "ok", "run_id": run_id, "n_events": n_events, "usage": usage, "offsets_in_use": list(offsets),
            "targets_due": len(due), "due_games_without_prices": missing, **counts}


def coverage(con: sqlite3.Connection, games: pd.DataFrame, schedule: odds_schedule.ScheduleConfig,
             provider: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Per game in [start, end]: pregame snapshot count, books, first/last observation, which
    scheduled offsets were hit, and the gap between the last pregame snapshot and tip-off."""
    tip = pd.to_datetime(games["tipoff_utc"], utc=True, errors="coerce")
    g = games[(tip >= start) & (tip <= end)].assign(tip=tip)
    snap = odds_store.load_snapshots(con)
    tg = odds_schedule.targets_for(g, schedule)
    tg["hit"] = odds_schedule.satisfied(tg, call_times(con, provider))
    rows = []
    for r in g.itertuples(index=False):
        s = snap[snap["game_key"] == r.game_key]
        t = tg[tg["game_key"] == r.game_key]
        rows.append({"game_key": r.game_key, "date": str(pd.Timestamp(r.date).date()),
                     "game": f"{r.away_team} @ {r.home_team}", "tipoff_utc": r.tip.isoformat(),
                     "n_snapshots": int(s["observed_at"].nunique()), "n_books": int(s["book"].nunique()),
                     "first_observed": s["observed_at"].min().isoformat() if len(s) else None,
                     "last_observed": s["observed_at"].max().isoformat() if len(s) else None,
                     "minutes_before_tip_of_last": round((r.tip - s["observed_at"].max()).total_seconds() / 60, 1)
                     if len(s) else None,
                     "offsets_hit": ",".join(str(o) for o in t.loc[t["hit"], "offset"]),
                     "offsets_missed": ",".join(str(o) for o in t.loc[~t["hit"] & (t["end"] < pd.Timestamp.now(tz="UTC")),
                                                                      "offset"])})
    return pd.DataFrame(rows)
