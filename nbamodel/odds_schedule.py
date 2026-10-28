"""When to spend API credits on odds snapshots.

Each scheduled game gets target times at fixed offsets before tip-off (default 24h, 6h, 1h,
15 min). One current-odds call returns every NBA game, so one call can satisfy many targets: a
call at time t satisfies a target T if t falls in T's window [T - early, T + late] and t is
before that game's tip-off.

The collector runs on a short poll (cron every few minutes) and calls the API only when some
unsatisfied target's time has arrived (T <= now <= T + late). `simulate` replays exactly that
rule over a schedule to estimate credit use, so the plan and the live behaviour agree.

The actual observation time of every snapshot is stored; target times are only intentions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# offset_minutes -> (early, late) tolerance in minutes. `late` stays below the offset so a
# satisfying call is always before tip-off.
DEFAULT_WINDOWS = {1440: (120, 120), 360: (45, 60), 60: (10, 20), 15: (5, 8)}
# Dropped first when the budget is short: the closing (15) and decision (60) snapshots matter most.
DEFAULT_PRIORITY = (15, 60, 360, 1440)


@dataclass
class ScheduleConfig:
    offsets_minutes: tuple = (1440, 360, 60, 15)
    windows: dict = field(default_factory=lambda: dict(DEFAULT_WINDOWS))
    priority: tuple = DEFAULT_PRIORITY
    poll_minutes: int = 5
    credits_per_call: int = 1            # markets x regions on The Odds API (h2h, us)
    monthly_credit_budget: int = 500     # free plan
    reserve_credits: int = 25            # never plan to spend these (manual checks, retries)

    @classmethod
    def from_dict(cls, d: dict | None) -> "ScheduleConfig":
        d = dict(d or {})
        if "windows" in d:
            d["windows"] = {int(k): tuple(v) for k, v in d["windows"].items()}
        for k in ("offsets_minutes", "priority"):
            if k in d:
                d[k] = tuple(int(x) for x in d[k])
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def window(self, offset: int) -> tuple:
        if offset in self.windows:
            return self.windows[offset]
        return (max(1, offset // 12), max(1, offset // 3))


def targets_for(games: pd.DataFrame, cfg: ScheduleConfig, offsets=None) -> pd.DataFrame:
    """One row per (game, offset): target time and its tolerance window. Games without a known
    tip-off are skipped (no target can be placed relative to an unknown start)."""
    offsets = tuple(offsets if offsets is not None else cfg.offsets_minutes)
    tip = pd.to_datetime(games["tipoff_utc"], utc=True, errors="coerce")
    g = games.assign(tip=tip)[tip.notna()]
    rows = []
    for r in g.itertuples(index=False):
        for off in offsets:
            early, late = cfg.window(off)
            t = r.tip - pd.Timedelta(minutes=off)
            rows.append({"game_key": r.game_key, "tip": r.tip, "offset": off, "target": t,
                         "start": t - pd.Timedelta(minutes=early), "end": t + pd.Timedelta(minutes=late)})
    cols = ["game_key", "tip", "offset", "target", "start", "end"]
    return pd.DataFrame(rows, columns=cols).sort_values("target").reset_index(drop=True)


def satisfied(targets: pd.DataFrame, call_times) -> pd.Series:
    """Which targets already have a call inside their window (and before tip-off)."""
    if targets.empty or len(call_times) == 0:
        return pd.Series(False, index=targets.index)
    ns = lambda s: pd.to_datetime(s, utc=True).astype("int64").to_numpy()  # noqa: E731
    times = np.sort(ns(pd.Series(list(call_times))))
    start = ns(targets["start"])
    end = np.minimum(ns(targets["end"]), ns(targets["tip"]) - 1)
    first = np.searchsorted(times, start, side="left")
    return pd.Series((first < len(times)) & (times[np.minimum(first, len(times) - 1)] <= end), index=targets.index)


def due(targets: pd.DataFrame, call_times, now: pd.Timestamp) -> pd.DataFrame:
    """Targets that justify a call right now: unsatisfied, target time reached, window still open,
    game not started."""
    if targets.empty:
        return targets
    open_ = (targets["target"] <= now) & (now <= targets["end"]) & (now < targets["tip"])
    return targets[open_ & ~satisfied(targets, call_times)]


def next_target(targets: pd.DataFrame, call_times, now: pd.Timestamp):
    """Earliest target time that still needs a call: unsatisfied, window not closed, game not started.
    It can be in the past when a target is due right now. None when nothing is left."""
    if targets.empty:
        return None
    left = targets[(now <= targets["end"]) & (now < targets["tip"]) & ~satisfied(targets, call_times)]
    return None if left.empty else left["target"].min()


def simulate(targets: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, poll_minutes: int = 5,
             prior_calls=()) -> list:
    """Replay the live rule (poll every `poll_minutes`, call when something is due) and return the
    call times it would make."""
    calls = list(pd.to_datetime(list(prior_calls), utc=True))
    if targets.empty:
        return []
    t = targets[(targets["end"] >= start) & (targets["target"] <= end)]
    made = []
    now = start.floor(f"{poll_minutes}min")
    pending = t.copy()
    step = pd.Timedelta(minutes=poll_minutes)
    while now <= end and not pending.empty:
        nxt = pending["target"].min()
        if nxt > now:  # skip idle polls
            now = nxt.ceil(f"{poll_minutes}min")
            continue
        if not due(pending, calls, now).empty:
            calls.append(now)
            made.append(now)
            pending = pending[~satisfied(pending, calls) & (pending["end"] >= now)]
        else:
            pending = pending[(pending["end"] >= now) & ~satisfied(pending, calls)]
        now += step
    return made


def plan(games: pd.DataFrame, cfg: ScheduleConfig, start: pd.Timestamp, end: pd.Timestamp,
         offsets=None) -> dict:
    """Credit estimate for collecting `offsets` over [start, end]."""
    tg = targets_for(games, cfg, offsets)
    calls = simulate(tg, start, end, cfg.poll_minutes)
    days = max(1.0, (end - start) / pd.Timedelta(days=1))
    tip = pd.to_datetime(games["tipoff_utc"], utc=True, errors="coerce")
    in_range = games[(tip >= start) & (tip <= end)]
    by_month = pd.Series(1, index=pd.DatetimeIndex(calls)).groupby(
        pd.DatetimeIndex(calls).tz_convert("US/Eastern").strftime("%Y-%m")).sum() if calls else pd.Series(dtype=int)
    return {"offsets_minutes": list(offsets if offsets is not None else cfg.offsets_minutes),
            "start": start.isoformat(), "end": end.isoformat(), "n_games": int(len(in_range)),
            "n_game_days": int(pd.to_datetime(in_range["date"]).nunique()) if len(in_range) else 0,
            "n_calls": len(calls), "credits": len(calls) * cfg.credits_per_call,
            "credits_per_30_days": round(len(calls) * cfg.credits_per_call / days * 30, 1),
            "max_calls_per_month": int(by_month.max()) if len(by_month) else 0,
            "calls_by_month": {k: int(v) for k, v in by_month.items()}}


def affordable_offsets(games: pd.DataFrame, cfg: ScheduleConfig, now: pd.Timestamp,
                       credits_remaining: int | None) -> tuple:
    """Largest set of offsets (dropping the lowest priority first) whose projected use until the
    provider's monthly reset fits in the remaining credits, less the reserve.

    The Odds API resets usage monthly; we assume the first of each UTC month. With no known
    remaining balance, the monthly budget is used."""
    month_end = (now + pd.offsets.MonthBegin(1)).normalize()
    budget = (credits_remaining if credits_remaining is not None else cfg.monthly_credit_budget) - cfg.reserve_credits
    order = [o for o in cfg.priority if o in cfg.offsets_minutes]
    order += [o for o in cfg.offsets_minutes if o not in order]
    for k in range(len(order), 0, -1):
        offs = tuple(sorted(order[:k], reverse=True))
        p = plan(games, cfg, now, month_end, offs)
        if p["credits"] <= budget:
            return offs
    return ()
