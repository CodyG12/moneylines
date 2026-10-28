"""Official NBA injury reports: fetch, parse and archive with publication timestamps.

Source: the league's official reports (https://official.nba.com/nba-injury-report-2026-27-season/),
published as PDFs at https://ak-static.cms.nba.com/referee/injury/. File names carry a nominal
time (hourly, e.g. Injury-Report_2023-01-10_05PM.pdf; every 15 minutes since 2026-27, e.g.
Injury-Report_2026-10-27_05_00PM.pdf); the page header carries the actual publication time
("Injury Report: 01/10/23 05:30 PM", US/Eastern), which is what every reader here uses. The
archive goes back to the 2019-20 season.

Storage (SQLite tables in store.INJURY_SCHEMA) is append-only:
  injury_report_files  one row per URL tried: publication and retrieval times, HTTP status,
                       sha256, rows parsed, parse status. Missing files are recorded too.
  injury_reports       a change log. A player's row is written when their status or reason differs
                       from their previous observation for that game, and a `not_listed` row when
                       a report still lists the game but no longer lists the player. Replaying the
                       log up to any publication time gives exactly that report's contents, so no
                       snapshot is lost and nothing is overwritten.

An unlisted player is not assumed available: readers return the listed statuses and a coverage
flag, and features decide how to treat absence. Final participation (box scores) is never used as
a substitute for what the report said before tip-off.
"""
from __future__ import annotations

import hashlib
import io
import logging
import re
import sqlite3
import time
import unicodedata
import urllib.error
import urllib.request

import pandas as pd

from .odds_api import TEAM_CODES

log = logging.getLogger(__name__)

SOURCE = "nba-official-injury-report"
BASE_URL = "https://ak-static.cms.nba.com/referee/injury/"
SEASON_PAGE = "https://official.nba.com/nba-injury-report-{season}-season/"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0 Safari/537.36")
ET = "US/Eastern"
STATUSES = ("out", "doubtful", "questionable", "probable", "available")
NOT_LISTED, NOT_SUBMITTED = "not_listed", "not_submitted"
COLUMNS = ("game_date", "game_time", "matchup", "team", "player", "status", "reason")
HEADER_WORDS = ("Game", "Game", "Matchup", "Team", "Player", "Current", "Reason")
_TEAMS = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in TEAM_CODES.items()}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


# ------------------------------------------------------------------ names
def team_code(text: str | None) -> str | None:
    """'Brooklyn Nets' / 'BrooklynNets' -> 'BKN'."""
    if not text:
        return None
    return _TEAMS.get(re.sub(r"[^a-z0-9]", "", text.lower()))


def display_name(raw: str) -> str:
    """'Bagley III, Marvin' / 'Ajayi,Michael' -> 'Marvin Bagley III' / 'Michael Ajayi'."""
    raw = re.sub(r"\s+", " ", raw).strip()
    if "," in raw:
        last, first = (s.strip() for s in raw.split(",", 1))
        return f"{first} {last}".strip()
    return raw


def player_key(name: str) -> str:
    """Matching key that survives accents, punctuation, spacing and suffixes:
    'Luka Dončić' -> 'lukadoncic', 'Marvin Bagley III' -> 'marvinbagley'."""
    s = unicodedata.normalize("NFKD", display_name(name)).encode("ascii", "ignore").decode().lower()
    parts = [p for p in re.split(r"[^a-z0-9]+", s) if p]
    while len(parts) > 2 and parts[-1] in _SUFFIXES:
        parts.pop()
    return "".join(parts)


# ------------------------------------------------------------------ URLs
def report_url(nominal_et: pd.Timestamp, quarter_hourly: bool) -> str:
    """URL of the report nominally published at `nominal_et` (naive US/Eastern time)."""
    t = pd.Timestamp(nominal_et)
    stem = t.strftime("%Y-%m-%d_%I") + (t.strftime("_%M") if quarter_hourly else "") + t.strftime("%p")
    return f"{BASE_URL}Injury-Report_{stem}.pdf"


def nominal_time(url: str) -> pd.Timestamp | None:
    m = re.search(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})(?:_(\d{2}))?(AM|PM)\.pdf", url)
    if not m:
        return None
    return pd.Timestamp(f"{m[1]} {m[2]}:{m[3] or '00'} {m[4]}")


def season_label(day: pd.Timestamp) -> str:
    start = day.year if day.month >= 8 else day.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def list_season_links(season: str, timeout: float = 30.0) -> list:
    """Report URLs linked from the season page. The page lists only the current day's reports."""
    html = _get(SEASON_PAGE.format(season=season), timeout)[1].decode("utf-8", "replace")
    return sorted(set(re.findall(r"https?://[^\"'\s]*Injury-Report_[^\"'\s]*\.pdf", html)))


def _get(url: str, timeout: float = 30.0) -> tuple:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""


# ------------------------------------------------------------------ parsing
def pdf_words(pdf_bytes: bytes) -> list:
    """Positioned words per page: [[{"x0", "top", "text"}, ...], ...]."""
    import pdfplumber  # optional dependency: only needed to read PDFs

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return [[{"x0": w["x0"], "top": w["top"], "text": w["text"]}
                 for w in page.extract_words(x_tolerance=1.5, keep_blank_chars=False)] for page in pdf.pages]


def _lines(words: list, tol: float = 3.0) -> list:
    """Group words into lines by vertical position, left to right."""
    out = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if out and abs(out[-1][0] - w["top"]) <= tol:
            out[-1][1].append(w)
        else:
            out.append([w["top"], [w]])
    return [(top, sorted(ws, key=lambda w: w["x0"])) for top, ws in out]


def _published(pages: list) -> pd.Timestamp | None:
    for words in pages[:1]:
        for _, ws in _lines(words):
            m = re.search(r"Injury Report:\s*(\d{2}/\d{2}/\d{2})\s+(\d{1,2}:\d{2})\s*(AM|PM)",
                          " ".join(w["text"] for w in ws))
            if m:
                local = pd.to_datetime(f"{m[1]} {m[2]} {m[3]}", format="%m/%d/%y %I:%M %p")
                return local.tz_localize(ET).tz_convert("UTC")
    return None


def _edges(lines: list) -> tuple:
    """(index of the header line, x0 of each column) from the 'Game Date ... Reason' header."""
    for i, (_, ws) in enumerate(lines):
        texts = [w["text"] for w in ws]
        if "Matchup" in texts and "Reason" in texts:
            starts, k = [], 0
            for w in ws:
                if k < len(HEADER_WORDS) and w["text"] == HEADER_WORDS[k]:
                    starts.append(w["x0"])
                    k += 1
            if len(starts) == len(COLUMNS):
                return i, starts
    return None, None


def parse_words(pages: list) -> tuple:
    """(published_at UTC, rows) from positioned words. Rows have game_date (ISO), game_time,
    matchup, away_team, home_team, team, player_name, status, reason, page. Team, matchup, date and
    time are printed only on the first row of their group, so they are carried forward; a reason
    that wraps onto two lines is printed above and below its player's row and is reattached."""
    published = _published(pages)
    ctx = {"game_date": None, "game_time": None, "matchup": None, "team": None}
    team_part = None
    rows, orphans = [], []
    edges = None
    for pno, words in enumerate(pages):
        lines = _lines(words)
        hi, page_edges = _edges(lines)
        if page_edges is not None:
            edges, body = page_edges, lines[hi + 1:]
        elif edges is not None:  # later pages repeat only the title, not the column header
            body = lines
        else:
            continue
        for top, ws in body:
            text = " ".join(w["text"] for w in ws)
            if re.search(r"Page\s*\d+\s*of\s*\d+", text) or text.startswith("Injury Report:"):
                continue
            cells = {c: [] for c in COLUMNS}
            for w in ws:
                col = max(i for i in range(len(edges)) if edges[i] - 4 <= w["x0"] or i == 0)
                cells[COLUMNS[col]].append(w["text"])
            cell = {c: " ".join(v).strip() for c, v in cells.items()}
            if re.match(r"\d{2}/\d{2}/\d{4}", cell["game_date"]):
                ctx["game_date"] = pd.to_datetime(cell["game_date"][:10], format="%m/%d/%Y").date().isoformat()
            if re.match(r"\d{1,2}:\d{2}", cell["game_time"]):
                ctx["game_time"] = cell["game_time"][:5]
            if re.match(r"[A-Z]{2,3}\s*@\s*[A-Z]{2,3}", cell["matchup"]):
                ctx["matchup"] = re.sub(r"\s", "", cell["matchup"])
                ctx["team"], team_part = None, None
            if cell["team"]:
                code = team_code(cell["team"]) or (team_code(f"{team_part} {cell['team']}") if team_part else None)
                if code:
                    ctx["team"], team_part = code, None
                else:
                    team_part = cell["team"] if team_part is None else f"{team_part} {cell['team']}"
            if "NOT YET SUBMITTED" in text.upper():
                rows.append({**ctx, "player_raw": None, "status": NOT_SUBMITTED, "reason": None,
                             "page": pno, "top": top, "own_reason": True})
                continue
            status = cell["status"].lower()
            if cell["player"] or status in STATUSES:
                rows.append({**ctx, "player_raw": cell["player"] or None,
                             "status": status if status in STATUSES else (status or None),
                             "reason": cell["reason"] or None, "page": pno, "top": top,
                             "own_reason": bool(cell["reason"])})
            elif cell["reason"]:
                orphans.append((pno, top, cell["reason"]))
    for pno, top, frag in orphans:
        near = [(abs(r["top"] - top), not r["own_reason"], i) for i, r in enumerate(rows)
                if r["page"] == pno and abs(r["top"] - top) <= 16 and r["player_raw"]]
        if not near:
            continue
        _, _, i = min(near, key=lambda t: (not t[1], t[0]))
        rows[i].setdefault("frags", []).append((top, frag))
    out = []
    for r in rows:
        parts = sorted([(r["top"], r["reason"])] if r["reason"] else [], key=lambda t: t[0])
        parts = sorted(parts + r.get("frags", []), key=lambda t: t[0])
        away, home = (r["matchup"].split("@") + [None])[:2] if r["matchup"] else (None, None)
        out.append({"game_date": r["game_date"], "game_time": r["game_time"], "matchup": r["matchup"],
                    "away_team": away, "home_team": home, "team": r["team"],
                    "player_name": display_name(r["player_raw"]) if r["player_raw"] else None,
                    "status": r["status"], "reason": " ".join(p for _, p in parts) or None, "page": r["page"]})
    cols = ["game_date", "game_time", "matchup", "away_team", "home_team", "team", "player_name", "status",
            "reason", "page"]
    return published, pd.DataFrame(out, columns=cols)


def parse_report(pdf_bytes: bytes) -> tuple:
    return parse_words(pdf_words(pdf_bytes))


# ------------------------------------------------------------------ storage
def _state(con: sqlite3.Connection, game_dates) -> dict:
    """Latest observation per (game_date, matchup, team, player_key) for these game dates."""
    if not game_dates:
        return {}
    q = ",".join("?" * len(game_dates))
    df = pd.read_sql(f"SELECT game_date, matchup, team, player_key, status, reason, published_at "
                     f"FROM injury_reports WHERE game_date IN ({q}) ORDER BY published_at, id",
                     con, params=list(game_dates))
    return {(r.game_date, r.matchup, r.team, r.player_key): (r.status, r.reason) for r in df.itertuples()}


def latest_published(con: sqlite3.Connection, game_dates, asof: str | None = None) -> str | None:
    """Publication time of the newest ingested report listing any of `game_dates` (optionally at
    or before `asof`). Read from the files table: an unchanged report adds no change-log rows but
    is still the freshest evidence."""
    sql, params = "SELECT published_at, game_dates FROM injury_report_files WHERE parse_status = 'ok'", []
    if asof is not None:
        sql += " AND published_at <= ?"
        params.append(asof)
    want = set(game_dates)
    best = None
    for pub, dates in con.execute(sql, params):
        if dates and want & set(dates.split(",")) and (best is None or pub > best):
            best = pub
    return best


def ingest(con: sqlite3.Connection, url: str, status_code: int, body: bytes, retrieved_at: pd.Timestamp,
           games: pd.DataFrame | None = None, parsed=None) -> dict:
    """Record one fetched URL and append the change log rows its report implies.

    Reports must be ingested in publication order per game date: rows for a date already covered
    by a newer report are skipped, and a file with nothing newer is recorded as `out_of_order`."""
    retrieved = pd.Timestamp(retrieved_at).isoformat()
    if con.execute("SELECT 1 FROM injury_report_files WHERE url = ?", (url,)).fetchone():
        return {"status": "duplicate", "url": url}
    nominal = nominal_time(url)
    base = (url, SOURCE, nominal.isoformat() if nominal is not None else None, retrieved, int(status_code))
    if status_code != 200 or not body:
        con.execute("INSERT OR IGNORE INTO injury_report_files (url, source, nominal_et, retrieved_at, http_status, "
                    "parse_status) VALUES (?,?,?,?,?,?)", (*base, "missing"))
        con.commit()
        return {"status": "missing", "url": url}
    sha = hashlib.sha256(body).hexdigest()
    try:
        published, rows = parsed if parsed is not None else parse_report(body)
    except Exception as exc:  # malformed PDF: keep the evidence, add nothing
        con.execute("INSERT OR IGNORE INTO injury_report_files (url, source, nominal_et, retrieved_at, http_status, "
                    "sha256, parse_status, error) VALUES (?,?,?,?,?,?,?,?)", (*base, sha, "error", str(exc)[:500]))
        con.commit()
        return {"status": "error", "url": url, "error": str(exc)}
    if published is None:
        published = nominal.tz_localize(ET).tz_convert("UTC") if nominal is not None else None
    pub = published.isoformat() if published is not None else None
    rows = rows[rows["game_date"].notna() & rows["team"].notna()].copy()
    # Per game date: a report only counts for the dates it is newer for. (A day-D report also
    # lists day D+1 games, which a later live report may already cover.)
    listed = sorted(set(rows["game_date"]))
    dates = [d for d in listed if pub is not None and (latest_published(con, [d]) or "") < pub]
    rows = rows[rows["game_date"].isin(dates)]
    if not dates:
        con.execute("INSERT OR IGNORE INTO injury_report_files (url, source, nominal_et, published_at, retrieved_at, "
                    "http_status, sha256, n_rows, parse_status, game_dates) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (*base[:3], pub, retrieved, int(status_code), sha, 0,
                     "out_of_order" if pub else "no_timestamp", ",".join(listed)))
        con.commit()
        return {"status": "out_of_order" if pub else "no_timestamp", "url": url}
    cur = con.execute("INSERT OR IGNORE INTO injury_report_files (url, source, nominal_et, published_at, "
                      "retrieved_at, http_status, sha256, n_rows, parse_status, game_dates) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (*base[:3], pub, retrieved, int(status_code), sha, len(rows), "ok", ",".join(dates)))
    if cur.rowcount == 0:  # URL already ingested
        return {"status": "duplicate", "url": url}
    file_id = cur.lastrowid
    keys = _game_keys(games)
    rows["player_key"] = rows["player_name"].map(lambda n: player_key(n) if n else "")
    prev = _state(con, dates)
    seen, inserts = set(), []
    for r in rows.itertuples(index=False):
        k = (r.game_date, r.matchup, r.team, r.player_key)
        seen.add(k)
        if prev.get(k) != (r.status, r.reason):
            inserts.append((r, r.status, r.reason))
    # A listed game's report is current for both teams, except a team that has not yet submitted.
    listed_games = {(r.game_date, r.matchup) for r in rows.itertuples(index=False)}
    pending = {(r.game_date, r.matchup, r.team) for r in rows.itertuples(index=False) if r.status == NOT_SUBMITTED}
    for (d, m, t, pk), (st, _) in prev.items():
        if (d, m) in listed_games and (d, m, t) not in pending and (d, m, t, pk) not in seen and st != NOT_LISTED:
            inserts.append((pd.Series({"game_date": d, "game_time": None, "matchup": m, "team": t,
                                       "player_name": None, "player_key": pk,
                                       "away_team": m.split("@")[0] if m else None,
                                       "home_team": m.split("@")[1] if m and "@" in m else None}), NOT_LISTED, None))
    con.executemany(
        "INSERT OR IGNORE INTO injury_reports (file_id, published_at, retrieved_at, source, game_date, game_time_et, "
        "matchup, home_team, away_team, game_key, team, player_name, player_key, status, reason) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(file_id, pub, retrieved, SOURCE, r.game_date, r.game_time, r.matchup, r.home_team, r.away_team,
          keys.get((r.game_date, r.home_team, r.away_team)), r.team, r.player_name, r.player_key, st, reason)
         for r, st, reason in inserts])
    con.commit()
    return {"status": "ok", "url": url, "published_at": pub, "n_rows": len(rows), "n_changes": len(inserts)}


def _game_keys(games: pd.DataFrame | None) -> dict:
    if games is None or games.empty:
        return {}
    d = pd.to_datetime(games["date"]).dt.date.astype(str)
    return dict(zip(zip(d, games["home_team"], games["away_team"]), games["game_key"]))


def known_urls(con: sqlite3.Connection) -> set:
    return {r[0] for r in con.execute("SELECT url FROM injury_report_files")}


def fetch_and_ingest(con: sqlite3.Connection, urls, games=None, pause: float = 1.0, timeout: float = 30.0) -> list:
    """Fetch URLs not tried before, in nominal-time order, and ingest each one."""
    have = known_urls(con)
    todo = sorted((u for u in set(urls) if u not in have), key=lambda u: (nominal_time(u) or pd.Timestamp.min, u))
    out = []
    for i, url in enumerate(todo):
        if i and pause:
            time.sleep(pause)
        try:
            code, body = _get(url, timeout)
        except OSError as exc:  # network failure: not recorded, so it is retried next time
            log.warning("injury report fetch failed for %s: %s", url, exc)
            out.append({"status": "network_error", "url": url, "error": str(exc)})
            continue
        out.append(ingest(con, url, code, body, pd.Timestamp.now(tz="UTC"), games))
    return out


# ------------------------------------------------------------------ reading
def statuses_asof(con: sqlite3.Connection, asof: pd.Timestamp, game_dates=None) -> pd.DataFrame:
    """Each player's status in the latest report published at or before `asof` (UTC), per game.
    Players dropped from a report that still lists their game are excluded; players never listed
    are simply absent (unknown, not 'available'). Includes report_published_at for freshness."""
    asof = pd.Timestamp(asof)
    asof = asof.tz_localize("UTC") if asof.tzinfo is None else asof.tz_convert("UTC")
    where, params = "published_at <= ?", [asof.isoformat()]
    if game_dates is not None:
        game_dates = list(game_dates)
        where += f" AND game_date IN ({','.join('?' * len(game_dates))})"
        params += game_dates
    df = pd.read_sql(f"SELECT * FROM injury_reports WHERE {where} ORDER BY published_at, id", con, params=params)
    if df.empty:
        return df
    last = df.groupby(["game_date", "matchup", "team", "player_key"], dropna=False).tail(1)
    return last[last["status"] != NOT_LISTED].reset_index(drop=True)


def report_coverage(con: sqlite3.Connection, asof: pd.Timestamp, game_date: str) -> dict:
    """Freshness of the latest report at `asof` that lists `game_date`, and which teams had not
    yet submitted in it."""
    asof = pd.Timestamp(asof)
    asof = asof.tz_localize("UTC") if asof.tzinfo is None else asof.tz_convert("UTC")
    r = latest_published(con, [game_date], asof.isoformat())
    if r is None:
        return {"report_published_at": None, "report_age_minutes": None, "not_submitted": []}
    st = statuses_asof(con, asof, [game_date])
    pending = sorted(set(st.loc[st["status"] == NOT_SUBMITTED, "team"]))
    return {"report_published_at": r, "report_age_minutes": (asof - pd.Timestamp(r)) / pd.Timedelta(minutes=1),
            "not_submitted": pending}


def status_changes(con: sqlite3.Connection, game_key: str) -> pd.DataFrame:
    """Every recorded status change for one game, in publication order."""
    return pd.read_sql("SELECT published_at, team, player_name, player_key, status, reason FROM injury_reports "
                       "WHERE game_key = ? ORDER BY published_at, id", con, params=(game_key,))


# ------------------------------------------------------------------ live refresh and backfill
def refresh(con: sqlite3.Connection, games=None, now: pd.Timestamp | None = None, pause: float = 0.5) -> list:
    """Ingest any of today's reports (linked from the season page) not tried before."""
    now = now or pd.Timestamp.now(tz="UTC")
    links = list_season_links(season_label(now.tz_convert(ET).tz_localize(None)))
    return fetch_and_ingest(con, links, games, pause=pause)


def backfill_urls(day: pd.Timestamp, last_tip_et: pd.Timestamp, quarter_hourly: bool, first_hour: int = 11) -> list:
    """Candidate report URLs for one game day: hourly (or on the hour and half hour, in the
    15-minute era) from `first_hour` ET until an hour before the day's last tip-off, which is the
    latest a pregame prediction (lead 60 min) can use."""
    day = pd.Timestamp(day).normalize()
    t, end = day + pd.Timedelta(hours=first_hour), pd.Timestamp(last_tip_et) - pd.Timedelta(hours=1)
    step = pd.Timedelta(minutes=30 if quarter_hourly else 60)
    out = []
    while t <= end:
        out.append(report_url(t, quarter_hourly))
        t += step
    return out


def backfill(con: sqlite3.Connection, games: pd.DataFrame, start: str, end: str, max_seconds: float = 3600,
             pause: float = 1.0) -> dict:
    """Fetch historical reports for game days in [start, end], oldest first, until `max_seconds`.
    Every URL tried is recorded (missing ones too), so reruns resume where this one stopped. The
    file-name format changed from hourly to 15-minute; each day tries the format that worked last,
    and the other one only if the first finds nothing."""
    t0 = time.monotonic()
    tip = pd.to_datetime(games["tipoff_utc"], utc=True, errors="coerce").dt.tz_convert(ET).dt.tz_localize(None)
    g = games.assign(_tip=tip, _day=pd.to_datetime(games["date"]).dt.normalize())
    g = g[(g["_day"] >= pd.Timestamp(start)) & (g["_day"] <= pd.Timestamp(end)) & g["_tip"].notna()]
    last_tips = g.groupby("_day")["_tip"].max().sort_index()
    have = known_urls(con)
    quarter = False
    done = fetched = found = 0
    for day, last_tip in last_tips.items():
        for fmt in (quarter, not quarter):
            urls = backfill_urls(day, last_tip, fmt)
            todo = [u for u in urls if u not in have]
            if not todo:
                ok = con.execute(f"SELECT COUNT(*) FROM injury_report_files WHERE parse_status != 'missing' AND url IN "
                                 f"({','.join('?' * len(urls))})", urls).fetchone()[0]
                if ok:
                    quarter = fmt
                    break
                continue
            if time.monotonic() - t0 > max_seconds:
                return {"complete": False, "days_done": done, "fetched": fetched, "found": found, "next_day": str(day.date())}
            res = fetch_and_ingest(con, todo, games, pause=pause)
            if any(r["status"] == "network_error" for r in res):
                return {"complete": False, "days_done": done, "fetched": fetched, "found": found,
                        "next_day": str(day.date()), "error": "network"}
            have.update(r["url"] for r in res)
            fetched += len(res)
            hits = sum(r["status"] != "missing" for r in res)
            found += hits
            if hits:
                quarter = fmt
                break
        done += 1
    return {"complete": True, "days_done": done, "fetched": fetched, "found": found, "next_day": None}
