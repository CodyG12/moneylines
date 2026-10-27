"""Client for The Odds API (https://the-odds-api.com), NBA moneylines (h2h).

Current odds work on the free plan (1 credit per call for one market and
region). Historical snapshots need a paid plan (10 credits per call) and go
back to June 2020. Responses are flattened into "snapshot" rows: one row per
(observation time, game, bookmaker), with decimal prices for each side.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

SNAPSHOT_COLUMNS = ["observed_at", "event_id", "commence_time", "home_team", "away_team",
                    "home_name", "away_name", "book", "market_last_update", "home_price", "away_price"]

TEAM_CODES = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


# Per-call credit cost on The Odds API: markets x regions (current odds); x10 for historical.
CREDITS_PER_CALL = {"current": 1, "historical": 10}
RETRY_STATUS = {429, 500, 502, 503, 504}


class OddsApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status in RETRY_STATUS


def load_api_key(env_file: str | Path = ".env") -> str:
    """ODDS_API_KEY from the environment, else from a local .env file."""
    key = os.environ.get("ODDS_API_KEY")
    if not key and Path(env_file).exists():
        for line in Path(env_file).read_text().splitlines():
            if line.strip().startswith("ODDS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise OddsApiError("ODDS_API_KEY is not set (environment or .env)")
    return key


def _get_once(path: str, params: dict, timeout: float = 30.0):
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            usage = {h: resp.headers.get(h) for h in ("x-requests-remaining", "x-requests-used", "x-requests-last")}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise OddsApiError(f"HTTP {exc.code} from The Odds API: {detail}", exc.code) from exc
    except (urllib.error.URLError, socket.timeout, ConnectionError) as exc:
        raise OddsApiError(f"network error calling The Odds API: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OddsApiError(f"invalid JSON from The Odds API: {exc}") from exc
    log.info("odds api usage: %s", usage)
    return body, usage


def with_retries(call, retries: int = 4, base_delay: float = 2.0, max_delay: float = 60.0, sleep=time.sleep):
    """Run `call()`, retrying rate limits (429), server errors and network failures with
    exponential backoff (base_delay * 2**attempt, capped). Auth/quota/validation errors
    (401, 403, 422, ...) are raised at once: retrying them only burns time."""
    for attempt in range(retries + 1):
        try:
            return call()
        except OddsApiError as exc:
            if not exc.retryable or attempt == retries:
                raise
            delay = min(max_delay, base_delay * 2 ** attempt)
            log.warning("odds api attempt %d failed (%s); retrying in %.0fs", attempt + 1, exc, delay)
            sleep(delay)


def _get(path: str, params: dict, timeout: float = 30.0, retries: int = 4, sleep=time.sleep):
    return with_retries(lambda: _get_once(path, params, timeout), retries=retries, sleep=sleep)


def fetch_current(api_key: str, regions: str = "us", bookmakers: str | None = None, odds_format: str = "decimal",
                  **retry):
    """Current prices. `observed_at` is our retrieval time: the response is the provider's
    view at that moment (each market also carries the provider's own `last_update`)."""
    params = {"apiKey": api_key, "regions": regions, "markets": "h2h", "oddsFormat": odds_format,
              "dateFormat": "iso"}
    if bookmakers:
        params["bookmakers"] = bookmakers
    body, usage = _get(f"/sports/{SPORT}/odds", params, **retry)
    observed_at = pd.Timestamp.now(tz="UTC").isoformat()
    return parse_events(body, observed_at, odds_format), usage


def fetch_historical(api_key: str, at_iso: str, regions: str = "us", bookmakers: str | None = None,
                     odds_format: str = "decimal", **retry):
    """Snapshot as of `at_iso` (paid plans only). The snapshot's own timestamp,
    which is at or before `at_iso`, becomes `observed_at`."""
    params = {"apiKey": api_key, "regions": regions, "markets": "h2h", "oddsFormat": odds_format,
              "dateFormat": "iso", "date": at_iso}
    if bookmakers:
        params["bookmakers"] = bookmakers
    body, usage = _get(f"/historical/sports/{SPORT}/odds", params, **retry)
    return parse_events(body.get("data", []), body["timestamp"], odds_format), usage


def fetch_events(api_key: str, **retry):
    """Upcoming NBA events (ids, teams, commence times). Costs no credits."""
    return _get(f"/sports/{SPORT}/events", {"apiKey": api_key, "dateFormat": "iso"}, **retry)


def parse_events(events: list, observed_at: str, odds_format: str = "decimal") -> pd.DataFrame:
    """Flatten provider events. With odds_format="american" the quoted American prices are
    kept (home_american/away_american) and home_price/away_price hold exact decimal conversions."""
    from .odds import american_to_decimal
    american = odds_format == "american"
    rows, unknown = [], set()
    for ev in events:
        home_name, away_name = ev.get("home_team"), ev.get("away_team")
        home_code, away_code = TEAM_CODES.get(home_name), TEAM_CODES.get(away_name)
        if home_code is None or away_code is None:
            unknown.update(n for n in (home_name, away_name) if n not in TEAM_CODES)
            continue
        for book in ev.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                if home_name not in prices or away_name not in prices:
                    continue
                row = {
                    "observed_at": observed_at, "event_id": ev["id"], "commence_time": ev["commence_time"],
                    "home_team": home_code, "away_team": away_code, "home_name": home_name,
                    "away_name": away_name, "book": book["key"],
                    "market_last_update": market.get("last_update", book.get("last_update")),
                }
                if american:
                    try:
                        h, a = float(prices[home_name]), float(prices[away_name])
                        row.update(home_american=h, away_american=a, home_price=float(american_to_decimal(h)),
                                   away_price=float(american_to_decimal(a)))
                    except (TypeError, ValueError):
                        row.update(home_american=prices[home_name], away_american=prices[away_name],
                                   home_price=float("nan"), away_price=float("nan"))
                else:
                    row.update(home_price=float(prices[home_name]), away_price=float(prices[away_name]))
                rows.append(row)
    if unknown:
        log.warning("skipped events with unrecognised team names: %s", sorted(unknown))
    cols = SNAPSHOT_COLUMNS + (["home_american", "away_american"] if american else [])
    return pd.DataFrame(rows, columns=cols)
