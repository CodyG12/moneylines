"""Pregame feature construction.

Every feature for a game on date D uses only games completed on dates < D.
Rest days, venue streaks and the like use the schedule (known in advance), not results.

Games are processed one *date* at a time: features for every game on D are read
from team state as of the start of D, and only then are D's results folded in.
So simultaneous games (same night) never see each other's outcomes.

Two feature lists:
  FEATURES      Phase 1 set, used by nba_model.py (unchanged)
  FEATURES_V2   Phase 2 set, used by the walk-forward models; see FEATURE_DEFINITIONS
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from .elo import EloConfig, compute_elo

FEATURES = [
    "elo_diff",
    "form_margin_diff",
    "form_winpct_diff",
    "rest_diff",
    "home_b2b",
    "away_b2b",
    "neutral_site_flag",
]

FEATURE_DEFINITIONS = {
    "elo_diff": "home minus away pregame Elo (home-court advantage excluded; see neutral_site_flag)",
    "neutral_site_flag": "1 when neither team is at home",
    "form5_margin_diff": "difference in mean point margin over each team's last 5 games this season, shrunk to 0",
    "form10_margin_diff": "same over the last 10 games",
    "form10_winpct_diff": "difference in win % over the last 10 games, shrunk to 0.5",
    "ewm_margin_diff": "difference in exponentially weighted margin (half-life 7 games) this season, shrunk",
    "adj_margin_diff": "season-to-date margin adjusted for opponent Elo and venue, shrunk",
    "wins_over_expected_diff": "season-to-date (wins - Elo-expected wins) per game, shrunk "
                               "(opponent-adjusted win %)",
    "sos_diff": "season-to-date mean opponent pregame Elo minus 1500, shrunk (strength of schedule)",
    "venue_margin_diff": "home team's season margin in home games minus away team's in road games, shrunk",
    "home_off_vs_away_def": "home offensive rating + away defensive rating, each relative to league average "
                            "(points per 100 possessions, season to date, shrunk)",
    "away_off_vs_home_def": "away offensive rating + home defensive rating, relative to league average",
    "net_rtg10_diff": "difference in net rating over the last 10 games with box scores, shrunk",
    "rest_diff": "home minus away days since previous game (capped at 7)",
    "home_b2b": "home team played the previous day",
    "away_b2b": "away team played the previous day",
    "home_streak": "consecutive home games for the home team, including this one",
    "away_streak": "consecutive road games for the away team, including this one",
    "h2h_resid": "head-to-head margin vs Elo expectation over this and last season's meetings, "
                 "home perspective, shrunk to 0 (prior of 4 games)",
    "min_season_games": "fewer of the two teams' games played this season (capped at 20); flags early season",
}
FEATURES_V2 = list(FEATURE_DEFINITIONS)

REST_CAP = 7  # days; also used for a team's first game of a season
ELO_PER_POINT = 28.0     # ~28 Elo points per point of expected margin
FORM_PRIOR = 3.0         # pseudo-games of zero margin added to rolling form
SEASON_PRIOR = 5.0       # pseudo-games for season-to-date aggregates
EWM_HALFLIFE = 7.0
EWM_PRIOR = 2.0
H2H_PRIOR = 4.0
POSS_PRIOR = 500.0       # pseudo-possessions at league-average efficiency
POSS10_PRIOR = 300.0
DEFAULT_RTG = 110.0      # only used before any box score exists in the data
LEAGUE_PRIOR_POSS = 20000.0
STREAK_CAP = 10
SEASON_GAMES_CAP = 20


def estimate_possessions(fga, oreb, tov, fta, opp_fga, opp_oreb, opp_tov, opp_fta):
    """Average of both teams' basic possession estimates (FGA - OREB + TOV + 0.44 FTA)."""
    own = fga - oreb + tov + 0.44 * fta
    opp = opp_fga - opp_oreb + opp_tov + 0.44 * opp_fta
    return 0.5 * (own + opp)


class _TeamState:
    __slots__ = ("season", "recent", "last_date", "n_games_total",
                 "last10", "ewm_num", "ewm_den", "season_n", "adj_sum", "wox_sum", "sos_sum",
                 "home_sum", "home_n", "road_sum", "road_n", "pts_for", "pts_against", "poss", "eff_n",
                 "eff10", "last_loc", "streak")

    def __init__(self, window: int):
        self.season = None
        self.recent = deque(maxlen=window)
        self.last_date = None
        self.n_games_total = 0
        self.last10 = deque(maxlen=10)
        self.eff10 = deque(maxlen=10)
        self.last_loc = None
        self.streak = 0
        self.reset_season()

    def reset_season(self):
        self.recent.clear()
        self.last10.clear()
        self.eff10.clear()
        self.ewm_num = self.ewm_den = 0.0
        self.season_n = 0
        self.adj_sum = self.wox_sum = self.sos_sum = 0.0
        self.home_sum = self.road_sum = 0.0
        self.home_n = self.road_n = 0
        self.pts_for = self.pts_against = self.poss = 0.0
        self.eff_n = 0
        self.last_loc = None
        self.streak = 0


class _League:
    """Season-to-date league efficiency, used as the shrinkage target for team ratings."""

    def __init__(self):
        self.season = None
        self.pts = self.poss = 0.0
        self.previous = DEFAULT_RTG

    def start(self, season):
        if season != self.season:
            if self.poss > 0:
                self.previous = 100.0 * self.pts / self.poss
            self.season, self.pts, self.poss = season, 0.0, 0.0

    @property
    def rtg(self):
        # Blend with last season's level (~100 games' worth) so opening nights aren't noisy.
        return 100.0 * (self.pts + LEAGUE_PRIOR_POSS * self.previous / 100.0) / (self.poss + LEAGUE_PRIOR_POSS)


def _shrunk(total, n, prior, centre=0.0):
    return (total + prior * centre) / (n + prior)


def build_features(games: pd.DataFrame, elo_config: EloConfig = EloConfig(), window: int = 10) -> pd.DataFrame:
    if window < 1:
        raise ValueError("window must be >= 1")
    elo, _ = compute_elo(games, elo_config)
    out = games.join(elo)

    n = len(games)
    names = ["home_form_margin", "away_form_margin", "home_form_winpct", "away_form_winpct",
             "home_form_n", "away_form_n", "home_days_rest", "away_days_rest", "home_history", "away_history"]
    for side in ("home", "away"):
        names += [f"{side}_{k}" for k in (
            "form5_margin", "form10_margin", "form10_winpct", "ewm_margin", "adj_margin", "wox", "sos",
            "venue_margin", "ortg", "drtg", "net_rtg10", "eff_n", "season_games", "streak", "last_game_date")]
    cols = {name: np.full(n, np.nan) for name in names}
    league_rtg = np.full(n, np.nan)
    h2h_resid = np.zeros(n)
    h2h_n = np.zeros(n)

    states: dict = {}
    league = _League()
    h2h: dict = {}
    home = games["home_team"].to_numpy()
    away = games["away_team"].to_numpy()
    season = games["season"].to_numpy()
    played = games["played"].to_numpy()
    neutral = games["neutral_site"].to_numpy().astype(bool)
    hs = games["home_score"].to_numpy(dtype=float)
    as_ = games["away_score"].to_numpy(dtype=float)
    margin = hs - as_
    home_pre = elo["home_elo_pre"].to_numpy()
    away_pre = elo["away_elo_pre"].to_numpy()
    elo_p = elo["elo_home_prob"].to_numpy()
    hca_points = np.where(neutral, 0.0, elo_config.home_advantage / ELO_PER_POINT)
    expected_margin = (home_pre - away_pre) / ELO_PER_POINT + hca_points

    box_cols = ["fga", "oreb", "tov", "fta"]
    if all(f"{s}_{c}" in games.columns for s in ("home", "away") for c in box_cols):
        b = {f"{s}_{c}": games[f"{s}_{c}"].to_numpy(dtype=float) for s in ("home", "away") for c in box_cols}
        poss = estimate_possessions(b["home_fga"], b["home_oreb"], b["home_tov"], b["home_fta"],
                                    b["away_fga"], b["away_oreb"], b["away_tov"], b["away_fta"])
    else:
        poss = np.full(n, np.nan)
    decay = 0.5 ** (1.0 / EWM_HALFLIFE)

    groups = games.groupby("date", sort=True).indices
    for date in sorted(groups):
        idx = groups[date]
        ts = pd.Timestamp(date)
        # 1) Read features from state as of the start of this date.
        for i in idx:
            league.start(season[i])
            league_rtg[i] = league.rtg
            for side, team in (("home", home[i]), ("away", away[i])):
                st = states.setdefault(team, _TeamState(window))
                new_season = st.season != season[i]
                if new_season:
                    st.reset_season()
                    st.season = season[i]
                if st.recent:
                    margins = [m for m, _ in st.recent]
                    wins = [w for _, w in st.recent]
                    cols[f"{side}_form_margin"][i] = float(np.mean(margins))
                    cols[f"{side}_form_winpct"][i] = float(np.mean(wins))
                else:  # no in-season history yet: neutral values, flagged via *_form_n
                    cols[f"{side}_form_margin"][i] = 0.0
                    cols[f"{side}_form_winpct"][i] = 0.5
                cols[f"{side}_form_n"][i] = len(st.recent)
                if st.last_date is None or new_season:
                    rest = REST_CAP
                else:
                    rest = min((ts - st.last_date).days, REST_CAP)
                cols[f"{side}_days_rest"][i] = rest
                cols[f"{side}_history"][i] = st.n_games_total

                l10 = list(st.last10)
                l5 = l10[-5:]
                cols[f"{side}_form5_margin"][i] = _shrunk(sum(m for m, _ in l5), len(l5), FORM_PRIOR)
                cols[f"{side}_form10_margin"][i] = _shrunk(sum(m for m, _ in l10), len(l10), FORM_PRIOR)
                cols[f"{side}_form10_winpct"][i] = _shrunk(sum(w for _, w in l10), len(l10), FORM_PRIOR, 0.5)
                cols[f"{side}_ewm_margin"][i] = st.ewm_num / (st.ewm_den + EWM_PRIOR)
                cols[f"{side}_adj_margin"][i] = _shrunk(st.adj_sum, st.season_n, SEASON_PRIOR)
                cols[f"{side}_wox"][i] = _shrunk(st.wox_sum, st.season_n, SEASON_PRIOR)
                cols[f"{side}_sos"][i] = _shrunk(st.sos_sum, st.season_n, SEASON_PRIOR)
                if side == "home":
                    cols["home_venue_margin"][i] = _shrunk(st.home_sum, st.home_n, SEASON_PRIOR)
                else:
                    cols["away_venue_margin"][i] = _shrunk(st.road_sum, st.road_n, SEASON_PRIOR)
                lg = league.rtg
                cols[f"{side}_ortg"][i] = 100.0 * (st.pts_for + POSS_PRIOR * lg / 100.0) / (st.poss + POSS_PRIOR)
                cols[f"{side}_drtg"][i] = 100.0 * (st.pts_against + POSS_PRIOR * lg / 100.0) / (st.poss + POSS_PRIOR)
                e_pf = sum(e[0] for e in st.eff10)
                e_pa = sum(e[1] for e in st.eff10)
                e_poss = sum(e[2] for e in st.eff10)
                cols[f"{side}_net_rtg10"][i] = 100.0 * (e_pf - e_pa) / (e_poss + POSS10_PRIOR)
                cols[f"{side}_eff_n"][i] = st.eff_n
                cols[f"{side}_season_games"][i] = st.season_n
                loc = "H" if (side == "home" and not neutral[i]) else "A"
                cols[f"{side}_streak"][i] = min(st.streak + 1 if st.last_loc == loc else 1, STREAK_CAP)
                cols[f"{side}_last_game_date"][i] = st.last_date.value if st.last_date is not None else np.nan

            key = (home[i], away[i]) if home[i] < away[i] else (away[i], home[i])
            sign = 1.0 if key[0] == home[i] else -1.0
            meetings = [r for s, r in h2h.get(key, ()) if s >= season[i] - 1]
            h2h_resid[i] = sign * sum(meetings) / (len(meetings) + H2H_PRIOR)
            h2h_n[i] = len(meetings)

        # 2) Only then fold in this date's results (and schedule-based state).
        for i in idx:
            for side, team, sgn, opp_pre in (("home", home[i], 1.0, away_pre[i]), ("away", away[i], -1.0, home_pre[i])):
                st = states[team]
                st.last_date = ts
                loc = "H" if (side == "home" and not neutral[i]) else "A"
                st.streak = int(cols[f"{side}_streak"][i])
                st.last_loc = loc
                if not played[i]:
                    continue
                m = sgn * margin[i]
                won = 1.0 if m > 0 else 0.0
                st.recent.append((m, won))
                st.n_games_total += 1
                st.last10.append((m, won))
                st.ewm_num = decay * st.ewm_num + m
                st.ewm_den = decay * st.ewm_den + 1.0
                venue = 0.0 if neutral[i] else (elo_config.home_advantage / ELO_PER_POINT) * sgn
                st.adj_sum += m + (opp_pre - 1500.0) / ELO_PER_POINT - venue
                st.wox_sum += won - (elo_p[i] if side == "home" else 1.0 - elo_p[i])
                st.sos_sum += opp_pre - 1500.0
                st.season_n += 1
                if loc == "H":
                    st.home_sum += m
                    st.home_n += 1
                elif not neutral[i]:
                    st.road_sum += m
                    st.road_n += 1
                if np.isfinite(poss[i]) and poss[i] > 0:
                    pf, pa = (hs[i], as_[i]) if side == "home" else (as_[i], hs[i])
                    st.pts_for += pf
                    st.pts_against += pa
                    st.poss += poss[i]
                    st.eff_n += 1
                    st.eff10.append((pf, pa, poss[i]))
            if played[i]:
                if np.isfinite(poss[i]) and poss[i] > 0:
                    league.pts += hs[i] + as_[i]
                    league.poss += 2 * poss[i]
                key = (home[i], away[i]) if home[i] < away[i] else (away[i], home[i])
                resid = margin[i] - expected_margin[i]
                h2h.setdefault(key, []).append((season[i], resid if key[0] == home[i] else -resid))

    for name, values in cols.items():
        out[name] = values
    for side in ("home", "away"):
        out[f"{side}_last_game_date"] = pd.to_datetime(out[f"{side}_last_game_date"])
    out["league_rtg"] = league_rtg
    out["h2h_n"] = h2h_n

    out["elo_diff"] = out["home_elo_pre"] - out["away_elo_pre"]
    out["form_margin_diff"] = out["home_form_margin"] - out["away_form_margin"]
    out["form_winpct_diff"] = out["home_form_winpct"] - out["away_form_winpct"]
    out["rest_diff"] = out["home_days_rest"] - out["away_days_rest"]
    out["home_b2b"] = (out["home_days_rest"] == 1).astype(float)
    out["away_b2b"] = (out["away_days_rest"] == 1).astype(float)
    out["neutral_site_flag"] = out["neutral_site"].astype(float)

    for k in ("form5_margin", "form10_margin", "form10_winpct", "ewm_margin", "adj_margin", "sos"):
        out[f"{k}_diff"] = out[f"home_{k}"] - out[f"away_{k}"]
    out["wins_over_expected_diff"] = out["home_wox"] - out["away_wox"]
    out["venue_margin_diff"] = out["home_venue_margin"] - out["away_venue_margin"]
    lg = out["league_rtg"]
    out["home_off_vs_away_def"] = (out["home_ortg"] - lg) + (out["away_drtg"] - lg)
    out["away_off_vs_home_def"] = (out["away_ortg"] - lg) + (out["home_drtg"] - lg)
    out["net_rtg10_diff"] = out["home_net_rtg10"] - out["away_net_rtg10"]
    out["h2h_resid"] = h2h_resid
    out["min_season_games"] = np.minimum(np.minimum(out["home_season_games"], out["away_season_games"]),
                                         SEASON_GAMES_CAP)
    return out
