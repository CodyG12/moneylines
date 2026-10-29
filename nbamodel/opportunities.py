"""Risk-adjusted ranking of today's moneyline prices, with an explicit no-bet outcome.

For each game: take the side with the higher expected value at the latest eligible price, then
  expected ROI      = p x decimal_odds - 1
  conservative ROI  = p_lower x decimal_odds - 1
where p_lower is p minus the model's historical calibration shortfall for similar predictions
(`fit_uncertainty`): in each predicted-probability band (separately for early-season games), the
gap between the mean prediction and an 80% one-sided lower bound on the observed win rate,
measured on out-of-sample walk-forward predictions from the development period (never the final
holdout). Bands with too few games fall back to the all-games band. The sportsbook price is
used only to compute ROI and the no-vig comparison; it is never a model input, so it is not
counted twice as evidence.

A game is eligible only if every check passes (`OpportunityConfig`); eligible games are ranked
by conservative ROI. Everything else is listed with every reason it was rejected. Nothing here
places a bet: selections are for the paper-trading ledger and for information only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .odds import kelly_fraction

LOWER_Z = 0.8416      # one-sided 80% bound
DEFAULT_BINS = (0.0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0)


@dataclass(frozen=True)
class OpportunityConfig:
    min_edge: float = 0.02                  # model minus no-vig probability for the selected side
    min_expected_roi: float = 0.02          # same threshold as the paper ledger's ev_threshold
    min_conservative_roi: float = 0.0       # p_lower x odds - 1 must not be negative
    max_abs_edge: float = 0.10              # the paper ledger's edge_too_large abstention, kept
    max_odds_age_minutes: float = 60.0
    max_injury_report_age_minutes: float = 180.0
    max_model_spread: float = 0.20          # disagreement between the model's component probabilities
    early_season_games: int = 10            # either team below this many games: early season
    early_season_extra_roi: float = 0.02    # early games need conservative ROI >= min + this
    rotation_minutes: float = 20.0          # a player is "important" from this many healthy minutes
    block_unresolved_injuries: bool = True  # a rotation player questionable/doubtful -> not eligible
    block_missing_injury_report: bool = True
    kelly_multiplier: float = 0.25          # paper stake: fraction of Kelly, sized on p_lower
    max_stake_fraction: float = 0.02        # of bankroll, per selection
    max_daily_exposure: float = 0.10        # of bankroll, all selections in a day
    max_selections_per_team: int = 1        # correlated exposure: one selection per team per day
    validated_edge: bool = False            # set true only once live CLV/market results support an edge

    @classmethod
    def from_dict(cls, d: dict | None) -> "OpportunityConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in allowed})


# ------------------------------------------------------------------ uncertainty
def fit_uncertainty(preds: pd.DataFrame, p_col: str, early_games: int = 10, bins=DEFAULT_BINS,
                    min_n: int = 100) -> dict:
    """Calibration shortfall table from out-of-sample predictions (columns p_col, home_win,
    home_season_games, away_season_games). Each game counts once per side."""
    d = preds.dropna(subset=[p_col, "home_win"])
    p = pd.concat([d[p_col], 1 - d[p_col]], ignore_index=True)
    y = pd.concat([d["home_win"], 1 - d["home_win"]], ignore_index=True).astype(float)
    early = pd.concat([d[["home_season_games", "away_season_games"]].min(axis=1) < early_games] * 2,
                      ignore_index=True)
    table = {"p_col": p_col, "early_games": early_games, "bins": list(bins), "lower_z": LOWER_Z,
             "min_n": min_n, "n_games": int(len(d)), "date_range": [str(d["date"].min()), str(d["date"].max())]
             if "date" in d else None, "groups": {}}
    for name, mask in (("all", pd.Series(True, index=p.index)), ("early", early), ("later", ~early)):
        cut = pd.cut(p[mask], list(bins), include_lowest=True)
        g = pd.DataFrame({"p": p[mask], "y": y[mask], "bin": cut}).groupby("bin", observed=False)
        rows = []
        for iv, grp in g:
            n = len(grp)
            if n == 0:
                rows.append({"lo": iv.left, "hi": iv.right, "n": 0, "shortfall": None})
                continue
            obs = grp["y"].mean()
            lower = obs - LOWER_Z * np.sqrt(max(obs * (1 - obs), 1e-9) / n)
            rows.append({"lo": float(iv.left), "hi": float(iv.right), "n": int(n), "pred": float(grp["p"].mean()),
                         "obs": float(obs), "shortfall": float(grp["p"].mean() - lower)})
        table["groups"][name] = rows
    return table


def save_uncertainty(table: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(table, indent=1))


def load_uncertainty(path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def shortfall(table: dict, p: float, early: bool) -> float:
    """Calibration shortfall for a side probability `p` (never negative: a band where the model was
    historically under-confident does not raise p)."""
    def lookup(group):
        for b in table["groups"][group]:
            if b["lo"] <= p <= b["hi"] and b["n"] >= table["min_n"] and b["shortfall"] is not None:
                return b["shortfall"]
        return None
    s = lookup("early" if early else "later")
    if s is None:
        s = lookup("all")
    return max(0.0, s if s is not None else 0.0)


# ------------------------------------------------------------------ ranking
REASONS = {
    "no_odds": "no sportsbook price",
    "stale_odds": "stale sportsbook odds",
    "negative_roi": "negative expected ROI",
    "low_roi": "expected ROI below minimum",
    "low_edge": "edge below minimum",
    "low_conservative_roi": "conservative ROI below minimum",
    "early_season": "early-season uncertainty (conservative ROI below the early-season minimum)",
    "edge_too_large": "large model-market disagreement (edge above the cap; likely missing information)",
    "model_disagreement": "model components disagree",
    "unresolved_injury": "unresolved injury uncertainty (rotation player questionable/doubtful)",
    "missing_injury": "missing injury information",
    "stale_injury": "injury report too old",
    "team_exposure": "correlated exposure (team already selected today)",
    "daily_exposure": "daily exposure cap reached",
}


def rank(preds: pd.DataFrame, cfg: OpportunityConfig, table: dict | None, now: pd.Timestamp,
         injury: pd.DataFrame | None = None, bankroll: float = 100.0, used_exposure: float = 0.0) -> pd.DataFrame:
    """One row per game: selected side and its numbers, eligibility, every rejection reason, and a
    paper stake for eligible games (ranked by conservative ROI).

    preds: today's prediction rows (p_home, home_odds/away_odds decimal, market_home_prob_novig,
    odds_timestamp, model_spread, home_/away_season_games, home_team, away_team, game_key).
    injury: per game_key — report_age_minutes, n_rotation_uncertain, report_missing (bool),
    out_players, uncertain_players (display strings). None when injury data is unavailable."""
    rows = []
    inj = injury.set_index("game_key") if injury is not None and not injury.empty else None
    for r in preds.to_dict("records"):
        out = {"game_key": r["game_key"], "home_team": r["home_team"], "away_team": r["away_team"],
               "tipoff_utc": r.get("tipoff_utc"), "reasons": []}
        early = min(r.get("home_season_games", 99), r.get("away_season_games", 99)) < cfg.early_season_games
        out["early_season"] = bool(early)
        if pd.isna(r.get("home_odds")) or pd.isna(r.get("away_odds")):
            p_side = max(r["p_home"], 1 - r["p_home"])
            out.update(side=None, selection=r["home_team"] if r["p_home"] >= 0.5 else r["away_team"], p=p_side,
                       p_lower=np.nan, decimal_odds=np.nan, market_p=np.nan, edge=np.nan, roi=np.nan,
                       conservative_roi=np.nan)
            out["reasons"].append("no_odds")
            rows.append(out)
            continue
        h, a, ph = float(r["home_odds"]), float(r["away_odds"]), float(r["p_home"])
        home = ph * h >= (1 - ph) * a
        p = ph if home else 1 - ph
        dec = h if home else a
        nv = float(r["market_home_prob_novig"]) if home else 1 - float(r["market_home_prob_novig"])
        sf = shortfall(table, p, early) if table else 0.0
        pl = max(p - sf, 0.0)
        out.update(side="home" if home else "away", selection=r["home_team"] if home else r["away_team"], p=p,
                   p_lower=pl, shortfall=sf, decimal_odds=dec, market_p=nv, edge=p - nv, roi=p * dec - 1,
                   conservative_roi=pl * dec - 1)
        age = (now - pd.Timestamp(r["odds_timestamp"])).total_seconds() / 60 if pd.notna(r.get("odds_timestamp")) else None
        out["odds_age_minutes"] = age
        reasons = out["reasons"]
        if age is None or age > cfg.max_odds_age_minutes:
            reasons.append("stale_odds")
        if out["roi"] < 0:
            reasons.append("negative_roi")
        elif out["roi"] < cfg.min_expected_roi:
            reasons.append("low_roi")
        if out["edge"] < cfg.min_edge:
            reasons.append("low_edge")
        if abs(out["edge"]) > cfg.max_abs_edge:
            reasons.append("edge_too_large")
        if out["conservative_roi"] < cfg.min_conservative_roi:
            reasons.append("low_conservative_roi")
        elif early and out["conservative_roi"] < cfg.min_conservative_roi + cfg.early_season_extra_roi:
            reasons.append("early_season")
        if pd.notna(r.get("model_spread")) and r["model_spread"] > cfg.max_model_spread:
            reasons.append("model_disagreement")
        if inj is not None and r["game_key"] in inj.index:
            g = inj.loc[r["game_key"]]
            out.update(out_players=g.get("out_players", ""), uncertain_players=g.get("uncertain_players", ""),
                       injury_report_age_minutes=g.get("report_age_minutes"))
            if g.get("report_missing") and cfg.block_missing_injury_report:
                reasons.append("missing_injury")
            elif pd.notna(g.get("report_age_minutes")) and g["report_age_minutes"] > cfg.max_injury_report_age_minutes:
                reasons.append("stale_injury")
            if g.get("n_rotation_uncertain", 0) > 0 and cfg.block_unresolved_injuries:
                reasons.append("unresolved_injury")
        elif cfg.block_missing_injury_report:
            reasons.append("missing_injury")
        rows.append(out)
    df = pd.DataFrame(rows)
    df["stake_fraction"] = 0.0
    ok = df["reasons"].map(len) == 0
    order = df[ok].sort_values(["conservative_roi", "roi"], ascending=False).index
    teams, room = {}, cfg.max_daily_exposure * bankroll - used_exposure
    for i in order:
        r = df.loc[i]
        if any(teams.get(t, 0) >= cfg.max_selections_per_team for t in (r["home_team"], r["away_team"])):
            df.at[i, "reasons"] = ["team_exposure"]
            continue
        f = min(float(kelly_fraction(r["p_lower"], r["decimal_odds"])) * cfg.kelly_multiplier, cfg.max_stake_fraction)
        stake = min(max(f, 0.0) * bankroll, max(room, 0.0))
        if stake <= 1e-9:
            df.at[i, "reasons"] = ["daily_exposure"]
            continue
        df.at[i, "stake_fraction"] = stake / bankroll
        room -= stake
        for t in (r["home_team"], r["away_team"]):
            teams[t] = teams.get(t, 0) + 1
    df["eligible"] = df["reasons"].map(len) == 0
    df["rank"] = np.nan
    elig = df[df["eligible"]].sort_values(["conservative_roi", "roi"], ascending=False).index
    df.loc[elig, "rank"] = range(1, len(elig) + 1)
    df["reason_text"] = df["reasons"].map(lambda rs: "; ".join(REASONS[x] for x in rs))
    return df


# ------------------------------------------------------------------ injury flags (risk controls)
def injury_flags(preds: pd.DataFrame, statuses: pd.DataFrame, coverage: dict, minutes: pd.DataFrame,
                 cfg: OpportunityConfig) -> pd.DataFrame:
    """Per game: report age, whether information is missing (no report yet, or a team has not
    submitted), rotation players out, and rotation players whose status is unresolved.

    statuses: injuries.statuses_asof rows for today; coverage: {game_date: report_coverage(...)};
    minutes: player_impact.minutes_asof (healthy minutes identify rotation players)."""
    no_minutes = minutes.empty            # cannot tell who is important: treat as missing information
    rotation = minutes[minutes["healthy"] >= cfg.rotation_minutes]
    rot_keys = set(zip(rotation["team"], rotation["player_key"]))
    rows = []
    for r in preds.to_dict("records"):
        day = str(pd.Timestamp(r["date"]).date())
        cov = coverage.get(day) or {}
        matchup = f"{r['away_team']}@{r['home_team']}"
        st = statuses[(statuses["game_date"] == day) & (statuses["matchup"] == matchup)] if len(statuses) else statuses
        pending = sorted(set(cov.get("not_submitted", [])) & {r["home_team"], r["away_team"]})
        rot = st[[k in rot_keys for k in zip(st["team"], st["player_key"])]] if len(st) else st
        out = rot[rot["status"].isin(["out"])] if len(rot) else rot
        unsure = rot[rot["status"].isin(["questionable", "doubtful"])] if len(rot) else rot
        fmt = lambda d: ", ".join(f"{n} ({t}{'' if s == 'out' else ', ' + s})"   # noqa: E731
                                  for n, t, s in zip(d["player_name"], d["team"], d["status"]))
        rows.append({"game_key": r["game_key"], "report_age_minutes": cov.get("report_age_minutes"),
                     "report_missing": cov.get("report_published_at") is None or bool(pending) or no_minutes,
                     "not_submitted": ", ".join(pending), "n_rotation_out": len(out),
                     "n_rotation_uncertain": len(unsure), "out_players": fmt(out) if len(out) else "",
                     "uncertain_players": fmt(unsure) if len(unsure) else ""})
    return pd.DataFrame(rows)
