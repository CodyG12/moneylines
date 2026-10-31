"""Does knowing the injury report improve the model? Walk-forward comparison on development games.

Variants (phase 4 spec, section 4), each fitted walk-forward with the settings in config:
  A_logistic        current champion: logistic regression, uncalibrated (the base of every comparison)
  A_logistic_platt, A_gbm, A_elo, A_blend    the same model families without injury features
  B_logistic        A + AVAILABILITY_FEATURES, uncalibrated
  C_platt           B with Platt calibration on the walk-forward calibration window
  C_isotonic        B with isotonic calibration on the same window
  D_blend           early-only Elo blend of B; its weight is chosen on earlier seasons only
  E_gbm, E_gbm_platt    gradient boosting with the injury features, raw and calibrated

Every variant is scored on the same games: played development games where both teams had an
official report before the prediction time. Injury models train on all earlier games; before the
first report (2019-11-20) the injury columns are NaN and carry no information. Reports only reach
the fit window after the 40-date calibration hold-back, so the first months of 2019-20 are close to
the baseline by construction; seasons are also reported separately.

Decision rule, fixed before any result was seen: adopt the injury variant with the lowest log loss
only if (a) its paired log-loss difference vs A_logistic has a 95% CI entirely below 0 (date-cluster
bootstrap) and (b) it is not worse than A_logistic in the early-season segment. Six variants are
compared on one set of games, so the winner's estimate is optimistic. The final holdout is never
used. No historical prices exist, so there is no market benchmark on these games.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import blend
from .data import DataError
from .dataset import AVAILABILITY_FEATURES
from .evaluation import compare_brier, compare_log_loss, full_score, score
from .features import FEATURES_V2
from .leakage import suspicious_label_correlation
from .walkforward import WalkForwardConfig, run_walk_forward

BASE = "A_logistic"
CANDIDATES = ["B_logistic", "C_platt", "C_isotonic", "D_blend", "E_gbm", "E_gbm_platt"]
VARIANTS = {"A_logistic": "A logistic, uncalibrated (champion)", "A_logistic_platt": "A logistic, Platt",
            "A_gbm": "A gradient boosting", "A_elo": "A Elo", "A_blend": "A early-only Elo blend",
            "B_logistic": "B +injury, uncalibrated", "C_platt": "C +injury, Platt", "C_isotonic": "C +injury, isotonic",
            "D_blend": "D +injury, early-only Elo blend", "E_gbm": "E gradient boosting +injury",
            "E_gbm_platt": "E gradient boosting +injury, Platt"}
TWIN = {"B_logistic": "A_logistic", "C_platt": "A_logistic_platt", "D_blend": "A_blend", "E_gbm": "A_gbm"}
EARLY = "early (<15 games)"
DECISION_RULE = ("adopt the lowest-log-loss injury variant only if its log-loss difference vs A_logistic has a "
                 "95% CI entirely below 0 and it is not worse in the early-season segment")


def decide(report: dict, candidates: list = CANDIDATES) -> dict:
    best = min(candidates, key=lambda c: report["overall"][c]["log_loss"])
    d = report["overall"][best]["log_loss_diff_vs_base"]
    early = report["by_segment"].get(EARLY, {}).get(best, {}).get("log_loss_diff_vs_base", {}).get("estimate")
    adopt = bool(d["ci_high"] < 0 and (early is None or early <= 0))
    return {"best": best, "adopt": adopt, "rule": DECISION_RULE, "log_loss_diff_vs_base": d, "early_season_diff": early,
            "multiple_testing_note": f"{len(candidates)} variants compared on the same games; the best one's "
                                     "improvement is optimistic by selection"}


def _diffs(d: pd.DataFrame, c: str, n_boot: int, seed: int) -> dict:
    y = d["home_win"].to_numpy(float)
    res = {}
    if c != BASE:
        res["log_loss_diff_vs_base"] = compare_log_loss(y, d[c], d[BASE], d["date"], n_boot, seed)
        res["brier_diff_vs_base"] = compare_brier(y, d[c], d[BASE], d["date"], n_boot, seed)
    if c in TWIN:
        res["log_loss_diff_vs_twin"] = compare_log_loss(y, d[c], d[TWIN[c]], d["date"], n_boot, seed)
    return res


def _block(d: pd.DataFrame, n_boot: int, seed: int) -> dict:
    y = d["home_win"].to_numpy(float)
    return {c: {**score(y, d[c]), **_diffs(d, c, n_boot, seed)} for c in VARIANTS}


def _margin_slope(d: pd.DataFrame) -> dict:
    """Points of actual margin per point of predicted injury impact, holding team strength fixed.
    1.0 means the feature is on the right scale; about 0 means it carries nothing the Elo gap lacks."""
    m = (d["home_score"] - d["away_score"]).to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d["injury_impact_diff"], d["elo_diff"]])
    beta, *_ = np.linalg.lstsq(X, m, rcond=None)
    resid = m - X @ beta
    se = np.sqrt(np.diag(resid @ resid / (len(m) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    return {"slope": float(beta[1]), "se": float(se[1])}


def _coverage(ds: pd.DataFrame, cfg: WalkForwardConfig) -> list:
    dev = ds[ds["played"] & (ds["date"] < pd.Timestamp(cfg.final_test_start))]
    dev = dev[dev["season"] >= ds[ds["report_available"] == 1]["season"].min()]
    t = dev.groupby("season").agg(games=("game_key", "size"), covered=("report_available", "sum"),
                                  first_date=("date", "min"), last_date=("date", "max")).reset_index()
    t["first_date"], t["last_date"] = t["first_date"].astype(str).str[:10], t["last_date"].astype(str).str[:10]
    return t.to_dict("records")


def run(games: pd.DataFrame, ds: pd.DataFrame, cfg: WalkForwardConfig, n_boot: int = 1000, seed: int = 0) -> tuple:
    """(report, scored predictions). `ds` must already have the availability columns attached."""
    if ds["report_available"].sum() == 0:
        raise DataError("no game has availability features: attach them to the dataset first")
    with_injury = list(FEATURES_V2) + AVAILABILITY_FEATURES
    base = run_walk_forward(games, replace(cfg, features=list(FEATURES_V2), models=("elo", "logistic", "gbm")), dataset=ds)
    inj = run_walk_forward(games, replace(cfg, features=with_injury, models=("logistic", "gbm"), calibration="platt"),
                           dataset=ds)
    iso = run_walk_forward(games, replace(cfg, features=with_injury, models=("logistic",), calibration="isotonic"),
                           dataset=ds)
    p = base.predictions.rename(columns={"p_logistic_raw": "A_logistic", "p_logistic": "A_logistic_platt",
                                         "p_gbm_raw": "A_gbm", "p_elo": "A_elo"})
    p = p.drop(columns=["p_gbm", "p_ensemble", "p_home_base_rate"], errors="ignore")
    p["date"] = pd.to_datetime(p["date"])
    p = p.merge(inj.predictions[["game_key", "p_logistic_raw", "p_logistic", "p_gbm_raw", "p_gbm"]].rename(
        columns={"p_logistic_raw": "B_logistic", "p_logistic": "C_platt", "p_gbm_raw": "E_gbm", "p_gbm": "E_gbm_platt"}),
        on="game_key", validate="one_to_one")
    p = p.merge(iso.predictions[["game_key", "p_logistic"]].rename(columns={"p_logistic": "C_isotonic"}),
                on="game_key", validate="one_to_one")
    # Blend weights are fitted on earlier seasons of the same out-of-sample predictions.
    for col, name in (("A_logistic", "A_blend"), ("B_logistic", "D_blend")):
        chained, _ = blend.forward_chain(p.assign(p_elo=p["A_elo"]), col, methods=("early_only",))
        p = p.merge(chained[["game_key", "p_blend_early_only"]].rename(columns={"p_blend_early_only": name}),
                    on="game_key", how="left", validate="one_to_one")
    extra = [c for c in ds.columns if c in AVAILABILITY_FEATURES or c in ("elo_diff", "home_unmatched", "away_unmatched")]
    p = p.merge(ds[["game_key"] + extra], on="game_key", validate="one_to_one")
    covered = p[p["report_available"] == 1].dropna(subset=list(VARIANTS)).copy()  # the blends skip the first season
    if covered.empty:
        raise DataError("no development game has both teams' reports before the prediction time")
    if (covered["date"] >= pd.Timestamp(cfg.final_test_start)).any():
        raise DataError("availability study must not score final-holdout games")
    suspicious = suspicious_label_correlation(covered, [c for c in AVAILABILITY_FEATURES if c != "report_available"])
    if suspicious:
        raise DataError(f"availability feature almost equals the label: {suspicious}")

    y = covered["home_win"].to_numpy(float)
    overall = {c: {**full_score(y, covered[c], covered["date"], n_boot, seed), **_diffs(covered, c, n_boot, seed)}
               for c in VARIANTS}
    seg = blend.segment_of(covered)
    heavy_cut = float(covered["injury_impact_diff"].abs().quantile(0.75))
    unsure_cut = float(covered["injury_impact_sd"].quantile(0.75))
    report = {
        "n": int(len(covered)), "dates": int(covered["date"].nunique()),
        "first_date": str(covered["date"].min().date()), "last_date": str(covered["date"].max().date()),
        "coverage": _coverage(ds, cfg), "overall": overall,
        "by_segment": {name: _block(covered[seg == name], n_boot, seed) | {"n": int((seg == name).sum())}
                       for name in blend.SEGMENTS if (seg == name).any()},
        "by_season": {str(s): _block(g, n_boot, seed) | {"n": int(len(g))} for s, g in covered.groupby("season")},
        "injury_heavy": {"abs_impact_cut": heavy_cut, "n": int((covered["injury_impact_diff"].abs() >= heavy_cut).sum()),
                         **_block(covered[covered["injury_impact_diff"].abs() >= heavy_cut], n_boot, seed)},
        "unresolved_uncertainty": {"sd_cut": unsure_cut, "n": int((covered["injury_impact_sd"] >= unsure_cut).sum()),
                                   **_block(covered[covered["injury_impact_sd"] >= unsure_cut], n_boot, seed)},
        "feature_diagnostics": {
            "margin_per_impact_point": _margin_slope(covered),
            "impact_diff_sd": float(covered["injury_impact_diff"].std()),
            "logistic_coefficients": inj.importance["logistic_standardized_coefficients"],
            "gbm_permutation": inj.importance.get("gbm_permutation_log_loss_increase", {}),
        },
        "market": {"n_priced": int(covered["home_odds"].notna().sum()) if "home_odds" in covered else 0,
                   "note": "no historical prices (free data only); market comparison comes from live-collected odds"},
        "config": {**inj.config, "features": with_injury},
    }
    report["decision"] = decide(report)
    return report, covered
