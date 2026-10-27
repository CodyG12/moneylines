"""Weekly performance report (Markdown) from the live prediction log. Reads stored records only:
scored pregame predictions, the prices recorded with them, the paper ledger and collection runs.
Small samples are labelled; nothing is extrapolated."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import clv, market, paper, store
from .evaluation import calibration_table, cluster_bootstrap_ratio, expected_calibration_error, full_score

MIN_N = 50  # below this a weekly number is mostly noise


def _f(v, fmt="{:.4f}"):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def _ci(d, fmt="{:+.4f}"):
    if not d or d.get("ci_low") is None or np.isnan(d.get("ci_low", np.nan)):
        return _f(d.get("estimate") if d else None, fmt)
    return f"{fmt.format(d['estimate'])} [{fmt.format(d['ci_low'])}, {fmt.format(d['ci_high'])}]"


def build(con: sqlite3.Connection, cfg: dict, end: pd.Timestamp, days: int = 7, reference: dict | None = None,
          n_boot: int = 1000) -> str:
    start = end - pd.Timedelta(days=days)
    in_week = lambda d: (pd.to_datetime(d) >= start.tz_localize(None)) & (pd.to_datetime(d) < end.tz_localize(None))  # noqa: E731
    L = [f"# Weekly performance report: {start.date()} to {(end - pd.Timedelta(days=1)).date()}",
         f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC from `{cfg['paths']['db']}`. Game dates are "
         "US/Eastern. Only predictions made before tip-off count. Weekly samples are small: treat every number as "
         f"preliminary below {MIN_N} games.", ""]

    # 1. prediction quality
    ev = store.official_evaluations(con)
    wk = ev[in_week(ev["game_date"])] if not ev.empty else ev
    L.append("## Prediction accuracy (published model, pregame only)")
    if wk.empty:
        L += ["No scored pregame predictions this week.", ""]
    else:
        for version, g in wk.groupby("model_version"):
            y, p = g["home_win"].to_numpy(float), g["p_home"].to_numpy(float)
            s = full_score(y, p, pd.to_datetime(g["game_date"]), n_boot)
            L += [f"Model `{version}`: {len(g)} games{' (PRELIMINARY)' if len(g) < MIN_N else ''}", "",
                  "| metric | this week [95% CI, needs 2+ game dates] | validated reference |", "|---|---|---|",
                  f"| accuracy | {_ci(s.get('accuracy_ci') or {'estimate': s['accuracy']}, '{:.3f}')} | - |",
                  f"| Brier score | {_ci(s.get('brier_ci') or {'estimate': s['brier']}, '{:.4f}')} | "
                  f"{_f((reference or {}).get('brier'))} |",
                  f"| log loss | {_ci(s.get('log_loss_ci') or {'estimate': s['log_loss']}, '{:.4f}')} | "
                  f"{_f((reference or {}).get('log_loss'))} |",
                  f"| calibration error (ECE) | {_f(expected_calibration_error(y, p))} | - |",
                  f"| mean predicted p(home) vs home win rate | {p.mean():.3f} vs {y.mean():.3f} | - |", ""]
            cal = calibration_table(y, p, n_bins=5)
            L += ["Calibration (5 bins; observed CI is a Wilson interval):", "",
                  "| predicted p(home) | games | mean predicted | observed home wins | observed 95% CI |",
                  "|---|---|---|---|---|"]
            L += [f"| {r.bin_low:.1f}-{r.bin_high:.1f} | {r.n} | {r.mean_predicted:.3f} | {r.observed_rate:.3f} | "
                  f"{r.observed_ci_low:.2f}-{r.observed_ci_high:.2f} |" for r in cal.itertuples()]
            L.append("")

    # 2. market benchmark
    book = cfg.get("paper", {}).get("book", cfg["paths"]["book"])
    L.append(f"## Model vs market ({book}, no-vig, prices recorded at each prediction's timestamp)")
    mf = market.live_comparison_frame(con, book)
    mf = mf[in_week(mf["game_date"])] if not mf.empty else mf
    if mf.empty:
        L += ["No scored predictions with a recorded sportsbook price this week.", ""]
    else:
        L += ["| horizon before tip | games | model | log loss | minus market [95% CI] | Brier |", "|---|---|---|---|---|---|"]
        for h in market.HORIZONS:
            r = market.benchmark(mf[mf["horizon"] == h], n_boot)
            if r.get("status") != "ok":
                continue
            L.append(f"| {h} | {r['n']} | market (no-vig) | {r['market']['log_loss']:.4f} | | {r['market']['brier']:.4f} |")
            for c, s in r["models"].items():
                L.append(f"| {h} | {r['n']} | {c} | {s['log_loss']:.4f} | {_ci(s['log_loss_minus_market'])} | "
                         f"{s['brier']:.4f} |")
        L += ["", "Negative 'minus market' means the model beat the market on the same games and timestamps.", ""]

    # 3. paper trading
    pcfg = paper.PaperConfig.from_dict(cfg.get("paper"))
    led = paper.ledger(con, pcfg.strategy)
    led = led[in_week(led["game_date"])] if not led.empty else led
    L.append(f"## Paper trading (SIMULATED, strategy `{pcfg.strategy}`, never real bets)")
    if led.empty:
        L += ["No paper decisions this week.", ""]
    else:
        counts = led["decision"].value_counts().to_dict()
        bets = led[led["decision"] == "bet"]
        settled = bets[bets["won"].notna()]
        L += [f"Decisions: {counts}", ""]
        if settled.empty:
            L += [f"{len(bets)} simulated bets, none settled yet.", ""]
        else:
            pf, sf = settled["pnl_flat"].to_numpy(float), settled["stake_flat"].to_numpy(float)
            roi = cluster_bootstrap_ratio(pf, sf, settled["game_date"], n_boot)
            eq = np.concatenate([[0.0], np.cumsum(pf)])
            L += ["| metric | value |", "|---|---|", f"| settled bets | {len(settled)} |",
                  f"| win rate vs break-even | {settled['won'].mean():.3f} vs {(1 / settled['decimal_odds']).mean():.3f} |",
                  f"| flat profit (units) | {pf.sum():+.2f} |", f"| flat ROI [95% CI] | {_ci(roi, '{:+.1%}')} |",
                  f"| flat max drawdown (units) | {float((np.maximum.accumulate(eq) - eq).max()):.2f} |",
                  f"| Kelly P&L | {settled['pnl_kelly'].sum():+.2f} |",
                  f"| mean estimated edge / EV | {bets['edge'].mean():+.3f} / {bets['est_ev'].mean():+.3f} |",
                  f"| mean price CLV (bets with a close) | {_f(bets['price_clv'].mean() if bets['price_clv'].notna().any() else None, '{:+.3f}')} |",
                  ""]
    c = clv.summary(con, pcfg.strategy)
    L += [f"Season-to-date CLV status counts: {c.get('status_counts', {})}. Positive CLV alone is not proof of "
          "profitability.", ""]

    # 4. data collection
    runs = pd.read_sql("SELECT * FROM odds_collection_runs", con)
    runs = runs[pd.to_datetime(runs["started_at"], utc=True) >= start] if not runs.empty else runs
    L.append("## Data collection")
    if runs.empty:
        L += ["No odds collection runs this week.", ""]
    else:
        rem = runs["credits_remaining"].dropna()
        L += [f"Collection runs: {runs['status'].value_counts().to_dict()}; snapshots stored "
              f"{int(runs['n_inserted'].fillna(0).sum())}; credits remaining "
              f"{int(rem.iloc[-1]) if len(rem) else 'unknown'}.", ""]
    al = pd.read_sql("SELECT created_at, kind, message FROM alerts", con)
    al = al[pd.to_datetime(al["created_at"], utc=True) >= start] if not al.empty else al
    if not al.empty:
        L += ["Alerts this week:", ""] + [f"- {r.created_at[:16]} {r.kind}: {r.message}" for r in
                                         al.drop_duplicates("message").itertuples()] + [""]
    return "\n".join(L)
