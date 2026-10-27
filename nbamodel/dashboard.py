"""Static, self-contained HTML dashboard (no server, no external assets).

Reads the model registry, the prediction database and the walk-forward / final-test
report directories, and writes one HTML file with charts embedded as PNG data URIs.
Every simulated result is labelled as such.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd

from . import clv, market, paper, plots, registry, store
from .report import MODEL_LABELS, fmt_ci

CSS = """
:root { --bg:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; --accent:#2a78d6;
        --warn-bg:#fff4e0; --warn-ink:#7a4a00; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#1a1a19; --panel:#222220;
        --ink:#f0efec; --ink2:#c3c2b7; --line:#3a3936; --accent:#3987e5; --warn-bg:#3a2c10; --warn-ink:#f3cf8a; } }
:root[data-theme="dark"] { --bg:#1a1a19; --panel:#222220; --ink:#f0efec; --ink2:#c3c2b7; --line:#3a3936;
        --accent:#3987e5; --warn-bg:#3a2c10; --warn-ink:#f3cf8a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 -apple-system, system-ui, sans-serif; }
main { max-width:1180px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 10px; border-bottom:1px solid var(--line);
     padding-bottom:6px; } h3 { font-size:14px; margin:18px 0 8px; color:var(--ink2); }
.sub { color:var(--ink2); margin:0 0 16px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:10px; }
.tile { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
.tile .k { color:var(--ink2); font-size:12px; } .tile .v { font-size:18px; font-weight:600; font-variant-numeric:tabular-nums; }
.note { background:var(--warn-bg); color:var(--warn-ink); border-radius:8px; padding:8px 12px; margin:10px 0; }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--panel); }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:13px; }
th, td { padding:6px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th:first-child, td:first-child, td.l, th.l { text-align:left; } th { color:var(--ink2); font-weight:600; }
.figs { display:grid; grid-template-columns:repeat(auto-fit, minmax(340px, 1fr)); gap:12px; }
figure { margin:0; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px; }
figure img { width:100%; height:auto; display:block; } figcaption { color:var(--ink2); font-size:12px; padding-top:4px; }
.tag { display:inline-block; font-size:11px; padding:1px 6px; border-radius:4px; border:1px solid var(--line);
       color:var(--ink2); margin-left:6px; }
"""


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _img(path) -> str | None:
    p = Path(path) if path else None
    if not p or not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def _table(df: pd.DataFrame, max_rows: int = 200, left=()) -> str:
    if df is None or df.empty:
        return "<p class='sub'>No rows yet.</p>"
    df = df.head(max_rows)
    head = "".join(f"<th class='{'l' if c in left else ''}'>{_e(c)}</th>" for c in df.columns)
    body = []
    for r in df.itertuples(index=False):
        cells = []
        for c, v in zip(df.columns, r):
            if isinstance(v, float):
                v = "" if pd.isna(v) else f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
            cells.append(f"<td class='{'l' if c in left else ''}'>{_e(v)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _tiles(items) -> str:
    return "<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='k'>{_e(k)}</div><div class='v'>{_e(v)}</div></div>" for k, v in items) + "</div>"


def _figs(figs: dict, captions: dict) -> str:
    out = []
    for key, cap in captions.items():
        src = _img(figs.get(key))
        if src:
            out.append(f"<figure><img src='{src}' alt='{_e(cap)}'><figcaption>{_e(cap)}</figcaption></figure>")
    return "<div class='figs'>" + "".join(out) + "</div>" if out else "<p class='sub'>No charts yet.</p>"


def _model_table(rep: dict) -> pd.DataFrame:
    rows = []
    for c, s in sorted(rep["models"].items(), key=lambda kv: kv[1]["log_loss"]):
        rows.append({"model": MODEL_LABELS.get(c, c), "log loss [95% CI]": fmt_ci(s.get("log_loss_ci")),
                     "brier": s["brier"], "accuracy": s["accuracy"], "roc_auc": s["roc_auc"], "ece": s["ece"],
                     "cal. slope": s["calibration_slope"], "n": s["n"]})
    return pd.DataFrame(rows)


def _backtest_section(title: str, d: Path, simulated_note: str) -> str:
    rp = d / "report.json"
    if not rp.exists():
        return f"<h2>{_e(title)}</h2><p class='sub'>Not run yet.</p>"
    rep = json.loads(rp.read_text())
    parts = [f"<h2>{_e(title)}</h2>",
             f"<p class='sub'>{rep['n_games']} games, {rep['date_range'][0]} to {rep['date_range'][1]}. "
             "Every probability was produced by a model trained only on earlier games.</p>"]
    for w in rep.get("final_test_warnings", []):
        parts.append(f"<div class='note'>{_e(w)}</div>")
    if rep.get("champion"):
        ch = rep["champion"]
        parts.append(f"<p>Champion by development log loss: <b>{_e(MODEL_LABELS.get(ch['champion'], ch['champion']))}"
                     f"</b>{' (' + _e(ch['note']) + ')' if ch.get('note') else ''}</p>")
    if rep.get("champion_from_development"):
        parts.append(f"<p>Champion fixed in advance: <b>{_e(MODEL_LABELS.get(rep['champion_from_development']))}</b></p>")
    parts.append(_table(_model_table(rep), left=("model", "log loss [95% CI]")))
    comps = [{"comparison": k.replace("p_", ""), "log loss diff [95% CI]": fmt_ci(v["log_loss"]),
              "brier diff [95% CI]": fmt_ci(v["brier"])} for k, v in rep["comparisons"].items()]
    parts.append("<h3>Paired differences (negative favours the first model; CIs resample whole dates)</h3>")
    parts.append(_table(pd.DataFrame(comps), left=("comparison", "log loss diff [95% CI]", "brier diff [95% CI]")))
    if rep.get("market"):
        parts.append(f"<p>Market (no-vig) log loss on {rep['market']['n_games_with_odds']} priced games: "
                     f"{rep['market']['market_novig']['log_loss']:.4f}</p>")
    else:
        parts.append("<p class='sub'>No historical sportsbook odds were supplied, so there is no market comparison.</p>")
    bs = d / "by_season.csv"
    if bs.exists():
        t = pd.read_csv(bs)
        keep = ["season", "n"] + [c for c in t.columns if c.endswith("_log_loss")]
        parts.append("<h3>Log loss by season</h3>" + _table(t[keep].rename(
            columns=lambda c: MODEL_LABELS.get(c.replace("_log_loss", ""), c))))
    parts.append(_figs(rep.get("figures", {}), {
        "scores": "Log loss and Brier score with 95% CIs", "reliability": "Reliability diagram (calibration curve)",
        "log_loss_by_season": "Log loss by season", "rolling": "Monthly log loss over time",
        "distribution": "Predicted probability distributions", "calibration_drift": "Calibration error by season",
        "logistic_standardized_coefficients": "Logistic regression standardized coefficients (last period)",
        "gbm_permutation_log_loss_increase": "Boosting permutation importance (last period, calibration window)"}))
    pt = rep.get("paper_trading") or {}
    parts.append(f"<h3>Paper trading <span class='tag'>SIMULATED</span></h3>")
    if pt.get("status") == "ok":
        fl = pt["flat"]
        parts.append(_tiles([("Bets", pt["n_bets"]), ("Flat ROI [95% CI]", fmt_ci(fl["roi"])),
                             ("Flat max drawdown (u)", f"{fl['max_drawdown_units']:.2f}"),
                             ("Kelly final bankroll", f"{pt['fractional_kelly']['final_bankroll']:.1f}")]))
        parts.append(f"<div class='note'>{_e(pt['caveat'])} "
                     f"{_e(pt.get('overfitting_check', {}).get('interpretation', ''))}</div>")
        src = _img(pt.get("figure"))
        if src:
            parts.append(f"<figure><img src='{src}' alt='Simulated bankroll'><figcaption>{_e(simulated_note)}"
                         "</figcaption></figure>")
    else:
        parts.append(f"<p class='sub'>Not simulated ({_e(pt.get('status', 'n/a'))}): paper trading runs only on "
                     "archived, timestamped sportsbook prices observed before each decision. Decisions: "
                     f"{_e(pt.get('decisions'))}</p>")
    return "".join(parts)


def _ci(d) -> str:
    return "" if not d else f"{d['estimate']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"


def _odds_section(con, cfg: dict, now: pd.Timestamp) -> str:
    book = cfg.get("paper", {}).get("book", cfg["paths"]["book"])
    parts = ["<h2>Odds monitoring <span class='tag'>LIVE DATA</span></h2>"]
    snap = pd.read_sql("SELECT * FROM odds_snapshots WHERE is_pregame = 1 AND match_status = 'matched'", con)
    runs = pd.read_sql("SELECT started_at, status, reason, n_events, n_inserted, n_invalid, n_unmatched, n_live, "
                       "credits_remaining, error FROM odds_collection_runs ORDER BY id DESC LIMIT 10", con)
    n_all = con.execute("SELECT COUNT(*), COUNT(DISTINCT observed_at) FROM odds_snapshots").fetchone()
    last_ok = runs.loc[runs["status"] == "ok", "started_at"].max() if not runs.empty else None
    parts.append(_tiles([("Snapshot rows stored", n_all[0]), ("Distinct observation times", n_all[1]),
                         ("Last successful collection", (last_ok or "never")[:16]),
                         ("Credits remaining", runs["credits_remaining"].dropna().iloc[0]
                          if not runs.empty and runs["credits_remaining"].notna().any() else "unknown")]))
    if not snap.empty:
        snap["observed_at"] = pd.to_datetime(snap["observed_at"], utc=True)
        snap["tip"] = pd.to_datetime(snap["tipoff_utc"], utc=True)
        up = snap[snap["tip"] > now].sort_values("observed_at")
        if not up.empty:
            latest = up.groupby(["game_key", "book"], as_index=False).tail(1)
            first = up.groupby(["game_key", "book"], as_index=False).head(1)
            ref = latest[latest["book"] == book].merge(
                first[["game_key", "book", "home_american", "away_american"]], on=["game_key", "book"],
                suffixes=("", "_open"))
            books = latest.groupby("game_key")["book"].nunique().rename("books")
            ref = ref.merge(books, left_on="game_key", right_index=True)
            t = pd.DataFrame({"tip-off (UTC)": ref["tipoff_utc"].str[:16], "game": ref["away_team"] + " @ " + ref["home_team"],
                              f"{book} home": ref["home_american"].map("{:+.0f}".format),
                              f"{book} away": ref["away_american"].map("{:+.0f}".format),
                              "home at first snapshot": ref["home_american_open"].map("{:+.0f}".format),
                              "latest snapshot (UTC)": ref["observed_at"].dt.strftime("%m-%d %H:%M"),
                              "books quoting": ref["books"]})
            parts.append("<h3>Upcoming games: latest pregame prices</h3>" + _table(
                t.sort_values("tip-off (UTC)"), left=("tip-off (UTC)", "game", "latest snapshot (UTC)")))
            other = latest.assign(price=latest["home_american"].map("{:+.0f}".format) + " / " +
                                  latest["away_american"].map("{:+.0f}".format))
            wide = other.pivot_table(index="game_key", columns="book", values="price", aggfunc="first")
            names = latest.drop_duplicates("game_key").set_index("game_key")
            wide.insert(0, "game", names["away_team"] + " @ " + names["home_team"])
            parts.append("<h3>All books (home / away, American)</h3>" + _table(wide.reset_index(drop=True),
                                                                              left=("game",)))
    pred = pd.read_sql("""SELECT DISTINCT p.game_key, p.game_date, p.away_team || ' @ ' || p.home_team AS game, p.tipoff_utc
                          FROM predictions p LEFT JOIN results r USING (game_key) WHERE r.game_key IS NULL""", con)
    if not pred.empty:
        missing = pred[~pred["game_key"].isin(snap["game_key"] if not snap.empty else [])]
        if not missing.empty:
            parts.append(f"<div class='note'>Missing odds: {len(missing)} upcoming predicted game(s) have no pregame "
                         f"snapshot yet: {_e(', '.join(missing['game']))}</div>")
    parts.append("<h3>Recent collection runs</h3>" + _table(runs, left=("started_at", "status", "reason", "error")))
    return "".join(parts)


def _market_section(con, cfg: dict) -> str:
    book = cfg.get("paper", {}).get("book", cfg["paths"]["book"])
    parts = ["<h2>Model versus market <span class='tag'>LIVE PREDICTIONS</span></h2>",
             f"<p class='sub'>Book: {_e(book)}. The price shown is the one recorded when the prediction was made "
             "(observed at or before it). Market probability is no-vig (proportional). Expected ROI assumes the "
             "model probability is correct.</p>"]
    v = market.upcoming_market_view(con, book)
    if v.empty:
        parts.append("<p class='sub'>No unresolved predictions.</p>")
    else:
        t = pd.DataFrame({"date": v["game_date"], "game": v["away_team"] + " @ " + v["home_team"],
                          "p(home) model": v["p_home"], "p(home) market": v["p_market"],
                          "model fair ML": v["fair_home_american"].map("{:+d}".format) + " / " +
                          v["fair_away_american"].map("{:+d}".format),
                          "book ML": [f"{h:+.0f} / {a:+.0f}" if pd.notna(h) else "no price"
                                      for h, a in zip(v["mkt_home_american"], v["mkt_away_american"])],
                          "edge (home)": v["edge_home"], "exp. ROI home": v["exp_roi_home"],
                          "exp. ROI away": v["exp_roi_away"], "price observed": v["mkt_observed_at"].str[5:16],
                          "predicted at": v["prediction_ts"].str[5:16]})
        parts.append(_table(t, left=("date", "game", "model fair ML", "book ML", "price observed", "predicted at")))
    bench = market.live_benchmark(con, book, n_boot=300)
    rows = []
    for h, r in bench["by_horizon"].items():
        if r.get("status") != "ok":
            rows.append({"horizon before tip": h, "n": r.get("n", 0), "model": "-", "log loss": None,
                         "minus market [95% CI]": r.get("status")})
            continue
        rows.append({"horizon before tip": h, "n": r["n"], "model": "Market (no-vig)",
                     "log loss": r["market"]["log_loss"], "minus market [95% CI]": ""})
        for c, s in r["models"].items():
            rows.append({"horizon before tip": h, "n": r["n"], "model": MODEL_LABELS.get(c, c),
                         "log loss": s["log_loss"], "minus market [95% CI]": _ci(s["log_loss_minus_market"])})
    parts.append("<h3>Market benchmark on scored live predictions <span class='tag'>LIVE EVALUATION</span></h3>"
                 "<p class='sub'>Same games and same timestamps for every row in a horizon. Negative difference "
                 "means the model beat the market. Fewer than 500 games is preliminary.</p>"
                 + _table(pd.DataFrame(rows), left=("horizon before tip", "model", "minus market [95% CI]")))
    return "".join(parts)


def _blend_section(cfg: dict) -> str:
    rp = Path(cfg["paths"].get("blend_dir", "out/blend")) / "report.json"
    parts = ["<h2>Early-season Elo blending <span class='tag'>HISTORICAL EVALUATION</span></h2>"]
    if not rp.exists():
        return parts[0] + "<p class='sub'>Not run yet: <code>python -m nbamodel blend-study</code>.</p>"
    rep = json.loads(rp.read_text())
    parts.append(f"<p class='sub'>P = w &times; P(Elo) + (1 &minus; w) &times; P({_e(rep['base_col'])}). Weights are "
                 f"chosen on earlier seasons and scored on later ones ({rep['n']} development games, seasons "
                 f"{rep['evaluated_seasons'][0]}-{rep['evaluated_seasons'][-1]}). Best method: "
                 f"<b>{_e(rep['best_method'])}</b> {_e(rep['final_params'])}. Enabled: <b>{'yes' if rep['enabled'] else 'no'}"
                 f"</b> (rule: {_e(rep['decision_rule'])}).</p>")
    rows = []
    for c, v in rep["overall"].items():
        row = {"model": c.replace("p_blend_", "blend: "), "log loss (all)": v["log_loss"],
               "diff vs base (all)": _ci(v.get("log_loss_diff_vs_base"))}
        for seg, blk in rep["by_segment"].items():
            row[f"diff {seg}"] = _ci(blk.get(c, {}).get("log_loss_diff_vs_base"))
        rows.append(row)
    parts.append(_table(pd.DataFrame(rows), left=tuple(["model"] + [k for k in rows[0] if k.startswith("diff")])))
    if rep.get("holdout_confirmation"):
        h = rep["holdout_confirmation"]["overall"]["p_blend"]
        parts.append(f"<p>Final-holdout confirmation (one look, nothing refitted): log loss diff vs base "
                     f"{_ci(h.get('log_loss_diff_vs_base'))}</p>")
    parts.append(_figs({"b": rep.get("figure")}, {"b": "Log-loss difference vs ML base by season segment (95% CI)"}))
    return "".join(parts)


def _paper_section(con, cfg: dict, out_dir: Path) -> str:
    pcfg = paper.PaperConfig.from_dict(cfg.get("paper"))
    s = paper.summarize(con, pcfg, n_boot=300)
    c = clv.summary(con, pcfg.strategy)
    parts = [f"<h2>Paper trading <span class='tag'>SIMULATED</span> <span class='tag'>LIVE LEDGER</span></h2>",
             f"<div class='note'>{_e(s['caveat'])} Strategy {_e(pcfg.strategy)}, book {_e(pcfg.book)}, EV threshold "
             f"{pcfg.ev_threshold:.0%}, decision {pcfg.decision_min_lead_minutes:.0f}-{pcfg.decision_max_lead_minutes:.0f} "
             "minutes before tip-off.</div>"]
    tiles = [("Decisions", sum(s["decisions"].values())), ("Simulated bets", s["n_bets"]),
             ("Settled / open", f"{s['n_settled']} / {s['n_open']}"),
             ("Mean est. EV", f"{s['mean_est_ev']:.3f}" if s["mean_est_ev"] is not None else "n/a"),
             ("Mean edge vs no-vig", f"{s['mean_edge_vs_novig']:.3f}" if s["mean_edge_vs_novig"] is not None else "n/a"),
             ("Mean price CLV (bets)", f"{s['mean_price_clv']:+.3f}" if s["mean_price_clv"] is not None else "n/a")]
    if s["status"] == "ok":
        tiles += [("Win rate", f"{s['win_rate']:.3f}"), ("Flat ROI [95% CI]", _ci(s["flat"]["roi"])),
                  ("Flat max drawdown", f"{s['flat']['max_drawdown_units']:.2f}u"),
                  ("Kelly bankroll", f"{s['kelly']['bankroll']:.2f}"),
                  ("Kelly max drawdown", f"{s['kelly']['max_drawdown_fraction']:.1%}")]
    parts.append(_tiles(tiles))
    parts.append(f"<p class='sub'>Decision counts: {_e(s['decisions'])}. CLV status: {_e(c.get('status_counts', {}))}. "
                 f"{_e(c.get('label', ''))}; positive CLV alone is not proof of profitability.</p>")
    if s.get("bankroll_history"):
        fig = plots.paper_bankroll(s["bankroll_history"], out_dir / "paper_bankroll.png", pcfg.initial_bankroll)
        parts.append(_figs({"b": fig}, {"b": "SIMULATED live paper-trading equity"}))
    led = paper.ledger(con, pcfg.strategy)
    if not led.empty:
        t = pd.DataFrame({"tip-off": led["tipoff_utc"].str[:16], "game": led["game_key"], "decision": led["decision"],
                          "side": led["side"], "price": led["american_odds"], "p(side)": led["p_side"],
                          "no-vig": led["market_p_side_novig"], "est. EV": led["est_ev"],
                          "Kelly stake": led["stake_kelly"], "won": led["won"], "flat P&L": led["pnl_flat"],
                          "price CLV": led["price_clv"]})
        parts.append("<h3>Decision ledger (newest first)</h3>" + _table(t.iloc[::-1], left=("tip-off", "game", "decision", "side")))
    return "".join(parts)


def build_dashboard(cfg: dict, out=None) -> str:
    paths = cfg["paths"]
    out = Path(out or paths["dashboard"])
    models = registry.list_models(paths["artifacts"])
    active = registry.active_version(paths["artifacts"])
    meta = next((m for m in models if m["version"] == active), None)
    con = store.connect(paths["db"])
    upcoming = pd.read_sql("""SELECT p.* FROM predictions p LEFT JOIN results r USING (game_key)
                              WHERE r.game_key IS NULL ORDER BY game_date, tipoff_utc, prediction_ts""", con)
    if not upcoming.empty:
        upcoming = upcoming.groupby(["game_key", "model_version"], as_index=False).tail(1)
    hist = store.official_evaluations(con)
    metrics = pd.read_sql("SELECT * FROM metrics_history ORDER BY computed_at DESC", con)
    runs = pd.read_sql("SELECT model_version, trained_at, reason, train_start, train_end, n_rows FROM training_runs "
                       "ORDER BY trained_at DESC", con)
    alerts = pd.read_sql("SELECT created_at, kind, model_version, message FROM alerts ORDER BY created_at DESC", con)
    now = pd.Timestamp.now(tz="UTC")
    phase3 = [_odds_section(con, cfg, now), _market_section(con, cfg), _paper_section(con, cfg, out.parent),
              _blend_section(cfg)]
    con.close()

    parts = [f"<h1>NBA moneyline model</h1><p class='sub'>Research and paper-trading dashboard. Generated "
             f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC. Fair odds are model prices, not bookmaker prices. "
             "No result here is evidence of future profit.</p>"]

    parts.append("<h2>Model overview</h2>")
    if meta:
        t = meta["training"]
        live = metrics[(metrics["model_version"] == active) & (metrics["window"] == "all")].head(1)
        ref = meta.get("reference_metrics") or {}
        parts.append(_tiles([
            ("Active model", meta["model"] + (" (experimental)" if meta.get("experimental") else "")),
            ("Version", meta["version"]), ("Training games", t["n_rows"]),
            ("Training range", f"{t['start']} to {t['end']}"), ("Last trained", meta["trained_at"][:16]),
            ("Walk-forward log loss", f"{ref['log_loss']:.4f}" if ref.get("log_loss") else "n/a"),
            ("Walk-forward Brier", f"{ref['brier']:.4f}" if ref.get("brier") else "n/a"),
            ("Live log loss", f"{live['log_loss'].iloc[0]:.4f} (n={int(live['n'].iloc[0])})" if not live.empty else "no results yet"),
            ("Live accuracy", f"{live['accuracy'].iloc[0]:.3f}" if not live.empty else "n/a"),
            ("Live Brier", f"{live['brier'].iloc[0]:.4f}" if not live.empty else "n/a"),
        ]))
    else:
        parts.append("<p class='sub'>No trained model yet. Run <code>python -m nbamodel train</code>.</p>")

    parts.append("<h2>Upcoming games</h2>")
    if not upcoming.empty:
        u = upcoming.assign(
            game=upcoming["away_team"] + " @ " + upcoming["home_team"],
            fair=upcoming["fair_home_american"].map("{:+d}".format) + " / " + upcoming["fair_away_american"].map("{:+d}".format),
            low_history=upcoming["low_history"].map({1: "yes", 0: ""}))
        cols = {"game_date": "date", "tipoff_utc": "tip-off (UTC)", "game": "game", "p_home": "p(home)",
                "p_away": "p(away)", "fair": "fair ML (home/away)", "model_spread": "model spread",
                "market_home_novig": "market p(home) no-vig", "low_history": "low history",
                "features_asof": "data through", "key_factors": "key factors", "model_version": "model"}
        parts.append(_table(u[list(cols)].rename(columns=cols), left=("date", "tip-off (UTC)", "game", "fair ML (home/away)",
                                                                      "key factors", "model", "data through")))
        parts.append("<p class='sub'>Model spread = max minus min of the Elo, logistic and boosting probabilities: a "
                     "disagreement signal, not a confidence interval.</p>")
    else:
        parts.append("<p class='sub'>No logged upcoming predictions. Run <code>python -m nbamodel predict</code>.</p>")

    parts.extend(phase3[:3])

    parts.append("<h2>Historical predictions (live log)</h2>")
    if not hist.empty:
        h = hist.sort_values("game_date", ascending=False).assign(
            game=hist["away_team"] + " @ " + hist["home_team"],
            result=hist["home_score"].astype(str) + "-" + hist["away_score"].astype(str),
            correct=hist["correct"].map({1: "yes", 0: "no"}))
        cols = {"game_date": "date", "game": "game", "p_home": "p(home)", "result": "home-away", "home_win": "home won",
                "correct": "correct", "abs_error": "|error|", "log_loss": "log loss", "model_version": "model",
                "prediction_ts": "predicted at"}
        parts.append(_table(h[list(cols)].rename(columns=cols), left=("date", "game", "model", "predicted at")))
    else:
        parts.append("<p class='sub'>No predicted games have results yet.</p>")

    parts.append(phase3[3])
    parts.append(_backtest_section("Backtesting: development walk-forward", Path(paths["walkforward_dir"]),
                                   "Simulated bankroll on development predictions"))
    parts.append(_backtest_section("Backtesting: final holdout", Path(paths["final_test_dir"]),
                                   "Simulated bankroll on final-holdout predictions"))

    parts.append("<h2>Model diagnostics</h2>")
    if meta:
        coef = meta.get("logistic_coefficients", {})
        parts.append("<h3>Production logistic coefficients (standardized)</h3>" + _table(pd.DataFrame(
            sorted(coef.items(), key=lambda kv: -abs(kv[1])), columns=["feature", "coefficient"]), left=("feature",)))
    parts.append("<h3>Live metrics history</h3>" + _table(metrics.drop(columns=["id"], errors="ignore").head(30),
                                                          left=("computed_at", "model_version", "window")))
    parts.append("<h3>Alerts</h3>" + _table(alerts, left=("created_at", "kind", "model_version", "message")))
    parts.append("<h3>Retraining history</h3>" + _table(runs, left=("model_version", "trained_at", "reason")))
    parts.append("<h3>Model registry</h3>" + _table(pd.DataFrame([
        {"version": m["version"], "model": m["model"], "trained_at": m["trained_at"][:16],
         "rows": m["training"]["n_rows"], "active": "yes" if m["version"] == active else "",
         "experimental": "yes" if m.get("experimental") else ""} for m in models]), left=("version", "model", "trained_at")))

    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width, initial-scale=1'><title>Moneyline Model Dashboard</title>"
           f"<style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return str(out)
