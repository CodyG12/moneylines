# NBA Moneyline Model

> **Phase 2** (walk-forward backtesting, three model families, calibration, prediction log, dashboard) is
> documented in [Phase 2](#phase-2-walk-forward-models-and-feedback) below. The Phase 1 pipeline
> (`nba_model.py`) is unchanged and still works.

A research and **paper-trading** system that estimates NBA home-team win
probabilities, converts them to fair moneyline odds, and evaluates them on
unseen, chronologically later games. It never places bets. Nothing here
assumes the model can beat sportsbooks: it has to show that on held-out data
with archived, timestamped prices, and it hasn't done so yet.

## What it does

1. Loads and validates a games CSV (rejects ties, duplicates, a team in two games on one date, half-filled scores, and so on).
2. Builds pregame features using **only games from earlier dates**:
   - Elo with home-court advantage, margin-of-victory scaling and between-season regression to the mean
   - recent form: mean point margin and win % over the last N games of the season
   - days of rest and back-to-back flags (taken from the schedule)
3. Splits games chronologically **by date** into train / validation / holdout (60/20/20 by default).
4. Fits standardized logistic regression on train, picks the regularization strength on validation log loss, refits on train+validation, and scores the holdout **once**.
5. Compares against two baselines: raw Elo and the home-team base rate. If odds are supplied, it also compares against the **no-vig market**.
6. Reports log loss, Brier score and accuracy with 95% CIs from a **date-clustered bootstrap**, plus a calibration table.
7. If odds are supplied, runs paper trading on the holdout: a flat 1-unit stake and capped fractional Kelly, with ROI CI and max drawdown.
8. Predicts any scheduled (score-less) games after the last played date.

## Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv          # if the repo was moved/renamed, recreate the venv: activate hard-codes its path
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Run on the synthetic sample

```bash
python nba_model.py --games data/sample/synthetic_games.csv --output out/predictions.csv
```

> `data/sample/synthetic_games.csv` is **SYNTHETIC**: simulated games between
> fictional teams `SYN01` to `SYN30`, created by `scripts/make_sample_data.py`
> (seed 7). It shows the format and exercises the code. Its results say
> nothing about real-world performance. In this sample the synthetic market is
> generated from the true probabilities, so the market *should* beat the model.

## Run on real NBA data

```bash
pip install -r requirements-fetch.txt        # nba_api (1.10.2 verified on Python 3.9)
python scripts/fetch_nba_api.py --seasons 2014-15 2015-16 2016-17 2017-18 2018-19 2019-20 2020-21 \
    2021-22 2022-23 2023-24 2024-25 2025-26 2026-27 \
    --out data/raw/nba_games.csv
python nba_model.py --games data/raw/nba_games.csv --output out/predictions.csv
```

The fetch pulls regular-season games only; playoffs and the NBA Cup final (which doesn't count in
regular-season stats) are excluded. It writes an explicit `season` column, so the 2019-20 bubble and
the late 2020-21 start are labeled correctly. Neutral-site games are flagged `neutral_site=1` and get no
home-court advantage. **Caveat:** the API marks them this way (both rows as `@`) only from 2024-25 on.
Earlier neutral games, such as London, Mexico City and the 2023 Cup semifinals, come back as ordinary
home/away games and aren't flagged. `nba_api` returns completed games only, so `predictions_upcoming.csv`
is empty unless you add scheduled games to the CSV yourself.

Or supply your own CSV in the format below. Use several full seasons. The
code refuses to run on fewer than 100 played games across 30 dates, but that
is only a minimum for the code to work. It is far too little for meaningful
evaluation.

## Predict upcoming games

```bash
python scripts/update_schedule.py --games data/raw/nba_games.csv --season 2026-27
python nba_model.py --games data/raw/nba_games.csv --output out/nba/predictions.csv --upcoming-days 3
```

`update_schedule.py` replaces the current season's rows with the NBA schedule. Finished games get their
scores. Scheduled games get blank scores, UTC tip-off times and neutral-site flags. Preseason and the NBA
Cup final are excluded. Rerun it daily so the latest results feed into Elo and form.

**Early-season caution:** for the first few games of a season, "recent form" averages only 1 to 4
games, and those predictions are flagged `[low history]`. On the validation period, games where either
team had played fewer than 5 games scored worse (log loss 0.653, n=160) than the rest of the season
(0.626, n=2,726). Treat early-season probabilities as less reliable. Shrinking form toward zero early in
the season is planned for Phase 2.

## Timestamped odds (The Odds API)

1. Get a free key at [the-odds-api.com](https://the-odds-api.com). Then `cp .env.example .env` and put
   the key in `.env`, which is git-ignored.
2. Collect snapshots. Each run appends every book's current NBA moneylines, stamped with the time it was
   observed:
   ```bash
   python scripts/collect_odds.py --out data/raw/odds_snapshots.csv
   ```
   On the free plan each run costs 1 credit (h2h market, `us` region). Run it a few times on game days,
   for example by cron, to build your own timestamped archive. With a **paid** plan you can backfill
   from June 2020 at 10 credits per snapshot:
   `--historical --start 2024-10-22 --end 2025-04-13 --time 22:00`.
3. Join the odds and run:
   ```bash
   python nba_model.py --games data/raw/nba_games.csv --odds data/raw/odds_snapshots.csv \
       --book draftkings --odds-lead-minutes 60 --output out/nba/predictions.csv
   ```
   For each game, the run uses the latest quote observed at least `--odds-lead-minutes` before
   tip-off. In-game quotes are never used. `--book best` takes the best price per side across all
   books. That assumes you hold an account at every book, so it overstates realistic edges.

## Input format

| column | required | notes |
|---|---|---|
| `date` | yes | game date, `YYYY-MM-DD` |
| `home_team`, `away_team` | yes | team codes, kept consistent across seasons |
| `home_score`, `away_score` | yes | leave **both** blank for scheduled games |
| `season` | recommended | season start year (2023 = 2023-24). Inferred from the month if absent, which mislabels the 2020 bubble/late start |
| `home_odds`, `away_odds` | optional | decimal moneylines from the **same** book/snapshot |
| `home_moneyline`, `away_moneyline` | optional | American odds, used if decimal is absent |
| `tipoff_utc` | optional | scheduled tip-off; odds quoted at or after it are dropped |
| `neutral_site` | optional | 1/true when neither team is truly at home (default 0) |
| `odds_timestamp` | optional | ISO time the quote was observed. Odds stamped after the game date are dropped. Without timestamps the run warns that the results are not executable-price backtests |

Lines starting with `#` are comments.

## Outputs

Given `--output out/predictions.csv`:

| file | contents |
|---|---|
| `predictions.csv` | holdout games: Elo and model probabilities, fair decimal/American odds, market no-vig probability, estimated EV per side |
| `predictions_metrics.json` | split boundaries, model selection, holdout metrics with CIs, market comparison, paper-trading summary, data warnings |
| `predictions_calibration.csv` | reliability table (predicted vs observed, Wilson intervals) |
| `predictions_paper_bets.csv` | each hypothetical bet with flat and Kelly stake/P&L (only when odds exist) |
| `predictions_upcoming.csv` | predictions for scheduled games, with a `low_history` flag |

Useful flags: `--window`, `--elo-k`, `--elo-home-advantage`, `--elo-carryover`,
`--train-frac`, `--validation-frac`, `--ev-threshold`, `--kelly-multiplier`,
`--n-boot`, `--seed`, `-v`.

## How to read the results

- **Log loss and Brier are the main metrics** (lower is better). Accuracy is shown but is not the target.
- A model only beats a baseline if the CI for `logistic_minus_<baseline>_log_loss` lies **entirely below 0**.
- **Estimated EV is not profit.** It assumes the model's probabilities are right. Realized holdout P&L is a single noisy sample, so read its ROI CI.
- Kelly sizing assumes exact probabilities, which no model has. The simulation uses 0.25x Kelly with per-bet (2%) and per-day (10%) caps, and it is still not "safe".
- Every CI resamples whole dates, because games on the same night aren't independent for betting purposes.

## Leakage safeguards (tested)

- Elo and form for date D come from games before D. Same-day results are never used (`tests/test_features.py`, `tests/test_elo.py`).
- Scrambling outcomes on or after a date leaves that date's features unchanged.
- Flipping every holdout outcome leaves the first holdout day's predictions and the chosen hyperparameter unchanged (`tests/test_pipeline.py`).
- Splits are by date, so no date appears in two partitions.
- Odds quoted after the game date are discarded.

Elo hyperparameters (K=20, HCA=100, carryover 0.75) are fixed defaults, not tuned. If you tune them, tune on validation only.

## Data sources and licensing

| source | what | cost / constraints |
|---|---|---|
| [`nba_api`](https://github.com/swar/nba_api) (stats.nba.com) | game results, box scores, team/player stats | Free. Unofficial wrapper of an undocumented site API. NBA.com terms limit use to personal, non-commercial purposes. Rate-limit your requests and don't redistribute the data. |
| [The Odds API](https://the-odds-api.com) | timestamped bookmaker moneylines | Free plan: current odds at 1 credit per call, with a small monthly quota. **Historical** snapshots (from June 2020, 5 to 10 minute granularity) need a paid plan at 10 credits per call. Check their terms before redistributing data. |
| Community CSV datasets (Kaggle, GitHub) | historical results, sometimes odds | Check each dataset's license. Odds archives often lack quote timestamps or mix opening and closing lines, so verify them before backtesting. |

No scraping is implemented. Provider conversion lives in `nbamodel/sources.py`.

## Phase 2: walk-forward models and feedback

All commands read `config/phase2.json`; override with `--config` or per-command flags.
`make <target>` wraps each of them (see `Makefile`).

| step | command |
|---|---|
| fetch history (box scores + tip-off times) | `make data-fetch` |
| refresh current season | `make data-update` (`scripts/update_schedule.py`) |
| build the training dataset | `python -m nbamodel build-dataset` |
| development walk-forward + champion choice | `python -m nbamodel walk-forward [--retrain weekly\|monthly\|season] [--window expanding\|rolling] [--calibration none\|platt\|isotonic]` |
| evaluate calibrated probabilities / any predictions file | in the walk-forward report; or `python -m nbamodel evaluate --predictions FILE --out DIR` |
| final holdout test (**once**) | `python -m nbamodel final-test` |
| train + version the production model | `python -m nbamodel train [--model ...]` |
| predict upcoming games | `python -m nbamodel predict [--days 3] [--odds data/raw/odds_snapshots.csv]` |
| record results, update metrics/Elo, check drift | `python -m nbamodel update-results [--retrain-if-due]` |
| dashboard | `python -m nbamodel dashboard` -> `out/dashboard/index.html` |
| tests | `make test` |

Game day: `make daily` (refresh data, reconcile and retrain if due, predict, rebuild dashboard).

### Data

`scripts/fetch_nba_api.py` now keeps team box-score totals (FGA, FGM, 3PM, FTA, OREB, DREB, TOV)
for possession estimates, adds tip-off times and neutral flags from the schedule endpoint where it
exists, and stamps `fetched_at`. `update_schedule.py` also fills in box scores for completed
current-season games. On the current file: 14,482 played regular-season games (2014-15 to 43 games
into 2026-27), all with box scores; tip-off times exist from 2017-18 onward (older rows use a nominal
09:00 ET prediction time). Each game gets a stable `game_key` (the NBA game id).

### Dataset and features (`nbamodel/dataset.py`, `nbamodel/features.py`)

Each row is one game as known at its `prediction_timestamp` (tip-off minus 60 minutes, or 09:00 ET).
Rows record `features_asof` (results through the previous ET date), each team's previous game date,
`source_fetched_at` and `train_eligible`. Final scores and box scores are kept only as labels, and
`validate_dataset` rejects any row whose inputs could post-date the game. `FEATURES_V2` contains:

- **Strength:** Elo difference, opponent- and venue-adjusted margin, wins over Elo expectation
  (opponent-adjusted win %), strength of schedule, home-games margin vs road-games margin
- **Recent form:** last-5 and last-10 margin, last-10 win %, exponentially weighted margin (half-life
  7 games), last-10 net rating
- **Efficiency:** offensive and defensive ratings per 100 possessions, season to date, as
  "home offence + away defence" and the reverse, relative to the league average
- **Schedule:** rest difference, back-to-backs, consecutive home or road games
- **Matchup:** head-to-head residual vs Elo over this season and last, shrunk with a 4-game prior;
  games played (early-season flag)

Every season-level and rolling stat is shrunk toward a neutral prior, which addresses the Phase 1
early-season caveat. Games are processed a date at a time, so same-night games never see each
other's results.

**Not included: injuries and lineups.** There is no reliable, timestamped historical source wired in.
`nbamodel/availability.py` is an adapter for one. It requires a report timestamp on every row and
uses only reports published before the prediction time.

**Leakage checks** (`nbamodel/leakage.py`, run by `build-dataset` and `walk-forward`, and tested): flip
every result on or after a cutoff and require features on the cutoff date to be unchanged, then
screen for features almost perfectly correlated with the label. Tests confirm the checks catch a
deliberately leaky "final margin" feature and an off-by-one rolling window.

### Models (`nbamodel/models.py`)

| | model | tuning, always inside the training window |
|---|---|---|
| A | Elo (`elo.py`) | K, home advantage and season carryover grid, chosen on the window's own one-step-ahead log loss |
| B | logistic regression (median impute + standardize + L2) | C on the last 20% of fit-window dates |
| C | gradient boosting (scikit-learn `HistGradientBoostingClassifier`, monotone in Elo / adjusted margin / efficiency matchups; `gbm_backend: "lightgbm"` optional) | 3-config grid, same inner split |
| D | ensemble (logit-mean of B and C, no fitted weights) | eligible as champion only if it beats both components |

Calibration (Platt or isotonic) is fitted on the 40 most recent training dates, which are held out
from model fitting. Calibrated and uncalibrated variants are both scored. Seeds are fixed. The
walk-forward picks the champion by development log loss. `train` refits it on all eligible games
and saves `artifacts/models/<version>/{model.joblib,metadata.json}`. The metadata records the
features with definitions, hyperparameters, training range, data hash, git commit and reference
walk-forward metrics. `active.json` points to the live version.

### Walk-forward (`nbamodel/walkforward.py`)

Monthly (or weekly, or per-season) periods, expanding (or rolling) window. Each period trains only
on games before it, fits preprocessing and hyperparameters on the fit window, fits the calibrator
on the later calibration window, and predicts from inputs with outcome columns removed. Outcomes
are joined only after all predictions are frozen. Development covers 2016-17 to 2024-25 (the first
two seasons are training only). The **final test** (from 2025-10-01) is evaluated by `final-test`
with the configuration fixed in advance. Each run is logged to `artifacts/final_test_log.jsonl` and
warns if the holdout has been used before.

### Results on real data (generated 2026-10-26)

Development walk-forward: 10,749 games, monthly retraining, 71 periods. 95% CIs resample whole dates.

| model | log loss [95% CI] | Brier | accuracy | AUC | ECE | cal. slope |
|---|---|---|---|---|---|---|
| **Logistic, uncalibrated (champion)** | **0.6237** [0.6171, 0.6312] | 0.2171 | 0.651 | 0.698 | 0.012 | 0.92 |
| Logistic, Platt-calibrated | 0.6264 [0.6198, 0.6334] | 0.2180 | 0.650 | 0.696 | 0.010 | 0.94 |
| Elo (tuned in-window) | 0.6265 [0.6197, 0.6337] | 0.2185 | 0.647 | 0.695 | 0.024 | 0.93 |
| Ensemble | 0.6267 [0.6202, 0.6335] | 0.2183 | 0.648 | 0.695 | 0.010 | 0.96 |
| Boosting, uncalibrated | 0.6284 [0.6213, 0.6361] | 0.2190 | 0.651 | 0.693 | 0.020 | 0.87 |
| Home-win base rate | 0.6854 [0.6827, 0.6882] | 0.2462 | 0.563 | 0.510 | 0.014 | n/a |

The champion's paired log-loss edge over Elo is -0.0028 [-0.0046, -0.0009]: real, but small. It is
worse than Elo in the first 5 games of a season (0.659 vs 0.653, n=711) and when training history
is short (2016-17). Platt calibration did not help, because the models were already well calibrated.

**Final holdout** (2025-26 plus 2026-27 through 10-25; 1,273 games; run once with the champion fixed
in advance):

| model | log loss [95% CI] | Brier | accuracy | AUC |
|---|---|---|---|---|
| **Logistic, uncalibrated (champion)** | **0.5964** [0.5733, 0.6184] | 0.2047 | 0.689 | 0.740 |
| Boosting, uncalibrated | 0.5972 [0.5748, 0.6199] | 0.2048 | 0.694 | 0.741 |
| Elo | 0.5986 [0.5754, 0.6205] | 0.2057 | 0.687 | 0.740 |
| Home-win base rate | 0.6883 [0.6810, 0.6952] | 0.2476 | 0.551 | 0.512 |

On the holdout, champion minus Elo is -0.0022 [-0.0077, +0.0033]: the same direction as development,
but not significant on one season. **No market comparison or profitability claim is possible yet:**
there is no archived, timestamped odds history. Paper trading refuses untimestamped or late prices
and reports `no_valid_odds`.

### Fair odds and paper trading

`odds.fair_odds_quote(p, home_decimal, away_decimal)` returns fair (zero-margin) model prices, the
book's prices, margin, break-even and no-vig probabilities, edge and EV, each labeled. For example,
p = 0.60 gives a fair line of -150/+150. Against a -130/+110 book it shows a 4.1% margin, a 54.3% no-vig market price and a
+6.2% EV estimate, which is an estimate and not a promise. `backtest.paper_trade` classifies every
game (`bet`, `below_threshold`, `no_odds`, `no_odds_timestamp`, `odds_after_decision`). It runs
flat-stake and capped fractional-Kelly simulations with per-day exposure caps, ROI CIs, drawdown and
CLV (from `odds_join.attach_odds_with_close`), plus a threshold sweep with a Bonferroni-widened
interval as an overfitting check.

### Prediction log and feedback (`nbamodel/store.py`, SQLite at `data/predictions.db` or `$MONEYLINES_DB`)

`predict` appends each prediction with its timestamp, model version, feature snapshot, both
probabilities, fair odds, market odds where available, model spread and key factors. Rows are never
updated. `update-results` matches final results and scores each prediction, excluding any made at or
after tip-off. It then snapshots metrics (all games, last 100, last 300) and team Elo, and flags
degradation when recent log loss exceeds the walk-forward reference by more than 2 standard errors
(at least 100 games). Retraining happens only on the configured cadence (7 days, when new results
exist) or on degradation, and always as a full refit with the validated configuration. Single
results never adjust weights.

### Dashboard

`out/dashboard/index.html` is a self-contained page (light and dark) with: model overview, upcoming
games with fair lines and data freshness, the live prediction log, walk-forward and final-test
comparisons with calibration and season charts, paper trading (labeled SIMULATED), feature
importance, metrics history, alerts, retraining history and the model registry.

### Known limitations

- No timestamped historical odds, so no market benchmark and no valid profitability backtest. Start
  `scripts/collect_odds.py` on a schedule now. A paid Odds API plan can backfill from June 2020.
- No injury or lineup data. This is likely the largest missing signal, especially near the market.
- Neutral-site games before 2024-25 are still unflagged. The schedule endpoint doesn't mark them either.
- Pre-2017 rows have no tip-off time, so they use a nominal 09:00 ET prediction time.
- Uncertainty for a single game is shown as model disagreement (spread), not a calibrated interval.
- Elo features use the default config. Only the Elo *baseline* is tuned per window.
- Logistic coefficients on correlated strength features (e.g. Elo vs adjusted margin) are not
  individually interpretable.

## Layout

```
nba_model.py              CLI entry point
nbamodel/
  data.py                 CSV loading and validation, data-quality report
  elo.py                  online Elo ratings
  features.py             leak-free pregame features
  pipeline.py             split, model selection, holdout evaluation, upcoming predictions
  evaluation.py           log loss, Brier, clustered bootstrap, calibration
  odds.py                 odds conversion, vig removal, EV, Kelly
  backtest.py             paper-trading simulation
  sources.py              provider format converters (nba_api game logs and schedule)
  odds_api.py             The Odds API client (current and historical)
  odds_join.py            attach the latest pre-tip-off quote to each game
  sample.py               deterministic synthetic data generator
  dataset.py              Phase 2 training dataset with timestamps and timing validation
  leakage.py              perturbation / label-correlation look-ahead checks
  models.py               Elo, logistic, gradient boosting, calibrators, ensemble
  walkforward.py          chronological walk-forward backtest
  report.py, plots.py     metrics, champion choice, charts
  production.py           production training and upcoming predictions
  registry.py             versioned model artifacts
  store.py                SQLite prediction log, result reconciliation, drift, retraining triggers
  availability.py         timestamped injury-report adapter (not in default features)
  dashboard.py            static HTML dashboard
  cli.py, __main__.py     `python -m nbamodel ...`
config/phase2.json        Phase 2 settings
Makefile                  reproducible commands
scripts/                  sample data, nba_api fetch, schedule update, odds collection
tests/                    pytest suite
data/sample/              synthetic fixture
```

## Roadmap

- **Done in Phase 2:** walk-forward validation, in-window Elo tuning, boosting, calibration,
  prediction log, drift checks, dashboard.
- **Next:** archive timestamped odds daily, then a market benchmark, CLV and a valid paper-trading backtest.
- Timestamped injury-report ingestion, through `availability.py`.
- Travel / time-zone features; a better early-season model (blend toward Elo when few games are played).
- FastAPI + interactive frontend; PostgreSQL if the SQLite log outgrows a single machine.
