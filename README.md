# NBA Moneyline Model: Phase 1

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
python3 -m venv .venv
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
scripts/                  sample data, nba_api fetch, schedule update, odds collection
tests/                    pytest suite
data/sample/              synthetic fixture
```

## Roadmap

- **Phase 2:** walk-forward (expanding window) validation per season, Elo hyperparameter search on validation, injury-adjusted strength, travel.
- **Phase 3:** XGBoost/LightGBM, isotonic/sigmoid calibration fitted on a separate calibration split, Optuna, MLflow.
- **Phase 4:** a prediction log with closing-line value, scheduled odds collection, PostgreSQL. Timestamped odds ingestion is done.
- **Phase 5:** FastAPI and a Next.js dashboard, plus a paper-trading ledger.
- **Phase 6:** drift monitoring and champion/challenger retraining on untouched holdouts.
