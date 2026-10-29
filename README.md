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
| [ESPN public API](https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard) | live results, team and player box scores (hosted collector) | Free, unofficial and undocumented; it can change without notice. Personal, non-commercial use; requests are spaced out. |
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

- No timestamped historical odds before 2026-27 (see Phase 3: live collection started 2026-10-27).
  A paid Odds API plan can backfill from June 2020 (`scripts/collect_odds.py --historical`).
- No injury or lineup data. This is likely the largest missing signal, especially near the market.
- Neutral-site games before 2024-25 are still unflagged. The schedule endpoint doesn't mark them either.
- Pre-2017 rows have no tip-off time, so they use a nominal 09:00 ET prediction time.
- Uncertainty for a single game is shown as model disagreement (spread), not a calibrated interval.
- Elo features use the default config. Only the Elo *baseline* is tuned per window.
- Logistic coefficients on correlated strength features (e.g. Elo vs adjusted margin) are not
  individually interpretable.

## Phase 3: odds archive, market benchmark, Elo blending, paper trading

Paper trading only: nothing in this project places or can place a real bet.

### Odds snapshot collection (`odds_collect.py`, `odds_store.py`, `odds_schedule.py`)

- **Provider:** The Odds API, current NBA moneylines (`h2h`, `regions=us`), requested in American
  format so quoted prices are stored exactly. One call returns every listed NBA game across about
  9 US books and costs **1 credit**. The free plan gives 500 credits a month. Put
  `ODDS_API_KEY=...` in `.env` (gitignored).
- **Storage:** append-only `odds_snapshots` table in `data/predictions.db`. Each row has the
  provider event id, NBA `game_key`, book, home/away American and decimal prices, the provider's
  scheduled start, the NBA tip-off, `observed_at` (when the price was seen), `provider_ts` (the
  book's last update) and `retrieved_at`. Every observation is kept; nothing is overwritten.
  Exact repeats (same event, book and observation time) are ignored.
- **Matching:** provider events map to NBA games by team pair, with the provider's start time
  within 18 hours of the NBA tip-off. Reversed home/away listings are re-oriented and flagged
  `flipped`. Two candidate games is `ambiguous` and no candidate is `unmatched`; neither is used.
  `update-results` re-matches snapshots stored before their game reached the schedule.
- **Validation:** rejects missing prices, American odds with |x| < 100, decimal odds ≤ 1, margins
  outside −2% to 25%, provider timestamps after the observation, and unknown teams. Prices
  observed at or after tip-off are stored but flagged `is_pregame = 0` and never read.
- **Reliability:** retries rate limits (429), 5xx and network errors with exponential backoff
  (2, 4, 8, 16 s). Auth and quota errors fail at once. Every run, including failures and skips, is
  logged in `odds_collection_runs` with counts and remaining credits.
- **Schedule:** targets are 24 h, 6 h, 1 h and 15 min before each tip-off. Run
  `scripts/collect_odds_cron.sh` from cron every 5 minutes. A credit is spent only when a target's
  time has arrived and no call has covered its tolerance window yet, so one call can serve every
  game sharing a tip-off. Stored times are actual observation times; targets are only intentions,
  and `odds-status` shows which offsets were hit or missed for each game.
- **Budget guard:** before each call, projected use to month end is checked against the remaining
  credits (less a 25-credit reserve). Offsets are dropped in the order 24 h, 6 h, 1 h, 15 min.
- **Credit estimate** (`make odds-plan`, 2026-27 schedule from 2026-10-27, 5-minute polling):

  | offsets | calls in the rest of the season | per 30 days | busiest month |
  |---|---|---|---|
  | 24h, 6h, 1h, 15m | 1,997 | 360 | 431 (Jan) |
  | 6h, 1h, 15m | 1,962 | 354 | 426 |
  | 1h, 15m | 1,482 | 267 | 324 |
  | 15m | 741 | 134 | 162 |

  The full schedule fits the free plan. The 24 h target adds few calls because it usually
  coincides with the previous day's slots. If the machine is asleep, targets are missed and
  reported as missed, never back-filled.

### Market benchmark (`market.py`)

No-vig method: proportional normalisation. Raw implied probability is `q = 1/decimal`, the margin
is `q_home + q_away − 1`, and the no-vig probability is `q_side / (q_home + q_away)`. It ignores
favourite-longshot bias.

`predict` now attaches, to every prediction, each book's latest pregame price observed at or before
the prediction's own timestamp (at most 120 minutes old). It stores them in `prediction_odds`, and
also stores the Elo, logistic, boosting, ensemble and shadow-blend probabilities. `market-benchmark`
scores every model and the no-vig market on identical games and timestamps. It groups results by
horizon before tip-off (≤30 min, 30 min–2 h, 2–12 h, >12 h) and reports log loss, Brier, accuracy,
calibration and date-clustered 95% CIs on model-minus-market differences. Fewer than 500 games is
labelled preliminary. A price observed after its prediction is never compared.

### Early-season Elo blending (`blend.py`)

The blend is `P = w·P_Elo + (1−w)·P_ML`, with the champion `p_logistic_raw` as the ML model. There
are five methods: fixed w, w per games-played bucket, w per season-phase bucket, exponential decay
in games played, and early-only (w for the first 15 games, 0 after). Weights are chosen
season by season on earlier seasons only (forward-chaining over development walk-forward
predictions). The final holdout is used for one confirmation look, which is logged in
`artifacts/final_test_log.jsonl`.

Result (2026-10-27, 9,519 development games, seasons 2017–2024):

| model | log loss | diff vs logistic [95% CI] | first 15 games diff [95% CI] |
|---|---|---|---|
| logistic (base) | 0.6229 | | |
| Elo alone | 0.6267 | +0.0038 [+0.0018, +0.0058] | −0.0000 [−0.0059, +0.0064] |
| blend: fixed (chosen w fell from 0.85 to 0.35 as training history grew) | 0.6235 | +0.0006 [−0.0006, +0.0018] | −0.0025 [−0.0065, +0.0016] |
| blend: decay | 0.6230 | +0.0001 [−0.0010, +0.0012] | −0.0016 [−0.0061, +0.0033] |
| blend: early-only (w=0.6) | 0.6225 | −0.0003 [−0.0013, +0.0007] | −0.0017 [−0.0064, +0.0033] |

Early in the season, Elo alone is as good as the logistic model, and blends are slightly better on
average. No method's improvement is distinguishable from zero, and blending everywhere hurts
mid-season and late-season. The holdout confirmation for early-only is +0.0001 [−0.0018, +0.0019].
**Blending is not enabled.** `p_blend` is logged next to every live prediction as a shadow
probability, so it keeps being tested on new games. The pre-set rule enables it only if the best
method's overall CI is below 0 and it is not worse early in the season.

### Closing-line value (`clv.py`)

Price-based CLV is `decision decimal / closing decimal − 1` for the same book and side. The close
is that book's final pregame snapshot, observed after the decision price and within 30 minutes of
tip-off. Otherwise the status is `no_close_snapshot`, `no_later_snapshot` or `close_too_early`. If
the decision used the best price across books, the close is the best price across books, labelled
`best_vs_best_close`; there is no other cross-book substitution. The no-vig probability move from
decision to close is also stored. Positive CLV alone is not proof of profitability.

### Live paper trading (`paper.py`)

There is one immutable decision per game per strategy. It is made at the first logged prediction
30–90 minutes before tip-off, before tip-off only, using only the price recorded with that
prediction. The outcomes are `bet` or an abstention reason: `below_threshold`, `no_odds`,
`stale_odds`, `low_history` (optional), `model_disagreement` (spread > 0.20), `edge_too_large`
(|model − no-vig| > 0.10, likelier a missing-information error than value) or
`daily_exposure_cap`. There is at most one side per game. Stakes are reported two ways: flat
1 unit, and ¼-Kelly capped at 2% of bankroll, 5 units per bet (a stand-in for book limits) and
10% of bankroll per date. Kelly uses the bankroll from results settled before the decision.
`update-results` settles decisions, computes CLV, appends market metrics and runs missing-data
checks. Reports include ROI with a date-clustered CI, win rate vs break-even, max drawdown, bankroll
history, mean estimated edge and EV, and CLV by model, book and season. Results are flagged
preliminary below 300 settled bets.

The historical walk-forward and final-test paper trading (`backtest.py`) now read odds from the
archive (`paths.odds = "db"`) and report price-based CLV next to the no-vig version. It only gains
games as the archive grows. There is still no pre-2026-27 odds history, because the free plan has
no historical endpoint.

### Commands

```bash
scripts/install_odds_schedule.sh   # macOS LaunchAgent: collect every 5 min (--uninstall to remove)
make collect-odds      # one collection pass (or cron: */5 * * * * .../scripts/collect_odds_cron.sh)
make odds-plan         # credit estimate      make odds-status   # runs + per-game coverage
make blend-study       # Elo blending study   make market-benchmark
make paper-trade       # due paper decisions + SIMULATED ledger
python -m nbamodel import-odds --csv old_snapshots.csv   # bring a CSV archive into the database
```

The dashboard adds odds monitoring (latest prices by book, movement since the first snapshot,
missing-odds warnings, collection runs), model versus market, the live market benchmark, the
blending study and the live paper ledger. Sections are labelled LIVE DATA, LIVE PREDICTIONS,
HISTORICAL EVALUATION or SIMULATED.

### Phase 3 limitations

- Collection started 2026-10-27, so no game has a closing snapshot yet. Every model-versus-market,
  CLV and paper-trading number is empty or preliminary until a few hundred games accumulate.
- The free plan has no historical odds. Backtests before 2026-27 still have no market comparison.
- Observation times depend on cron running and the machine being awake; missed targets are
  reported, not hidden.
- Books post lines about a day ahead, so the 24 h snapshot often has no prices for later games.
- The model has no injury or lineup information. Large model-versus-market gaps early in the
  season (e.g. 22 points on DET @ NYK, 2026-10-27) are why `edge_too_large` exists.
- Proportional de-vig is the only method implemented. Power and Shin methods would shift
  longshot probabilities slightly.

## Daily operation

Everything below is paper trading and research. Nothing places a real bet.

**Every game day (one command):**

```bash
make today
```

With hosted collection (below), `make today` pulls the collector's state and prints today's table
from it without writing anything (`today --no-refresh --no-log`). When this Mac is the collector,
use `make today-local`, which runs four steps:
1. Refreshes results and the schedule from nba_api.
2. Reconciles finished games: adds results and scores, updates accuracy, Brier and log loss, settles
   paper decisions and computes CLV.
3. Reuses archived odds if they are under 30 minutes old; otherwise it spends 1 Odds API credit.
4. Predicts today's not-yet-started games with the active model and saves each prediction with its
   model version, timestamp, feature snapshot and every book price available at that moment. Then it
   prints the table: predicted winner, win probability, fair odds, book odds, no-vig probabilities,
   edge, expected ROI, whether the pick differs from the book favourite, and flags.

Variants: `python -m nbamodel today --date 2026-10-28` for another date, `--no-refresh` to never
call the API, `--markdown` for a Markdown table.

**Automatic (once installed with `scripts/install_odds_schedule.sh`):** the LaunchAgent runs
`scripts/collect_odds_cron.sh` every 5 minutes. It takes odds snapshots 24 h, 6 h, 1 h and 15 min
before tip-off and logs predictions and paper decisions with each one. Once a day after 10:00 UTC
it runs data-update, update-results (with a retrain when the 7-day cadence is due) and dashboard.
Logs go to `logs/odds_collect.log` and `logs/reconcile.log`.

**Weekly:**

```bash
make weekly-report     # out/reports/weekly_<date>.md
```

The report covers accuracy, Brier, log loss, calibration bins, model vs market by horizon, paper
trading (win rate vs break-even, ROI with CI, drawdown, CLV) and data-collection health.

**Reconciliation never edits a prediction.** Results, scores, paper results and CLV go into separate
tables (`results`, `evaluations`, `paper_results`, `clv_results`). The original probability, feature
snapshot and recorded prices stay as they were written.

**Limits of the local schedule:** a LaunchAgent runs only while you are logged in and the Mac is
awake. It does not run while the computer is asleep or off. Missed snapshot targets are recorded as
missed. After a wake, the next 5-minute run catches up on reconciliation, but missed odds cannot be
recovered. To keep collecting with the Mac off, run the same `scripts/collect_odds_cron.sh` on an
always-on machine (a small VPS or a Raspberry Pi) with the repo, `.venv`, `.env` and the database.

### Hosted collection (GitHub Actions)

So collection keeps running while the Mac sleeps, `.github/workflows/collector.yml` runs
everything on GitHub's runners. The Mac has no scheduled job.

- **Collector:** started every 30 minutes, one run at a time (the next run waits in the queue).
  Each run (`scripts/actions_collector.sh`) restores the state, then every 5 minutes it ingests new
  injury reports, runs `collect-odds --then-predict`, and pushes the state when anything changed.
  It ends after 5h40m, or early when no snapshot target opens within 6 hours (`python -m nbamodel
  odds-next`). GitHub often starts scheduled runs late, so a run that is already polling covers the
  15-minute targets.
- **Results from ESPN:** once a day after 10:00 UTC (retried at most hourly), `results-update`
  takes final scores, team box stats, tip-off times for the next 7 days and player box scores
  from ESPN's public API (`nbamodel/espn.py`). Then `update-results --retrain-if-due` reconciles.
  stats.nba.com does not answer GitHub's runners and timed out from home connections on
  2026-10-28/29. ESPN answers both.
  - Cross-checked on 2026-10-29 against stats.nba.com for all 56 games from 2026-10-20 to
    2026-10-27: scores, FGM, FGA, 3PM, FTA, OREB, DREB and turnovers all match.
  - Every game must pass validation (player points add up to the team score) before it is scored.
  - A score already in the games file is never overwritten; a mismatch is logged.
  - ESPN's API is unofficial and undocumented. It sometimes serves empty responses, which are
    retried with different query parameters.
- **State is encrypted:** the repo is public, so the `state` branch (database, games file, models,
  walk-forward report, upcoming predictions, `logs/actions/`) only holds AES-256 ciphertext
  (`scripts/state_sync.sh`). Each push replaces the branch with a single commit. Each run that
  reconciles also uploads the encrypted state as a 14-day artifact. Only the collector writes
  `state`. The older `results` branch (the games file the Mac pushed in the stats.nba.com era) is
  only read once, to bootstrap a state that has no games file.
- **Read-only on the Mac:** `make pull` replaces `data/predictions.db`, the games file,
  `artifacts/` and `logs/actions/` with the collector's copy and rebuilds the dashboard. Disconnect
  DBCode first; the script refuses while the file is open. Anything written locally is overwritten
  on the next pull.

One-time setup:

```bash
openssl rand -base64 32                  # put it in .env as STATE_KEY=...
gh secret set STATE_KEY                  # paste the same value
gh secret set ODDS_API_KEY               # paste the Odds API key
scripts/install_odds_schedule.sh --uninstall   # stop local odds collection, if it was installed
make seed-state                          # upload this Mac's database, games file and models
gh workflow run collector -f check_only=true   # restore + status only; then let the schedule run
```

The retired Mac results job is removed with `scripts/install_results_schedule.sh --uninstall`. To
go back to local collection: `gh workflow disable collector`, `make pull`, then
`scripts/install_odds_schedule.sh`. GitHub disables scheduled workflows after 60 days without
repository activity, and the collector's pushes may not count. If the schedule gets disabled,
re-enable it in the Actions tab.

## Phase 4: injury-aware predictions (in progress)

Built in milestones. Done:
- **1.** official injury report ingestion
- **2a.** ESPN results and current-season player box scores (see Hosted collection)
- **2b.** the player box-score history backfill
- **3 (core).** player-impact features, computed but not yet fed to a model
- **5/6.** the risk-adjusted opportunity ranking in `make today`

Next: **4.** the injury-aware model comparison with calibration, then **7.** the selection backtest
on live prices. Until milestone 4, the model's probabilities do not use injuries; the report uses
injuries only as risk controls.

### Official injury reports (`nbamodel/injuries.py`)

**Source.** The league's official reports are published as PDFs at
https://official.nba.com/nba-injury-report-2026-27-season/. Since 2026-27 they come out every 15
minutes on game days; before that the files are hourly. The archive starts in the 2019-20 season.
Both your Mac and GitHub's runners can download them (tested 2026-10-28). They are free: no API
credits are used.

**Timestamps.** Each report's page header gives its real publication time (US/Eastern), and that is
the time stored. The file name can be off: `Injury-Report_2023-01-10_05PM.pdf` says "05:30 PM" in its
header. Retrieval time, URL and the PDF's sha256 are stored too.

**Storage is append-only** (`injury_report_files`, `injury_reports` in `data/predictions.db`):
- Every URL tried gets a row, including missing ones.
- Player rows are a change log. A row is written when a player's status or reason changes, or as
  `not_listed` when a report still lists the game but drops the player.
- `injuries.statuses_asof(con, t)` replays the log, so the contents of any report published at or
  before `t` can be rebuilt exactly. A report published after `t` is never visible.
- One night of 15-minute reports is about 130 rows.

**Rules the readers follow:**
- A player who was never listed is unknown, not available.
- "NOT YET SUBMITTED" is kept as a team-level `not_submitted` status.
- Report freshness comes from the newest report file, which is not always the newest change.
- Box scores and final participation are never substituted for a pregame report.

**Parsing.** `pdfplumber` gives word positions, and the columns come from the header row on page 1
(later pages have no header). Team, matchup, date and time appear only on a group's first row, so
they are carried forward. A reason that wraps is printed above and below its player's row, and is
joined back to that player. The tests use word fixtures captured from a 2023 report and a 2026
report.

**Collection:**
- Live: every hosted collector pass runs `injuries-refresh`, which ingests any of today's reports
  that are new. The season page only links the current day's reports.
- History: `.github/workflows/injury-backfill.yml` (start it from the Actions tab or with
  `gh workflow run injury-backfill`). It fills `data/injury_history.db` from 2019-20 onward,
  oldest first. For each game day it tries the hourly files (or :00 and :30 in the 15-minute era)
  from 11 AM ET until an hour before the last tip-off. That's the latest report a pregame
  prediction can use. Requests are 1 second apart. Each run saves its progress every 30 minutes to
  the encrypted `injury-history` branch and starts the next run until the archive is complete.
  Expect several runs. The history database is kept separate from the collector's state because it
  is about 40 MB and rarely changes.

```bash
python -m nbamodel injuries-refresh                     # new reports for today -> data/predictions.db
python -m nbamodel injuries-status [--date 2026-10-28]  # latest statuses + report age + teams not submitted
python -m nbamodel injuries-backfill --db data/injury_history.db --start 2019-10-22 [--end ...] [--max-seconds N]
scripts/state_sync.sh pull-history                      # local copy of the backfilled archive
```

**Limitations:**
- Only official statuses are covered: no beat-reporter news, no confirmed starting lineups.
- A late scratch after the last report before the decision time is not seen.
- Reports from before 2019-20 do not exist in the archive, so injury-aware backtests cover about
  half of the 14,403 training games.
- A team's first report of the day often says "NOT YET SUBMITTED".
- The 15-minute era's first files of the day can be missing.

### Player box-score history (`espn.backfill`)

`.github/workflows/player-backfill.yml` fills `data/player_history.db` (table `player_games`)
with ESPN box scores for every played game in the games file from 2014-15 to 2025-26. It runs on
the same schedule pattern as the injury backfill: manual start, progress saved every 30 minutes
to the encrypted `player-history` branch, and the next run started automatically. It makes about
17,000 requests (one scoreboard per date, one box score per game), spaced 1 second apart, so it
takes several runs.

- Each finished date is logged in `espn_fetch_log`, so reruns resume where the last run stopped.
- A game whose player points don't add up to the final score is recorded in `espn_invalid` and not
  stored. Before that, it is fetched once more with different query parameters.
- **Gap:** ESPN has no box score at all for some older games. In January 2015 about 1 in 6 games
  lists every player with no minutes or points. Those are recorded as "ESPN has no player stats
  for this game". Player-impact features treat them as missing; nothing is filled in.
- The current season's box scores come from the collector's daily reconcile and go into
  `data/predictions.db`.

```bash
gh workflow run player-backfill                     # 2014-10-28 .. 2026-06-30
scripts/state_sync.sh pull-history player           # local copy
python -m nbamodel players-backfill --db data/player_history.db --start ... --end ... [--max-seconds N]
```

### Player-impact features (`nbamodel/player_impact.py`)

Each feature is the change in a team's expected margin versus its recent form, given the injury
report at prediction time. In points: `impact = sum_i (r_i - r_rep) * (P_i * healthy_i - recent_i) / 48`.
- `r_i` is the player's rating: game-level regularized plus-minus. Ridge shrinks each player toward
  a prior based on his minutes per game, so the deep bench is not pulled up to league average.
  Ratings are refit every Monday from earlier games only.
- `r_rep` is replacement level: the rating of low-minute players.
- `healthy_i` is the player's average minutes over his last 10 games played. `recent_i` is his
  average minutes over his team's last 10 games, counting 0 when he didn't play. That 10-game
  window is the one the team features use.
- `P_i` is P(plays | report status), measured on earlier seasons by matching statuses to box
  scores. A player who was never listed counts as `unlisted`, which is also a measured rate.

Because the feature is a change versus recent form, a long absence counts about 0: it is already in
the team's numbers. A newly absent or returning player moves it. `impact_sd` carries the
uncertainty from questionable players. Games with no report before prediction time get NaN, never
"healthy".

### Opportunity ranking (`nbamodel/opportunities.py`, `make today` sections B to D)

For each game, `make today` takes the side with the better expected value at the latest price and
computes:
- `expected ROI = p x decimal_odds - 1`
- `conservative ROI = p_lower x decimal_odds - 1`

`p_lower` is `p` minus the model's measured calibration shortfall for similar predictions. In each
probability band (early season and later season separately) it is the gap between the average
prediction and an 80% one-sided lower bound on how often those picks actually won. The table
(`config/uncertainty.json`, built by `python -m nbamodel uncertainty-fit`) uses development
walk-forward predictions only: 10,749 games from 2016-10 to 2025-04, never the final holdout.
It shows mild overconfidence on big favourites (85%+ predicted, 85.6% observed) and a lot of it
early in the season (75 to 85% predicted, shortfall of 9 to 13 points). That is why a 70 to 80%
pick is not treated as low-risk.

Only games that pass every check are ranked, by conservative ROI. The market price is used only to
price the bet; it is never a model input. Every other game is listed under GAMES TO AVOID with all
of its reasons. When nothing qualifies, the report says **NO QUALIFYING OPPORTUNITIES TODAY**.

Settings (`config/phase2.json` -> `opportunities`). None of these were tuned on the final holdout.

| Setting | Default | Meaning |
|---|---|---|
| `min_edge` | 0.02 | model minus no-vig probability for the selected side |
| `min_expected_roi` | 0.02 | same as the paper ledger's `ev_threshold` |
| `min_conservative_roi` | 0.0 | `p_lower x odds - 1` must be non-negative |
| `max_abs_edge` | 0.10 | Phase 3's `edge_too_large` abstention, kept: a bigger gap usually means missing information |
| `max_odds_age_minutes` | 60 | stale price |
| `max_injury_report_age_minutes` | 180 | stale injury report |
| `max_model_spread` | 0.20 | the model's components disagree |
| `early_season_games`, `early_season_extra_roi` | 10, 0.02 | either team under 10 games needs 2 more points of conservative ROI |
| `rotation_minutes` | 20 | important player: at least 20 healthy minutes this season |
| `block_unresolved_injuries` | true | a rotation player questionable or doubtful: not eligible |
| `block_missing_injury_report` | true | no report yet, a team not submitted, or no box scores to judge importance: not eligible |
| `kelly_multiplier`, `max_stake_fraction` | 0.25, 0.02 | paper stake: quarter Kelly on `p_lower`, at most 2% of bankroll |
| `max_daily_exposure` | 0.10 | at most 10% of bankroll across the day's selections |
| `max_selections_per_team` | 1 | correlated exposure: one selection per team per day |
| `validated_edge` | false | while false, the summary warns that no edge has been shown |

API costs: none new. The ranking reads the odds archive. `make today` never calls The Odds API;
it runs `--no-refresh --no-log`.

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
  odds_store.py           Phase 3 append-only odds snapshot archive: validation, event matching
  odds_schedule.py        snapshot targets, credit simulator, budget guard
  odds_collect.py         one collection pass (provider -> validate -> store -> run log)
  market.py               no-vig benchmark at matching prediction times
  blend.py                early-season Elo blending study
  clv.py                  price-based closing-line value
  paper.py                live paper-trading ledger (decisions, settlement, bankroll)
  injuries.py             official NBA injury reports: fetch, parse, append-only change log
  espn.py                 ESPN results, tip-off times and player box scores (validated)
  player_impact.py        player ratings, minutes, availability probabilities, injury features
  opportunities.py        conservative probabilities, risk controls, ranking, no-bet outcome
  cli.py, __main__.py     `python -m nbamodel ...`
config/phase2.json        settings (Phase 3 adds the odds, paper and blend sections)
Makefile                  reproducible commands
scripts/                  sample data, nba_api fetch, schedule update, odds collection
.github/workflows/        hosted odds collector (GitHub Actions)
tests/                    pytest suite
data/sample/              synthetic fixture
```

## Roadmap

- **Done in Phase 2:** walk-forward validation, in-window Elo tuning, boosting, calibration,
  prediction log, drift checks, dashboard.
- **Done in Phase 3:** odds archive and scheduler, market benchmark, Elo blending study (not enabled),
  price CLV, live paper ledger.
- **Next:** accumulate about 300 games of snapshots, then read the first model-versus-market and CLV results.
- Timestamped injury-report ingestion, through `availability.py`.
- Travel / time-zone features; preseason priors (roster changes) for the early season, where blending alone did not help.
- FastAPI + interactive frontend; PostgreSQL if the SQLite log outgrows a single machine.
