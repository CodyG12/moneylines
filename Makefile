# Reproducible Phase 2 commands. Settings live in config/phase2.json.
PY ?= .venv/bin/python
SEASON ?= 2026-27
NBA = $(PY) -m nbamodel

.PHONY: test data-fetch data-update dataset walk-forward final-test train predict update-results retrain dashboard daily phase1 \
        collect-odds odds-plan odds-status blend-study market-benchmark paper-trade today today-local weekly-report \
        pull seed-state results-update

test:            ## full pytest suite
	$(PY) -m pytest -q

data-fetch:      ## download all historical seasons with box scores and tip-off times
	$(PY) scripts/fetch_nba_api.py --seasons 2014-15 2015-16 2016-17 2017-18 2018-19 2019-20 2020-21 \
	    2021-22 2022-23 2023-24 2024-25 2025-26 --out data/raw/nba_games.csv
	$(PY) scripts/update_schedule.py --games data/raw/nba_games.csv --season $(SEASON)

data-update:     ## refresh the current season (results, box scores, schedule)
	$(PY) scripts/update_schedule.py --games data/raw/nba_games.csv --season $(SEASON)

dataset:         ## leakage-checked training dataset -> data/processed/training_dataset.csv
	$(NBA) build-dataset

walk-forward:    ## development walk-forward backtest + champion selection -> out/walkforward/
	$(NBA) walk-forward

final-test:      ## ONE-TIME evaluation on the untouched final period -> out/final_test/
	$(NBA) final-test

train:           ## fit the champion on all completed games, version and activate it
	$(NBA) train

predict:         ## predict upcoming games, log them to data/predictions.db
	$(NBA) predict

update-results:  ## reconcile finished games, update metrics/Elo, check degradation
	$(NBA) update-results

retrain:         ## update results and retrain if scheduled or degraded
	$(NBA) update-results --retrain-if-due

dashboard:       ## static HTML dashboard -> out/dashboard/index.html
	$(NBA) dashboard

daily: data-update retrain predict dashboard  ## typical game-day routine

# ---- Phase 3: odds archive, market benchmark, blending, paper trading (see README) ----
collect-odds:    ## store an odds snapshot if a scheduled target is due (cron: scripts/collect_odds_cron.sh)
	$(NBA) collect-odds --then-predict

odds-plan:       ## API credit estimate for the snapshot schedule
	$(NBA) odds-plan

odds-status:     ## collection runs and per-game snapshot coverage
	$(NBA) odds-status

blend-study:     ## early-season Elo blending study (development data; one holdout confirmation)
	$(NBA) blend-study

market-benchmark: ## models vs no-vig market at matching prediction times
	$(NBA) market-benchmark

paper-trade:     ## due paper decisions + SIMULATED ledger report (never real bets)
	$(NBA) paper-trade

today:           ## game day (hosted collection): pull the collector's state, show today's predictions, write nothing
	$(MAKE) -s pull
	$(NBA) today --no-refresh --no-log

today-local:     ## game day when this Mac is the collector: refresh results, reconcile, refresh stale odds, predict
	$(MAKE) -s data-update
	$(NBA) update-results
	$(NBA) today

# ---- Hosted collection: GitHub Actions collects odds, injuries and ESPN results (see README) ----
pull:            ## read-only local copy of the collector's state (database, models, logs) + dashboard
	scripts/state_sync.sh pull-state
	$(NBA) dashboard

seed-state:      ## ONE-TIME: upload this Mac's database, games file and models as the collector's starting state
	scripts/state_sync.sh push-state

results-update:  ## scores, tip-off times and player box scores from ESPN (the collector runs this daily)
	$(NBA) results-update

weekly-report:   ## last 7 days: accuracy, Brier, log loss, calibration, market, paper -> out/reports/
	$(NBA) weekly-report

phase1:          ## original Phase 1 pipeline (unchanged)
	$(PY) nba_model.py --games data/raw/nba_games.csv --output out/nba/predictions.csv --upcoming-days 3
