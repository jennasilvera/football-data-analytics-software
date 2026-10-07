PYTHON = python
RESULTS = data/sample/historical_results_sample.csv
HOME_TEAM = Argentina
AWAY_TEAM = France
N_SIMULATIONS = 500
HOME_ODDS = 2.20
DRAW_ODDS = 3.40
AWAY_ODDS = 3.50
FINAL_HOME_SCORE = 1
FINAL_AWAY_SCORE = 1
CLOSING_HOME_ODDS = 2.10
CLOSING_DRAW_ODDS = 3.25
CLOSING_AWAY_ODDS = 3.60
MARKET_ODDS_PATH = data/sample/market_odds_sample.csv
SETTLEMENT_RESULTS_PATH = data/sample/settlement_results_sample.csv
STRATEGY_POLICY_PATH = outputs/strategy_policy_edges.csv
STAKE_SIZING_PATH = outputs/stake_sizing_edges.csv

.PHONY: install lint test check health ingest build-elo build-features backtest report-backtest poisson report-match simulate-group-stage report-group-stage demo clean

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

lint:
	ruff check .

test:
	pytest

check: lint test

.PHONY: demo-v2 demo-v2-comparison
V2_MODEL ?= logistic
V2_EXTRA_ARGS ?=
demo-v2:
	$(PYTHON) -m football_analytics research \
		--results data/sample/v2/results.csv \
		--teams data/sample/v2/teams.json \
		--competitions data/sample/v2/competitions.json \
		--source-id synthetic-v2 \
		--ingested-at 2020-06-01T00:00:00Z \
		--assert-senior-mens-a \
		--cutoff 2020-03-01T00:00:00Z \
		--cutoff 2020-04-01T00:00:00Z \
		--evaluation-days 30 --min-train 12 --calibration-bins 5 \
		--model $(V2_MODEL) $(V2_EXTRA_ARGS)

demo-v2-comparison:
	$(MAKE) demo-v2 V2_MODEL=all

health:
	$(PYTHON) -m wc_forecast health

ingest:
	$(PYTHON) -m wc_forecast ingest-results $(RESULTS)

build-elo:
	$(PYTHON) -m wc_forecast build-elo

build-features:
	$(PYTHON) -m wc_forecast build-features

backtest:
	$(PYTHON) -m wc_forecast backtest-logistic

report-backtest:
	$(PYTHON) -m wc_forecast report-backtest

poisson:
	$(PYTHON) -m wc_forecast predict-poisson $(HOME_TEAM) $(AWAY_TEAM)

report-match:
	$(PYTHON) -m wc_forecast report-match $(HOME_TEAM) $(AWAY_TEAM)

market-edge:
	$(PYTHON) -m wc_forecast evaluate-market $(HOME_TEAM) $(AWAY_TEAM) --home-odds $(HOME_ODDS) --draw-odds $(DRAW_ODDS) --away-odds $(AWAY_ODDS)

batch-market:
	$(PYTHON) -m wc_forecast batch-evaluate-market $(MARKET_ODDS_PATH)

strategy-policy:
	$(PYTHON) -m wc_forecast apply-strategy-policy outputs/batch_market_edges.csv --output $(STRATEGY_POLICY_PATH)

stake-sizing:
	$(PYTHON) -m wc_forecast size-stakes $(STRATEGY_POLICY_PATH) --output $(STAKE_SIZING_PATH)

log-batch-predictions:
	$(PYTHON) -m wc_forecast log-batch-predictions $(STAKE_SIZING_PATH)

settle-batch-predictions:
	$(PYTHON) -m wc_forecast settle-batch-predictions $(SETTLEMENT_RESULTS_PATH)

log-prediction:
	$(PYTHON) -m wc_forecast log-prediction $(HOME_TEAM) $(AWAY_TEAM) --home-odds $(HOME_ODDS) --draw-odds $(DRAW_ODDS) --away-odds $(AWAY_ODDS)

settle-prediction:
	@PREDICTION_ID=$$(tail -n 1 outputs/prediction_ledger.csv | cut -d',' -f1); \
	$(PYTHON) -m wc_forecast settle-prediction "$$PREDICTION_ID" \
	  --final-home-score $(FINAL_HOME_SCORE) \
	  --final-away-score $(FINAL_AWAY_SCORE) \
	  --closing-home-odds $(CLOSING_HOME_ODDS) \
	  --closing-draw-odds $(CLOSING_DRAW_ODDS) \
	  --closing-away-odds $(CLOSING_AWAY_ODDS)

report-ledger:
	$(PYTHON) -m wc_forecast report-ledger

simulate-group-stage:
	$(PYTHON) -m wc_forecast simulate-group-stage --n-simulations $(N_SIMULATIONS)

report-group-stage:
	$(PYTHON) -m wc_forecast report-group-stage

demo: ingest build-elo build-features backtest report-backtest poisson report-match market-edge batch-market strategy-policy stake-sizing log-batch-predictions settle-batch-predictions log-prediction settle-prediction report-ledger simulate-group-stage report-group-stage
	@echo "Demo pipeline complete."

clean:
	rm -f outputs/*.csv
	rm -f reports/logistic_backtest_report.md
	rm -f reports/match_prediction_report.md
	rm -f reports/group_stage_simulation_report.md
	rm -f data/processed/*.csv

.PHONY: validate forecast-sample forecast-report forecast-audit forecast-artifacts forecast-workflow

validate:
	ruff check .
	pytest
	python -m wc_forecast health

forecast-sample:
	python -m wc_forecast ingest-world-cup-fixtures \
		data/sample/world_cup_2026_fixtures_sample.csv \
		--output data/processed/world_cup_2026_fixtures.csv
	python -m wc_forecast forecast-upcoming-fixtures \
		data/processed/world_cup_2026_fixtures.csv \
		--as-of 2026-06-20T00:00:00Z \
		--train-cutoff-date 2026-01-01 \
		--rating-cutoff-date 2026-06-19 \
		--output outputs/world_cup_2026_upcoming_forecasts.csv

forecast-report:
	python -m wc_forecast summarize-upcoming-forecasts \
		outputs/world_cup_2026_upcoming_forecasts.csv \
		--output outputs/world_cup_2026_upcoming_forecast_report.md

forecast-audit:
	python -m wc_forecast audit-upcoming-forecasts \
		outputs/world_cup_2026_upcoming_forecasts.csv \
		--output outputs/world_cup_2026_upcoming_forecast_audit.csv

forecast-artifacts:
	python -m wc_forecast list-forecast-artifacts \
		--output outputs/forecast_artifact_index.csv

forecast-workflow:
	python -m wc_forecast run-upcoming-world-cup-forecast \
		--from-date 2026-06-20 \
		--train-cutoff-date 2026-01-01 \
		--rating-cutoff-date 2026-06-19 \
		--output outputs/world_cup_2026_upcoming_forecasts.csv

.PHONY: demo-v2-nested
demo-v2-nested:
	$(MAKE) demo-v2 V2_MODEL=all V2_EXTRA_ARGS="--postprocess-days 21 --min-postprocess 5"
