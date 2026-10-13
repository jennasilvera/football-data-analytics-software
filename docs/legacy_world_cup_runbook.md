# Daily Forecast Runbook

This runbook describes how to operate the World Cup Forecasting Engine for a daily upcoming-match forecast cycle.

The workflow is designed for an analyst, trading-operations-style support user, data engineer, or model reviewer who wants to update fixtures, run forecasts, review outputs, and check model warnings.

## Purpose

The daily forecast process produces:

- Upcoming fixture probability forecasts
- A human-readable Markdown forecast report
- A compact forecast audit file
- An artifact index showing generated outputs

The process is intended for analytical and educational use only. It is not betting advice.

## Daily Inputs

The expected raw fixture file is:

    data/raw/world_cup_2026_fixtures.csv

At minimum, it should contain:

- `date`
- `home_team`
- `away_team`

Optional columns:

- `tournament`
- `neutral`
- `status`

If optional fields are missing, the fixture ingestion command defaults to:

- `tournament = FIFA World Cup`
- `neutral = true`
- `status = Scheduled`

## Main Command

Run the full workflow with:

    python -m wc_forecast run-upcoming-world-cup-forecast \
      --from-date 2026-06-20 \
      --train-cutoff-date 2026-01-01 \
      --rating-cutoff-date 2026-06-19 \
      --output outputs/world_cup_2026_upcoming_forecasts.csv

This command performs:

1. Fixture ingestion
2. Feature rebuilding
3. Upcoming fixture forecasting
4. Forecast report generation
5. Forecast audit generation
6. Artifact indexing

## Expected Outputs

The main outputs are:

- `outputs/world_cup_2026_upcoming_forecasts.csv`
- `outputs/world_cup_2026_upcoming_forecast_report.md`
- `outputs/world_cup_2026_upcoming_forecast_audit.csv`
- `outputs/forecast_artifact_index.csv`

Supporting validation outputs may include:

- `outputs/rolling_backtest_metrics.csv`
- `outputs/feature_ablation_results.csv`

## Review Checklist

After running the workflow, review these items:

### 1. Forecast CSV

Open:

    outputs/world_cup_2026_upcoming_forecasts.csv

Check:

- Every expected upcoming fixture appears.
- Completed fixtures are excluded.
- TBD knockout placeholders are excluded unless explicitly included.
- Probabilities sum to approximately 1.0.
- `predicted_winner` matches the highest outcome probability.

### 2. Forecast Report

Open:

    outputs/world_cup_2026_upcoming_forecast_report.md

Review:

- Highest-confidence forecasts
- Most uncertain matches
- Potential upset watch
- Rating warnings

Use this file for presentation or project walkthrough purposes.

### 3. Forecast Audit

Open:

    outputs/world_cup_2026_upcoming_forecast_audit.csv

Review:

- `forecast_count`
- `average_confidence`
- `highest_confidence_match`
- `lowest_confidence_match`
- `rating_warning_count`
- `alias_lookup_count`
- predicted outcome distribution

A high warning count or unexpected alias count should trigger review.

### 4. Artifact Index

Open:

    outputs/forecast_artifact_index.csv

Check that expected artifacts exist and were recently modified.

## Common Issues

### Missing Fixture File

If the full processed fixture file does not exist, the CLI will tell you to create it with:

    python -m wc_forecast ingest-world-cup-fixtures \
      data/raw/world_cup_2026_fixtures.csv \
      --output data/processed/world_cup_2026_fixtures.csv

### Missing Raw Fixture File

Create or update:

    data/raw/world_cup_2026_fixtures.csv

Then rerun the workflow.

### Unexpected Team Rating Warning

A rating warning may occur when a team is missing from historical results or requires alias handling.

Review:

- Team spelling
- Country naming conventions
- Alias mappings
- Whether the team exists in the historical results data

### Too Few Forecasts

If fewer matches appear than expected, check:

- `--from-date`
- `--through-date`
- fixture `status`
- whether knockout matches still contain `TBD`

## Useful Commands

Run only fixture ingestion:

    python -m wc_forecast ingest-world-cup-fixtures \
      data/raw/world_cup_2026_fixtures.csv \
      --output data/processed/world_cup_2026_fixtures.csv

Run only upcoming forecasts:

    python -m wc_forecast forecast-upcoming-fixtures \
      data/processed/world_cup_2026_fixtures.csv \
      --from-date 2026-06-20 \
      --train-cutoff-date 2026-01-01 \
      --rating-cutoff-date 2026-06-19 \
      --output outputs/world_cup_2026_upcoming_forecasts.csv

Generate only the Markdown report:

    python -m wc_forecast summarize-upcoming-forecasts \
      outputs/world_cup_2026_upcoming_forecasts.csv \
      --output outputs/world_cup_2026_upcoming_forecast_report.md

Generate only the audit:

    python -m wc_forecast audit-upcoming-forecasts \
      outputs/world_cup_2026_upcoming_forecasts.csv \
      --output outputs/world_cup_2026_upcoming_forecast_audit.csv

List forecast artifacts:

    python -m wc_forecast list-forecast-artifacts \
      --output outputs/forecast_artifact_index.csv

## Model Governance Notes

This project uses time-aware validation and probability-based metrics. Forecast quality should be judged by calibration-oriented metrics such as log loss and Brier score, not only by winner accuracy.

When presenting forecasts, emphasize that the system outputs probabilities rather than certainty.

## Not Betting Advice

This workflow is for model development, analytical forecasting, and portfolio demonstration. It is not betting advice.
