# Sample Data Notice

Files in this directory are synthetic or simplified demo inputs used for tests, CI, and local smoke runs.

`world_cup_2026_fixtures_sample.csv` is not the official FIFA World Cup 2026 schedule. It is a synthetic, rating-safe fixture slate designed to exercise the upcoming-forecast workflow without fallback-rating warnings.

For real forecasting runs, place an actual fixture schedule in `data/raw/world_cup_2026_fixtures.csv` and run:

    python -m wc_forecast ingest-world-cup-fixtures data/raw/world_cup_2026_fixtures.csv


`v2/results.csv` is a wholly synthetic 48-match senior national-team test fixture.
The accompanying reviewed demo catalogs exist solely for the native V2 smoke
workflow (`make demo-v2`); they are not a complete production entity catalog.
