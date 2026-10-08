# Historical confederation membership and evaluation

Present-day team catalogs are not evidence of historical membership. Native research
accepts a separate, versioned JSON input containing published **complete timelines**
for canonical teams. This is diagnostic context; it does not change model features,
training cohorts or forecasts.

```json
{
  "schema_version": 1,
  "releases": [
    {
      "team_id": "ARG",
      "periods": [
        {"confederation": "CONMEBOL", "valid_from": "2018-01-01", "valid_to": null}
      ],
      "metadata": {
        "source": "synthetic-example",
        "source_record_id": "example-1",
        "ingested_at": "2020-06-01T00:00:00Z",
        "available_at": "2019-01-01T00:00:00Z",
        "leakage_risk": "safe",
        "legal_use_notes": "Artificial example; not a historical publication claim"
      }
    }
  ]
}
```

All dates and publication claims above are illustrative. The checked-in demo is
synthetic; production research requires documented real membership releases.

## Time and revision rules

- `valid_from` is inclusive; `valid_to` is exclusive. Null means open-ended.
- Intervals within one release cannot overlap. Gaps are permitted and mean unknown.
- Each new release replaces that team's entire timeline. It is not an incremental
  transfer event. An empty timeline explicitly withdraws all previous claims.
- For a forecast, select the latest release with publication `available_at` no later
  than prediction time, then resolve the interval containing the **match date**.
  A transfer announced in advance may therefore apply to a future fixture.
- A later correction never revises the membership assigned to an earlier forecast.
  Overlapping releases are revisions; overlapping intervals inside a release are errors.
- Equal publication times for the same team are ambiguous and rejected, including
  competing sources. Resolve that conflict explicitly upstream.
- A missing eligible release or a gap produces `unknown`. There is no fallback to
  the static current-team catalog or to an older superseded release.

Publication must be known; source record identity, safe leakage classification and
legal-use notes are required. Team IDs must exist in the supplied canonical catalog.
Ingestion time is retained but need not precede a historical forecast: this supports
archived releases collected later. As with other research inputs, a supplied publication
timestamp is an assertion requiring evidence, not proof the platform observed it live.
This is distinct from operational SQLite reads that additionally enforce recording time.

## Reports and reproducibility

`research --membership-history FILE.json` emits three dimensions: home confederation,
away confederation and ordered home/away confederation pair. Each dimension partitions
the full evaluation sample, including unknown memberships. Do not pool across dimensions.
All probabilities and scores retain the original home/draw/away convention.

The JSON includes every match's assigned memberships and selected release IDs, known
coverage, sample counts, small-sample flags, proper scores, the complete membership
input and its content identity. Release/interval ordering is canonicalized. A future
release changes input/report provenance but cannot change prior assignments or scores.
The original diagnostic identity links the output to audited canonical result joins.
A `.confederations.md` sibling file exposes coverage and slice scores for review.

```bash
make demo-v2-confederations
```

The demo verifies integration, not confederation-specific predictive skill. The [confederation-pair baseline](confederation_frequency_baseline.md) now uses these
timelines for training-only probability estimates. Historical federation-strength
estimation and sample-adequacy conclusions remain separate research work.
