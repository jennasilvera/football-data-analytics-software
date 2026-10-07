# Native score-target contract

Native research predicts goals and home/draw/away outcomes at the end of
regulation time, including stoppage time. Extra time and penalty shootouts are
excluded. Policy identity: `regulation_time_goals_and_1x2_v1`.

## Source declarations

| `score_basis` | Meaning | Native research |
|---|---|---|
| `unknown` | Period not documented | Reject |
| `regulation_time` | Regulation plus stoppage time | Accept |
| `after_extra_time` | Cumulative score after extra time | Reject |
| `including_shootout` | Source totals include shootout tallies | Reject |
| `shootout_only` | Shootout tallies alone | Reject |

The label describes `home_score` and `away_score`, not which team advanced.
A regulation draw remains a draw when a team later wins in extra time or a
shootout. Do not infer regulation scores from the final score, tournament,
winning team, or presence of a shootout.

The CSV adapter reads a `score_basis` column when present. Without that column,
use `research --score-basis regulation_time` only when source documentation
supports that assertion for every row. The API equivalent is the typed
`ScoreBasis.REGULATION_TIME` adapter policy. The default is `UNKNOWN`.
A mapped blank remains unknown; a conflicting row and source assertion raises
an error. Invalid enum labels also fail rather than falling back.

Ingestion preserves unknown and other-period records and exports their labels
in normalized CSVs. Research rejects the entire batch if any completed target
has an unsupported basis, even outside the chosen evaluation windows. It does
not silently remove records or change the research population. History consumers
reject unsupported eligible results; Poisson fitting and canonical-to-rating
conversion enforce the same contract. Temporal availability remains a separate
policy, so knowing a result's period does not prove when it was available.

## Identities and compatibility

The target policy is included in model dataset and backtest identities,
experiment manifests, Poisson artifacts, research/forecast JSON, and comparison
output. New versions are model dataset schema 3, Poisson schema 2, experiment
manifest schema 2, research report schema 5, and forecast report schema 2.
Old Poisson models and experiment manifests fail loading; retrain from a source
with verified score semantics. Do not relabel an old artifact or edit its hash.
The legacy `wc_forecast` workflow is retained unchanged and does not provide
this native contract.

Synthetic V2 sample rows explicitly declare regulation scores. This demonstrates
software behavior, not provider validation or predictive skill. Existing native
rating parity tests now explicitly declare their synthetic scores' period.

## Remaining source work

This milestone does not reconstruct regulation scores from extra-time totals,
store separate period-by-period tallies, verify licenses, or reconstruct historical
source revisions. A source with multiple periods needs documented regulation
fields mapped into these targets, with appropriate source/version provenance.
Real-data rollout and market comparisons still require source verification and
compatible market settlement rules. Nested calibration and ensemble fitting are described in
[temporal postprocessing](temporal_postprocessing.md).
