# Native V2 research workflow

Run `make demo-v2` after installing the project and `requirements.txt`. This
executes canonical ingestion, rolling-form feature generation, chronological
model fitting, out-of-sample prediction, calibration diagnostics, and experiment
registration. The 48-match fixture in `data/sample/v2` is entirely synthetic;
its metrics test software behavior and provide no evidence of predictive skill.

## Inputs and declared assumptions

`python -m football_analytics research --help` lists the interface. The installed
`football-analytics` command is equivalent. `--model class-frequency` runs the
training-only smoothed frequency benchmark; `--model poisson` runs the canonical
score model; `--model all` evaluates all four families on identical folds and
emits a paired comparison report. Required
inputs are a results CSV, governed team and competition JSON catalogs, source ID,
timezone-aware ingestion time, one or more timezone-aware training cutoffs, and
evaluation-window length. The CSV uses the retained legacy results schema:
`date,home_team,away_team,home_score,away_score,tournament,neutral`.
Scores must additionally have an explicit `score_basis` column, or a documented
source-wide `--score-basis regulation_time` assertion. Missing labels default to
unknown and stop native research. Mapped row labels cannot be overridden by the
source assertion. See [score target contract](score_target_contract.md).

The mandatory `--assert-senior-mens-a` flag is a source-level assertion that rows
are official senior men's A-internationals. It is not an automatic verification
of source eligibility. Mixed or unreviewed sources must use source-specific
adapters and scope assessments before entering this workflow. Catalog entries
are reviewed identities, not entities generated from arbitrary input names.

Unresolved entities, duplicates, excluded competitions, or other normalization
failures stop the run with exit code 2 and row audits under `rejected-input/`.
No experiment is registered for that failed run. Invalid CSV or catalog syntax
also fails rather than silently dropping rows.

The initial executable baseline uses rolling form over 5 and 10 prior matches.
Unavailable form values use explicitly versioned zero imputation with missing
and imputed indicators. It does not infer team strength from a default Elo.
Logistic regression is the default; `--model hist-gradient-boosting` selects the
other native V2 estimator. All three outcome classes must occur in training.

The CSV adapter has date-only results and no publication timestamps. The existing
conservative next-UTC-day result-eligibility policy applies, with forecasts just
before each match date. This is a research assumption, not proof of historical
source availability or protection against retrospective source revisions.

## Evaluation

Training is expanding-window by default. Add `--training-days N` for a rolling
window. Targets must be available by each training cutoff. Feature history is
limited to information eligible at each individual prediction time. The fitted
model remains fixed within a fold, while history can advance as results become
eligible. This is a sequential forecast backtest, not a single batch forecast
made at the start of the evaluation window.

Evaluation windows cannot overlap: pooled metrics count each forecast once.
Skipped folds retain their reasons and counts in the report; no valid folds is
an error. The `calibration` field reports probability reliability diagnostics.
Optional `--postprocess-days` also fits learned transforms using a chronological
inner holdout, with its own audit and artifacts. Hyperparameter search and model
selection still require additional validation and an untouched final test period.

## Outputs and reproducibility

`outputs/v2-research/research_<sha256>.json` contains:

- exact input-byte digest, source assertion, ingestion time, and legal-use notes;
- complete resolved catalog definitions and split/model specifications;
- per-fold train/evaluation dataset IDs and every out-of-sample prediction;
- aggregate/fold probability metrics and skipped-fold audit;
- reliability bins and calibration metrics;
- the self-validating experiment manifest;
- Python, NumPy, pandas, SciPy, and scikit-learn versions.

The report is published atomically without replacing existing content. Its name
is the SHA-256 digest of its full bytes. The same inputs, declared timestamps,
parameters, code revision, and runtime produce the same report on a deterministic
runtime. Supply `--code-revision` explicitly when recording a source revision;
the command does not assume a dirty checkout represents a clean commit.

The experiment manifest is additionally stored through `JsonExperimentRegistry`
in `experiments/`. Reports capture more provenance than the current manifest
identity: runtime/catalog/source changes can create a different report while
retaining an experiment ID if the evaluated model datasets are unchanged.
Reproducibility across different numerical libraries or hardware is not promised.
Poisson fitted state is persisted as self-validating JSON, without pickle.
Operational training now persists frequency, logistic and numeric boosting state as
portable JSON in a forecast bundle, with export parity checks. Research runs continue
to emit their fold Poisson/transform artifacts. See the [model card](model_card.md).

## Integration status

The integration branch includes the competition-context repair (#14), market
features (#12), model/evaluation layer (#13), calibration diagnostics (#15), and
experiment registry (#16). It preserves their Git ancestry and the existing
squad/travel providers. Probability contracts now live in the domain layer to
avoid a feature/evaluation import cycle; the old evaluation import remains a
compatible re-export. Fresh-process imports are regression-tested.

Native model bundles, temporal calibration/ensembles, research extensions, operational
services, API, dashboard and deployment artifacts are implemented. See the
[requirements map](implementation_status.md) for current coverage and the external
data, empirical-validation and commercial-release gates.

## Comparing models

`make demo-v2-comparison` evaluates class frequency (the reference), logistic,
histogram gradient boosting and Poisson. Comparisons require identical training
match identities, evaluation match/outcome identities, forecast times and fold
cutoffs. No silently intersected sample or dropped unseen-team row is allowed.
Metrics are recomputed from predictions and differences are descriptive; there is
no automatic winner promotion or statistical significance claim. The JSON report
contains all runs and the Markdown report contains the paired score table.

Research report schema 5 declares `target_policy_id` and retains `runs` and `comparison`. The existing top-level
`backtest`, `calibration`, `manifest` and stdout `metrics` describe the first
(reference) run. Model artifacts are listed separately in stdout. Every evaluated
model gets an experiment manifest. Poisson score forecasts retain expected goals,
the full grid, modal score, entropy and omitted tail probability.

## Learned probability transforms

`make demo-v2-nested` adds chronological inner holdout fitting to the four-model
comparison. See [temporal postprocessing](temporal_postprocessing.md) for the
complete fitting policy, artifacts, failure behavior and evaluation limits.


## Declared research extensions

`--feature-groups form,schedule,competition` selects feature families before fitting.
`--feature-context FILE.json` supplies timestamped rankings, squad, market, competition,
venue/team locations and manual records. See the data dictionary for contracts.
`--ablate` evaluates a logistic/boosting full model and leave-one-family-out runs;
it requires at least two declared families and identical temporal evaluation samples.
`--paired-uncertainty` adds calendar-block paired-loss intervals with explicit sample
and dependence assumptions. `--market-snapshots FILE.json` adds a prediction-time
benchmark with publication/settlement checks and unmatched-coverage reasons.
Closing-time benchmarks are available through the typed evaluation service and require
known exact kickoff; they never become an earlier forecast feature.

## Competition-aware benchmark

`--model competition-frequency` evaluates a training-only competition baseline.
`--model competition-comparison` pairs it with global class frequency. Existing
`--model all` and nested ensemble member sets are unchanged. See the
[baseline specification](competition_frequency_baseline.md) for shrinkage, catalog
identity and explicit unseen-group fallback semantics.

## Historical confederation slices

`--membership-history FILE.json` resolves each team's membership for the match date
using the latest complete timeline published by its prediction cutoff. The research
JSON embeds membership releases, lineage, coverage, assignments and slice metrics;
a sibling `.confederations.md` report presents the scores. Unknown memberships
remain in each partition. See [membership semantics](historical_confederations.md).
Run `make demo-v2-confederations` for an explicitly synthetic publication fixture.

## Confederation probability benchmark

`--model confederation-frequency --membership-history FILE.json` fits an ordered
home/away confederation-pair frequency baseline. `--model confederation-comparison`
pairs it with global class frequency on identical folds. Unknown memberships and
unseen pairs use the global training estimate; sparse known pairs shrink toward it.
The membership file is loaded once for both features and diagnostic lineage. See the
[model specification](confederation_frequency_baseline.md) for temporal and fallback rules.
