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
an error. Calibration outputs are diagnostics of held-out probabilities, not a
fitted recalibration model. Hyperparameter search and model selection require
additional nested temporal validation.

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
Classifier binaries are not yet persisted. See the [Poisson model card](poisson_model_card.md).

## Integration status

The integration branch includes the competition-context repair (#14), market
features (#12), model/evaluation layer (#13), calibration diagnostics (#15), and
experiment registry (#16). It preserves their Git ancestry and the existing
squad/travel providers. Probability contracts now live in the domain layer to
avoid a feature/evaluation import cycle; the old evaluation import remains a
compatible re-export. Fresh-process imports are regression-tested.

Native Poisson migration, portable score-model artifacts and paired comparisons
are now implemented. The full reimplementation remains in progress: see the
[requirements map](implementation_status.md) for remaining calibration/ensemble,
source governance, classifier persistence, evaluation, API and deployment work.

## Comparing models

`make demo-v2-comparison` evaluates class frequency (the reference), logistic,
histogram gradient boosting and Poisson. Comparisons require identical training
match identities, evaluation match/outcome identities, forecast times and fold
cutoffs. No silently intersected sample or dropped unseen-team row is allowed.
Metrics are recomputed from predictions and differences are descriptive; there is
no automatic winner promotion or statistical significance claim. The JSON report
contains all runs and the Markdown report contains the paired score table.

Research report schema 2 adds `runs` and `comparison`. The existing top-level
`backtest`, `calibration`, `manifest` and stdout `metrics` describe the first
(reference) run. Model artifacts are listed separately in stdout. Every evaluated
model gets an experiment manifest. Poisson score forecasts retain expected goals,
the full grid, modal score, entropy and omitted tail probability.
