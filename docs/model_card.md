# Native forecast bundle model card

**Status:** implemented research software; real-data performance and probability-interval coverage unvalidated. Version 0.3.0. Run `football-analytics model-card --model-id ID` for the concrete registered bundle's provenance and release status.

## Intended and excluded uses

Estimate regulation-time home/draw/away probabilities for known canonical senior men's A-international teams, compare forecasts chronologically, study ratings and inspect football intelligence. Not guaranteed results, a causal model, verified betting advice, player-level projections, or a validated model for excluded competitions.

## Modeling approach

| Member | Approach | Important limitation |
|---|---|---|
| Frequency | Training-only 1X2 counts with declared smoothing | No team/context discrimination |
| Logistic | Standardized numeric features with regularized multinomial classifier | Linear contributions; sparse histories and imputation affect fit |
| Histogram boosting | Numeric histogram gradient-boosted trees | Export supports numerical splits only; exporter tied to locked sklearn version |
| Poisson | Team attack/defense strengths, home effect and independent score distributions | Independent goals; finite grid tail explicitly reported |
| Temperature calibration | Held-out inner temporal probability rescaling | Calibration learned on inner models may not transfer unchanged after refit |
| Convex ensemble | Nonnegative, sum-to-one held-out weights over base probabilities | Correlated members and small holdouts can produce unstable weights |

The default bundle uses form features. Explicit feature groups enable strength, schedule, competition, ranking, squad, market, travel and manual inputs. Missing inputs receive declared zero materialization plus status indicators; Glicko has a declared prior. A rankings-only logistic experiment is supported by `--feature-groups rankings`; it is not a claim of a separately validated FIFA model.

Glicko-1 uses daily simultaneous rating periods, initial 1500/RD350, and RD inflation of 50 rating points per 30 days in quadrature (capped at 350). These are declared research settings. The numerical update is checked against the [original worked example](https://www.glicko.net/glicko/glicko.pdf); football calibration remains empirical work. Rating intervals are approximate rating-scale intervals, not match-probability credible intervals.

## Evaluation and promotion

Predeclare temporal expanding/rolling folds, feature families and sample minima. Inner holdouts fit temperature/ensemble parameters before untouched outer outcomes. Report log loss, Brier, RPS, accuracy, reliability bins, competition/team/year/venue slices and coverage. Compare models on identical training/evaluation identities; use family ablations and strictly cutoff-joined market benchmarks. Block-bootstrap loss intervals disclose block assumptions and return insufficient coverage when fewer than four blocks exist.

Promotion requires linked research with matching source digest and an explicit review reason. Operators must inspect feature/model correspondence and an untouched final evaluation period. Approval is an auditable decision, not a performance threshold or license certificate. Synthetic acceptance approval is confined to a temporary database.

## Interpretability, uncertainty and bias

Forecasts show up to eight standardized logistic coefficient contributions to the chosen class. They do not explain the ensemble causally. Poisson score outputs remain raw Poisson even when reported 1X2 probabilities are recalibrated/ensembled. Entropy describes distribution spread; it does not establish confidence or correctness. `confidence_category` is `unvalidated` and `uncertainty_interval` is null until a validated interval procedure exists.

Potential biases include sparse nations, uneven federation/source coverage, reporting delays, confederation imbalance, historical rule changes, squad-data missingness, national-team turnover and market selection bias. Fixed catalogs do not prove historical confederation membership. Manual observations require timestamps and legal notes; hindsight edits must remain revisions.

## Updates and monitoring

Retrain after governed data refresh, compare candidates and review releases. Serving refuses history older than 90 days by default; this is an operational setting, not a calibrated optimum. Monitor missingness, drift, model age, coverage, request errors and post-match losses. Residual watchlists are descriptive heuristics, not validated regression/breakout predictions. Empirical service and research limits are tracked in [commercial readiness](commercialization_notes.md).
