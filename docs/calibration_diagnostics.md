# Calibration Diagnostics

## Purpose

Calibration asks whether forecast probabilities match observed frequencies. A model can have useful ranking or classification accuracy and still be poorly calibrated, which makes probabilities difficult to interpret operationally.

V2 treats calibration diagnostics as part of model evaluation rather than as a plotting concern.

## Current diagnostic contract

The native V2 calibration layer consumes immutable `ScoredPrediction` objects and produces typed reliability reports for each canonical outcome:

- home win
- draw
- away win

Each outcome is evaluated one-vs-rest using equal-width probability bins.

## Metrics

### Expected Calibration Error

ECE is the forecast-count-weighted mean absolute difference between the average forecast probability and observed outcome frequency in each non-empty bin.

Lower is better; zero means the observed frequencies match the binned forecast probabilities exactly.

### Maximum Calibration Error

MCE is the largest absolute calibration gap among populated bins. It highlights localized reliability failures that can be hidden by an average metric.

### Calibration bias

Calibration bias is:

```text
overall observed frequency - overall mean forecast probability
```

A negative value indicates overprediction of that outcome on average; a positive value indicates underprediction.

## Stable bin representation

Empty reliability bins are retained with null summary values. This gives downstream tables and plots a stable shape without inventing observations or dropping parts of the probability range.

Probabilities equal to zero belong to the first bin. Probabilities equal to one belong to the final bin.

## Artifact identity

Calibration reports receive deterministic content identities derived from the number of bins plus the actual outcomes and probability distributions in the evaluation sample.

The identity is order-independent: reordering the same scored predictions does not create a different report identity.

## Separation from probability calibration

This module measures reliability; it does not fit a calibrator.

Future probability calibration must use held-out or nested out-of-sample predictions. A calibrator must never be fit on the same observations used to report its final calibration quality.

Planned calibrator research may include multinomial or one-vs-rest approaches such as temperature/logistic scaling and isotonic methods when sample size justifies them.

## Interpretation constraints

Calibration metrics are sample-dependent. Reports should be accompanied by sample counts and, for production decisions, sliced by time, competition, confederation, forecast horizon, and other relevant regimes.

Small bins should not be interpreted as precise empirical probabilities merely because a numerical observed frequency is available.
