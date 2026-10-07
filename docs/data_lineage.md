# Data lineage and time policy

Every source observation carries source identity, optional provider record/version, event time, publication availability, ingestion time, leakage-risk class and legal-use notes. UTC-aware timestamps are mandatory when a timestamp exists. Missing timestamps stay missing.

Operational repository rows add an independently recorded time, immutable content hash and canonical entity ID. A correction is another row. Historical reads require both publication and recording times to precede the requested cutoff. Catalogs and predictions use the same mechanism. Operators must not backdate recording times to make newly obtained information appear historically observable.

Historical CSV adapters preserve raw-file SHA256 and explicit senior-men's/score-basis assertions. Their research eligibility rule (known result availability or conservative next UTC day) is an assumption about when targets could be learned, not proof of actual source publication. `train` therefore requires `--historical-availability-assumed`. Real point-in-time validation additionally requires archived source releases and corrections.

Model datasets hash their feature schema, materialization policy, target policy and training examples. Experiment manifests retain folds, runtime/model identity and source/catalog hashes. Forecast bundles freeze source records and feature context. Forecast IDs cover their probabilities, inputs, model identities, timestamps and optional market comparison. Feature values retain observed/missing/imputed status and source/model/artifact lineage; a numeric zero never erases missingness.

Date-only matches are forecast before the UTC match date. Exact kickoff permits intraday cutoffs. Unknown/extra-time/shootout score targets are rejected. Market observations require an explicit regulation-time settlement assertion. Closing odds are a separate evaluation benchmark, never inserted into an earlier prediction-time feature vector.

Legal-use notes are declarations, not a verified license. The operator owns evidence of permitted collection, retention, derived-model use and redistribution. Store credentials outside data artifacts and source control.
