# Implementation and verification record

This report separates actual sample evidence from synthetic software tests.
Generated local artifacts are ignored by Git and contain no committed patient
rows or real model predictions.

## Real sample: structural and feature verification only

- Input actually read: `data.csv`, 40,594 bytes, **299 rows and 299 people**.
- Dispatch dates: **2020-01-14 and 2020-02-10**; age bands delivered in this
  particular sample are 65–69 and 70–74. This is not a representative cohort claim.
- No duplicate invitation IDs, same-person/time ambiguities, or contradictory
  cumulative attendance flags were found.
- Source first-invitation flag: 16 rows; all 299 are first **observed input**
  rows. These meanings must not be conflated.
- Source invitation count equals sequence in 266 of 299 rows; source counts
  appear to include more than observed extract history. Counter semantics were
  **not** inferred, corrected or used as predictors.
- Screening recency is missing in 239 rows; duration bands are missing in
  3 GP and 124 gynecology rows. Missingness was retained.
- Safe feature construction produced **101 registry-resolved features** across
  the requested snapshot/longitudinal presets. All longitudinal source histories
  in the sample have zero earlier observed rows; they do not validate empirical
  history effects.
- `inspect`, `features` and `self-test` completed. Benchmarking without
  independently verified outcome ascertainment correctly returned exit code 2.
  Even with a later verified cutoff, these two 2020 dates cannot provide
  expanding supervised training, later mature validation, and an untouched test.
- No target-driven real-sample EDA, real-sample model fitting or real-sample
  test scoring was claimed. No outcome cutoff was guessed from insurance dates.
- The full `Rakhaton2026.csv` was only encountered as a filename/file-size
  metadata entry. It was not opened, profiled, hashed, copied or trained on.

Artifacts: `artifacts/sample_audit/schema_report.md`, `schema_audit.csv`,
`inspection.json`, `feature_registry.json`, `feature_availability.json` and
`feature_summary.json`.

## Offline tests

The suite uses explicitly synthetic fixtures, no network, no Azure and no full
CSV. It covers binary/schema errors, CSV formats and string IDs, target masks,
exclusive availability boundaries including leap dates/reporting lag,
chronological folds, deliberately leaky splits, patient overlap contracts,
train-only transformations, future and target invariance, unknown categories,
cross-chunk histories, whole-patient sampling, all seven mandatory families over
all three horizons, two MLP architectures/settings, calibration partitions,
PAV properties against an independent isotonic implementation, metric semantics,
bootstrap grouping, CLI orchestration, failed/interrupted trials, resume
identity, label-free ordered prediction and fresh-process serialization.

The full CLI integration test executes inspect → features → EDA → benchmark →
train/save → fresh-process load → evaluate → predict → explain. A sentinel test
fails if a sample-only command attempts to open the named full-file path.

Final measured default suite: **143 passed in 82.58 seconds**, with no warnings.

Coverage is measured with branches and subprocess instrumentation:

| Area | Statement + branch coverage |
|---|---:|
| Whole production file | 91.68% (93.73% statements; 86.56% branches) |
| Schema/parsing contracts | 98.00% |
| Temporal availability/splitting | 96.36% |
| Feature construction/resolution | 99.13% |
| Coherence/class mapping | 100.00% |
| Persistence | 100.00% |

Critical-area branch-only coverage is 97.50% for schema, 94.44% for temporal
contracts, 97.62% for features, and 100% for coherence/persistence. The final
tests exercise incompatible horizon/library/code versions, cleanup after
serialization failure, and invalid estimator class mapping.
Optional installed-library/SHAP branches, some corruption/error
branches and the complete patient-disjoint artifact evaluation remain gaps;
native category preprocessing itself is tested. Combined coverage is not
misrepresented as branch-only coverage.

## Completed synthetic model experiment

**Not empirical VZP evidence.** The fixed seeded synthetic fixture contains
1,200 invitations, 120 repeated people and 2015–2024 invitation years. Fully
observed validation contains 240 invitations in 2021/2022; the frozen test
contains 120 invitations in 2024. Synthetic ascertainment is explicitly set to
2026-01-02; it is not a date asserted for the real data.

The main run requested all mandatory/available optional families, two quick
parameter settings per nontrivial family, and seven presets:
raw snapshot, snapshot engineered, longitudinal, all safe, no geography,
no calendar, and no longitudinal. On this schema two pairs of presets resolve
to identical features and share fits, explicitly recorded in aliases.

- **65 successful three-horizon trials**, each across two chronological folds.
- All seven mandatory families completed; LightGBM, XGBoost and CatBoost were
  explicitly skipped because their optional packages were absent.
- Two additional coherent interval-logistic configurations completed on the
  same validation rows. Their macro log losses were 0.586974 and 0.609302;
  these are mechanics checks, not clinical performance.
- Allocated search budget: **120 seconds**.
- Measured loading: **0.068 seconds**; search: **92.893 seconds**;
  feature construction: **0.116 seconds**;
  final refit: **0.281 seconds**; reporting: **1.036 seconds**;
  total after load: **94.366 seconds**; command total: **94.459 seconds**.
- Individual three-horizon/two-fold trials took **0.444–4.078 seconds**.
  Ten of 65 trials recorded warnings, principally bounded MLP convergence
  warnings. They are retained in the ledger, not suppressed.
- The search completed its finite meaningful schedule before the deadline.
  It did not sleep or repeat identical feature-set fits to consume the budget.
- Validation-selected components were dummy/snapshot-engineered at 90 and
  180 days and decision-tree/no-geography at 365 days. Dummy champions are valid
  weak-signal results, not a failure to implement stronger estimators.
- Final coherent validation macro log loss: **0.576700**.

All main and calibration candidates were frozen **before** their separate final
evaluation commands. The synthetic test did not change the selection.

### Explicit locked synthetic test

| Horizon | Log loss | Brier | ROC AUC |
|---|---:|---:|---:|
| 90 days | 0.350907 | 0.098454 | 0.500000 |
| 180 days | 0.604148 | 0.206808 | 0.485714 |
| 365 days | 0.690519 | 0.248724 | 0.527065 |

Macro final log loss: **0.548524**. Raw vectors violated nesting on **1/120**
invitations; final projected vectors violated it on **0/120**. The 30-repeat
patient-cluster percentile interval for macro log loss was **[0.514681, 0.594483]**.
These numbers illustrate scoring mechanics on synthetic labels, not predictive
utility or population calibration. Projection can change even a constant
component's AUC through dependence on another horizon; component and deployed
metrics are therefore reported separately.

The saved artifact was loaded by a new executable process for final evaluation
(1.950 seconds including load). Held-out development explanation took 2.205
seconds including load. The real-sample inspect and feature commands took
0.065 and 0.220 seconds respectively, excluding interpreter startup.
Another fresh process predicted 20 shuffled, target-free synthetic invitations
with external predictor history. Output order, `(20,3)` probability shape,
finite bounds, named horizon order and final coherence were verified.
Fresh-process prediction equality is also covered by a tolerance-`1e-12` test.

Main outputs under `artifacts/synthetic_verification/run/`:

- Frozen `selection.json`, `test_manifest.json`, `validation_bundle.joblib`
  and `model_bundle.joblib`.
- Append-only `trials.jsonl`, per-family best-tested parameters, validation
  leaderboard, exact resolved-feature reports/aliases and checkpoint identities.
- Paired fixed-setting feature comparisons and per-horizon/fold differences.
- Raw/calibrated/final metrics, reliability plots and prediction quantiles.
- Explicit `final_evaluation/` report, patient bootstrap, subgroup and quarterly
  sensitivity metrics; no final-test ranking was used for model choice.
- Held-out-development explanation of 120 seeded rows, nine semantic permutation
  groups with three repeats, anonymous local sensitivity diagnostics, and three
  supported **synthetic** rule-defined segments. SHAP was cleanly skipped as
  unavailable. Generated explanations distinguish final-wrapper permutation
  from base-estimator effects.

## Separate chronological calibration verification

Before final evaluation, a separate sigmoid run froze six successful
dummy/logistic configurations over two feature sets. Base models were trained
before a dedicated 365-day calibration window, whose labels matured before
the next validation/test origin.

- Allocated budget: 30 seconds; measured search: **3.515 seconds**.
- Measured calibration work across trials: **0.435 seconds**;
  final refit: **0.234 seconds**; total after load: **4.813 seconds**.
- No training warnings were recorded.
- Final coherent validation macro log loss: **0.582122**.
- Explicit frozen synthetic test macro log loss: **0.560513**.
- Raw test violation frequency: **0.775**; calibrated/final: **0**.
- This separate exercise is not a test-selected replacement for the main
  champion and does not demonstrate calibration transfer to real patients.

Isotonic fitting is covered by adequately sized synthetic unit data; this run
did not claim a separate empirical isotonic-calibration finding.

## Remaining limitations

There is no verified real outcome cutoff, usable empirical chronological split,
known sample-selection mechanism or empirical patient-history validation. A
patient-disjoint stress fit is implemented, but this synthetic fixture reuses
every patient in test, leaving no disjoint supervised training cohort. It is
not reported as successfully evaluated.

The full-data path is chunk-read/selected-column **materialization**, not
out-of-core training; it is cross-chunk tested but unexecuted on the full file.
Memory guards are conservative frame/matrix checks, not guarantees for every
estimator's internals. Fit cancellation is a soft launch deadline plus bounded
iterations, not a hard worker timeout. The interval benchmark is validation-only.
Additional candidate feature families and per-segment cluster intervals remain
listed explicitly in the README, rather than being implied implemented.

No patient-level output, model binary or dataset was added to Git or committed
by this implementation. The full CSV was already tracked before this task and
was left unchanged; an ignore pattern does not untrack an existing file. No external
service received patient-level input, and no Azure resources/jobs or contacts
were created.
