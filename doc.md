# Mammography participation after an invitation

A local, CPU-first, single-file Python workflow for predicting **no recorded
screening within 90, 180 and 365 days after invitation dispatch**. This is not
scheduled-appointment no-show prediction. No outreach, web application, cloud
provisioning or paid job is implemented.

## Current evidence and important blockers

- The explicitly supplied development sample is **`data.csv`**, not
  `Rakhaton2026.csv`. Only the former was opened during implementation.
- The sample contains **299 invitations, 299 people, and two dispatch dates:
  14 January and 10 February 2020**. All observed input histories contain one row.
  It cannot support honest chronological training/validation/test scoring.
- `Rakathon_Datove_rozhrani.docx` and `data/readme.docx` contain the same
  29 September 2026 dictionary text. Word table shading was checked against
  delivered/undelivered fields; the supplied Markdown adds explicit delivery
  markers. `Rakathon.md` supplies the updated invitation/support brief.
- The dictionary describes 1,344,556 invitations in 2020–2024 for 508,527 people.
  Those are **documented source-cohort counts**, not measurements of the full CSV.
- Continuous insurance through 2025 does **not** verify an outcome observation
  cutoff. No such cutoff or recording-delay guarantee is supplied.
- There is **no empirical VZP model, valid sample test metric, or population
  calibration claim**. Completed model comparisons and persistence checks use
  separately labeled synthetic data. See [VERIFICATION.md](VERIFICATION.md).

The pipeline refuses to invent an observation end or substitute a random split.
The full CSV was not opened, scanned, hashed, copied or trained on.

## Setup

Python 3.11+ is recommended; verification used Python 3.13.15 on Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python mammography_pipeline.py self-test
pytest -q
```

`requirements.txt` specifies supported core ranges; `requirements-lock.txt`
records the exact verification environment, including test dependencies.
NumPy is bounded below 2.4 for the verified pandas 2.x environment.
`requirements-dev.txt` is the less tightly pinned test installation route.
Optional estimators/SHAP are separate:

```bash
python -m pip install -r requirements-optional.txt
```

Optional packages are imported lazily. Their absence is reported, not replaced
by a different family. Use `--strict-optional` to make absence an error.
No optional package was installed in the verification environment.

## Run the real sample safely

These commands need no outcome-end assumption and do not fit models:

```bash
python mammography_pipeline.py inspect \
  --data data.csv --data-mode sample --output-dir artifacts/sample_audit

python mammography_pipeline.py features \
  --data data.csv --data-mode sample \
  --feature-sets raw_snapshot,snapshot_engineered,longitudinal_no_labels,all_safe \
  --output-dir artifacts/sample_audit

python mammography_pipeline.py features --list-features
python mammography_pipeline.py benchmark --help
```

The audit writes one schema entry per delivered column, canonical aliases,
missingness, cardinality, units, temporal reference, inclusion decisions and
unresolved questions. The feature command writes a registry availability
report; it does not export patient rows unless `--export-cache` is requested.

This is an intentional error until an independent source verifies the cutoff:

```bash
python mammography_pipeline.py benchmark --data data.csv --data-mode sample
```

Even a verified later cutoff cannot create the years of missing invitation
history needed for evaluation. Do not fill in a hypothetical cutoff to present
sample scoring as valid.

## Reproduce the synthetic verification

Synthetic identifiers start with `SYNTH-`. This fixture has 1,200 invitations,
120 people, and ten invitation years. It is a software test, not a simulation
validated against the actual sample distribution.

```bash
python -c "from pathlib import Path; from tests.conftest import synthetic_frame; p=Path('artifacts/synthetic_verification/input_SYNTHETIC.csv'); p.parent.mkdir(parents=True, exist_ok=True); synthetic_frame(120).to_csv(p, index=False, date_format='%Y-%m-%d')"

python mammography_pipeline.py benchmark \
  --data artifacts/synthetic_verification/input_SYNTHETIC.csv \
  --data-mode sample --date-format %Y-%m-%d \
  --outcome-data-end 2026-01-02 --test-year 2024 \
  --assume-t0-snapshots --models all-available \
  --feature-sets raw_snapshot,snapshot_engineered,longitudinal_no_labels,all_safe,all_safe_no_geography,all_safe_no_calendar,all_safe_no_longitudinal \
  --preset quick --budget-seconds 120 --max-trials 70 --n-jobs 2 \
  --bootstrap 30 --joint-interval --save-model \
  --provenance 'SYNTHETIC verification only; not empirical sample evidence' \
  --output-dir artifacts/synthetic_verification/run

# Choices are already frozen; evaluation is a separate, explicit action.
python mammography_pipeline.py evaluate \
  --model-path artifacts/synthetic_verification/run/model_bundle.joblib \
  --data artifacts/synthetic_verification/input_SYNTHETIC.csv \
  --data-mode sample --split test --bootstrap 30 \
  --output-dir artifacts/synthetic_verification/run/final_evaluation

python mammography_pipeline.py explain \
  --model-path artifacts/synthetic_verification/run/validation_bundle.joblib \
  --data artifacts/synthetic_verification/input_SYNTHETIC.csv \
  --explanation-rows 120 --permutation-repeats 3 --shap \
  --output-dir artifacts/synthetic_verification/run/explanations
```

Use a new output directory for a new experiment. A compatible interrupted or
completed search can be resumed by repeating the benchmark command with
`--resume`. It skips finished trials, not their required reporting. Data,
folds, schema/configuration, code and library versions must match. The budget
is a fresh search allocation on each resume; it can be increased. Completed
failures remain recorded; an interrupted, unfinished trial may be retried.

For a separately frozen calibration experiment, before examining its test:

```bash
python mammography_pipeline.py benchmark \
  --data artifacts/synthetic_verification/input_SYNTHETIC.csv \
  --date-format %Y-%m-%d --outcome-data-end 2026-01-02 --test-year 2024 \
  --assume-t0-snapshots --models logistic \
  --feature-sets raw_snapshot,snapshot_engineered --preset quick \
  --budget-seconds 30 --n-jobs 2 --calibration sigmoid --calibration-days 365 \
  --bootstrap 30 --save-model \
  --provenance 'SYNTHETIC calibration mechanics only; not empirical sample evidence' \
  --output-dir artifacts/synthetic_verification/calibration
```

`train --selection-path .../selection.json --model-path .../model_bundle.joblib`
performs the same frozen-selection refit when `benchmark --save-model` was not
used. Parsing, feature/availability configuration and selection semantics come
from that selection, not new default CLI settings. Changed development
predictors/labels are rejected. Later deployment retraining must be a new run;
it cannot inherit an old artifact's evaluation metrics.

## Data and target contract

One row is one invitation at T0. Identifiers remain strings. Explicit canonical
aliases are built in and can be extended with `--schema-mapping`; see
[schema_mapping.example.json](schema_mapping.example.json). CSV defaults are
comma, UTF-8 with optional BOM, and `%d.%m.%Y`; these are persisted for inference.
Use an explicit date format for other inputs. A normal run never discovers
or enumerates CSV contents automatically.

Binary A/N, Y/N, Boolean and 0/1 values map explicitly; unknown encodings fail.
Blank/NULL binary values remain missing. Unknown extra columns are audited but
never used as predictors. Duplicate/aliased column collisions, missing IDs,
invalid dates, negative/fractional counts, conflicting IDs, ambiguous
same-person/same-time rows, inconsistent screening history and nonnested
observed labels stop execution. Duplicates are not silently aggregated.

```
y_nonattendance_h = 1 - recorded_screening_flag_h
output order = [p_nonattendance_90d, p_nonattendance_180d, p_nonattendance_365d]
1 >= p90 >= p180 >= p365 >= 0
```

These are nested cumulative events, **not** exclusive classes: probabilities
are not normalized to sum to one. The deployed wrapper extracts class 1 by
`classes_`, optionally calibrates each component, then applies deterministic,
equal-weight nonincreasing PAV. For example `[0.7, 0.8, 0.4]` becomes
`[0.75, 0.75, 0.4]`, not a sorted vector. Raw, calibrated and final scores,
violation frequency/magnitude and projection changes are all retained.

`NEVER_SCREENED_BEFORE_FLAG` is named `no_screen_observed_5y`; it does not
establish lifetime nonparticipation. Contacts count unique provider/day
combinations, not visits or unique providers across the entire year. The
territorial office is not a district, travel distance or deprivation score.

Three ambiguous source fields are audited but disabled: source invitation
count, source sequence, and source invitation recency. The sample's source
counter and sequence do not always agree; their semantics must not be inferred
from correlations. Source first-invitation status remains distinct from
`observed_prior_rows == 0`.

## Temporal methodology

1. Reserve a manifest using invitation dates/keys **before target-driven work**.
   The default test is the latest complete calendar year with conservative full
   365-day follow-up; `--test-year` or paired explicit dates override it.
2. For horizon h, availability is `T0 + h days + reporting_lag_days`. Both
   positive and negative labels use this same conservative delay.
3. Training requires `T0 < fit_origin` and `label_available_at < fit_origin`.
   Observation cutoff and selection deadlines are also **exclusive**. Missing
   or immature outcomes remain excluded, never negative attendance labels.
4. Expanding chronological validation uses date boundaries and outcome maturity,
   not a row-count gap. A full year immediately before test usually cannot
   support 365-day selection. In the synthetic run, 2021/2022 validation is
   available before the 2024 selection deadline.
5. `--cohort-policy horizon` uses more recent eligible shorter-horizon labels;
   `common` uses a shared mature three-target training cohort. Bundle selection
   and scoring use identical fully observed validation rows. Additional
   horizon-specific final scores retain their own counts.
6. All learned preprocessing is fit inside the permitted partition. Scaling
   is used for logistic/MLP, not trees. MLP and histogram boosting have random
   internal early stopping disabled.
7. Per-horizon selection minimizes common-cohort chronological log loss with
   equal horizon weights. The complete projected bundle is evaluated on
   validation as well. The leaderboard does not rank or select by final test.
8. Final models/transforms are frozen before test. Test evaluation checks its
   manifest and rejects supervised training/calibration ID overlap.

`--assume-t0-snapshots` is required for fitting: recording timestamps are absent,
so retrospective snapshots must be assumed available at dispatch. This does not
prove absence of predictor backfill. `--reporting-lag-days 0` is similarly an
assumption unless independently verified.

Patient overlap is allowed in the operational estimand. No IDs, hashes, label
proxies or future-inclusive patient totals enter predictors. Histories are
chronologically assembled before slicing and restored to input order; no
history reset at a split boundary. Training vocabulary/scaling remains fixed.

`--patient-disjoint-stress` predeclares a separate artifact excluding every test
person from supervised base/calibration fitting. Earlier predictor snapshots
are still allowed. This requires enough non-test people; the provided sample
and the all-repeated synthetic fixture cannot establish this stress estimand.
Do not replace the operational evaluation with it.

## Features and independently selectable presets

Every feature is registered with source columns, formula, units, type, overlapping
semantic groups/tags, availability, missingness meaning, history requirements,
leakage caveats and hypothesis/evidence status. Names and order are deterministic.
Selection deduplicates by canonical name; exactly identical resolved presets
share fits and are reported as aliases rather than wasting the search budget.

| Preset | Contents |
|---|---|
| `minimal` | Age band, first-invitation flag, screening count/recency, registrations |
| `raw_snapshot` | All audited delivered raw T0 predictors, excluding ambiguous counters |
| `demographics_calendar` | Age bands, territorial office, calendar candidates |
| `screening_invitation` | Screening and invitation snapshots/transformations |
| `prevention_engagement` | Window-specific prevention/contact features |
| `registration_continuity` | Registration flags/bands, insurance and selected interactions |
| `snapshot_engineered` | Approved snapshots, missingness, log counts, bins, annualized rates, interactions |
| `longitudinal_no_labels` | Snapshot features plus strictly earlier predictor history |
| `longitudinal_with_mature_labels` | Explicit opt-in mature prior-outcome extension |
| `all_safe` | Available default-safe snapshot and longitudinal features |
| `all_safe_no_geography` | Geographic family removed |
| `all_safe_no_calendar` | Calendar family removed |
| `all_safe_no_longitudinal` | Predictor-history family removed |

`--include-groups` adds families; `--exclude-groups` removes them. The missing-source
report is explicit; zero usable features is an error. By default cyclic month
is used instead of redundant month/quarter/day one-hot encodings. Explicitly
including `calendar` enables those alternatives for an ablation. `age_order`
is the documented band order, not an exact age or midpoint reconstruction.

Longitudinal features include observed input counts/elapsed time, prior
inter-invitation interval statistics, density with a one-year denominator floor,
two predictor lags, rolling-summary changes, annualized changes, and expanding
past snapshot statistics. A rolling-summary delta is **not** a count of new
medical events. Partial source histories remain partial, even for seen patients.

Annualized prevention counts respect each 24/36-month exposure window. Dividing
these by 12-month contacts is an **intensity ratio**, not a probability/share.
Missing domains and zero denominators stay unknown. The 24-month screening gap
is a behavioral reference, not verified eligibility or a clinical recommendation.

`--historical-outcomes off` is the default. When on, an as-of join admits an
earlier observed outcome only after its entire horizon plus reporting lag and
before the current dispatch, also respecting verified ascertainment. Pending
outcomes are not failures. These features require outcome access in deployment.
No response streak or event-level causal attribution is inferred.

## Models, search, calibration and diagnostics

Mandatory factories: prior dummy, regularized logistic regression, decision
tree, random forest, Extra Trees, histogram gradient boosting and sklearn MLP.
The dummy is always included. Quick mode tests two parameter configurations per
nontrivial family and defaults to two feature sets. MLP quick architectures are
`(32,)` and `(64,32)` with different regularization; broader search also supports
`(64,)`, `(128,)`, `(128,64)` and `(128,64,32)` with learning-rate alternatives.

`--models all-available` also requests LightGBM, XGBoost and CatBoost. CPU adapters
use OHE by default; `--categorical-mode native` preserves a train-only category
vocabulary with distinct missing/unknown handling for these adapters. Native
categorical paths were unit-tested at the transformer level, **not** fitted
against absent optional libraries.

The runner searches family × feature set × valid parameters × horizons across
chronological folds. Trials run sequentially with bounded estimator/OpenMP
threads. Quick uses a small grid; standard/extensive support seeded randomized
ordering of conditional finite candidate spaces (`--search-method random`).
This is not Optuna or continuous Bayesian optimization.

Trials checkpoint their status, exact features, parameters, cohort policy,
fold counts/bounds, warnings, metrics and elapsed time in append-only JSONL.
Internal validation predictions/checkpoint models stay under ignored artifacts.
Resume identity includes development values, key manifests, code, schema,
configuration, fold boundaries and package versions. No cross-fold learned
transform cache is reused.

Calibration is `none`, `sigmoid` or `isotonic`. The uncalibrated model remains
available in all cases. A dedicated later, label-mature chronological window
calibrates an earlier-frozen base; validation comes later still. Isotonic requires
at least 200 calibration rows and both classes; sigmoid requires at least 30.
The base is **not** refitted after calibration. Each method's training reduction
and bounds are saved. Choose calibration methods in separate predeclared
development runs, not from test metrics.

Reports include log loss, Brier score and improvement over fold-training priors
on the same rows; ROC AUC; AP for each positive-class convention with its
prevalence baseline; reliability bins/Wilson intervals and descriptive ECE;
probability quantiles; 0.5 threshold metrics/confusion matrix; and top
5/10/20% risk-ranking diagnostics. No tuned threshold is claimed. Final scoring
adds patient-cluster bootstrap intervals, paired differences against a frozen
prior baseline, seen/unseen and first/repeat/sparse-history slices, and quarterly
block sensitivity. Single-class and empty cohorts remain undefined where
appropriate. Row Wilson intervals are descriptive, not cluster-adjusted.

Fixed-setting, paired-row feature-set differences and per-fold differences are
separate from independently retuned per-family/horizon winners. More features
are not assumed better. Stability and uncertainty must be re-established on
proper empirical data.

## Explanations, segments and later intervention experiment

Use **`validation_bundle.joblib`**, not the all-pretest refit, for held-out
development explanations. Grouped repeated permutation scores the final
calibrated/coherent wrapper on a seeded bounded sample. Derived columns are
perturbed together by semantic family; cross-family combinations can still be
unrealistic, so this is a diagnostic, not a simulated intervention.

Logistic coefficients refer to standardized numeric inputs and full regularized
one-hot encoding with no dropped reference category; they explain uncalibrated
logits. Optional bounded tree SHAP uses supervised-training-only backgrounds and
explains a **base tree**, not calibration/PAV. Missing SHAP is explicitly skipped.
Anonymous local reference sensitivities omit identifiers and attributes, are
not additive attributions, and are not intervention effects. Support-screened
conditional-bin summaries do not manipulate correlated features.

Transparent outcome-free segment rules cover no recorded contacts, contacts
without recorded prevention, long recency with prior screening, and registration
without recorded prevention. Reports include support, observed/predicted risks,
descriptive uncertainty and suppressed undersized groups. Segments overlap.
Only **synthetic** supported-segment reports could be generated here.

Always distinguish:

1. Observed association in a permitted data partition.
2. Association used by a specific fitted predictive model.
3. Tentative behavioral interpretation.
4. Intervention hypothesis requiring a prospective experiment.

### Prospective study proposal, not a claim of benefit

- Randomize **patients**, not invitation rows, to standard communication versus
  added support/navigation, subject to consent, ethics and operational approval.
- Predeclare **180-day attendance** as the primary endpoint; 90/365 days are
  secondary with a multiplicity plan. The choice must be confirmed before
  enrollment, not selected retrospectively from significance.
- Stratify randomization by prespecified segment and/or frozen risk category
  if support permits; maintain one assignment per patient across invitations.
  Define one index invitation and the policy for subsequent invitations so
  repeated exposures do not create competing assignments or double counting.
- Analyze intention to treat at patient level; predeclare censoring,
  ascertainment, contamination, repeated-invitation handling and missing outcomes.
  Report absolute attendance-rate differences and confidence intervals, not
  merely risk-model AUC.
- Before calculating sample size, obtain an empirical baseline attendance rate,
  the minimum worthwhile absolute improvement, allocation ratio, desired power,
  significance level, follow-up loss and planned subgroup analysis constraints.
  Use a two-proportion/patient-level power calculation with sensitivity analysis.
  No effect size or sample size has been invented here.
- Collect channel availability/preferences, reasons for declining, actual
  appointment access/capacity and intervention delivery/adherence. A territorial
  association alone is not proof of an access barrier. Predicted nonattendance
  risk is not predicted treatment benefit. No causal/uplift model is fitted
  without randomized intervention assignments.

## Prediction and persisted artifacts

Only load trusted joblib files: Python serialization can execute arbitrary code.
Bundles store all three fitted components, class/output semantics, preprocessing,
category vocabulary, calibration, feature definitions, aliases/CSV format,
history policy, fit/calibration cutoffs, manifests/fingerprints, seeds, warnings,
environment and implementation hashes. No patient-history dataframe is embedded.
Fit-ID/person hashes are audit metadata only and never predictors.

```bash
# Replace with explicitly approved local sample paths.
python mammography_pipeline.py predict \
  --model-path artifacts/quick/model_bundle.joblib \
  --data /path/to/new_invitations_sample.csv --data-mode sample \
  --history-data /path/to/history_sample.csv \
  --data-as-of "$PREDICTION_DATA_AS_OF" \
  --output artifacts/quick/predictions.csv
```

Prediction requires dispatch IDs/person IDs/date and each trained source field,
not target columns for default features. It accepts arbitrary input row order
and restores it. Longitudinal artifacts require external history or explicit
`--partial-history`; missing history is never silently converted to a complete
zero history. Target-free inference for historical-outcome artifacts is not
promised. Longitudinal evaluation requires the complete original input history.

The first output columns are:

```text
invitation_id,person_id,invitation_date,p_nonattendance_90d,p_nonattendance_180d,p_nonattendance_365d
SYNTH-EXAMPLE,SYNTH-P-EXAMPLE,2024-01-01,0.80,0.70,0.60
```

The invented row illustrates the schema, not an actual prediction. Additional
named columns provide raw/calibrated components, attendance complements,
model version and history-quality diagnostics. Bundle version, code hash and
recorded package versions must match; restore the environment or retrain.
Model/JSON saves use same-directory temporary files and atomic replacement.

## Resources, full-data path and existing Azure execution

| Preset | Default allocated search budget | Policy |
|---|---:|---|
| `smoke` | 60 s | One tiny configuration per requested family |
| `quick` | 120 s | Two configurations per nontrivial family, small networks/trees |
| `standard` | 1,200 s | Larger finite spaces and estimator iteration limits |
| `extensive` | 3,600 s | At least one hour allocated by default; configurable longer |

Budgets apply to **search launch**, not loading/features/refit/reports/explanation.
Active fits finish with bounded iterations; hard per-fit cancellation is not
implemented. The global deadline stops launching new trials, and incomplete
coverage is reported. A broad round-robin explores requested families before
deeper settings, but no claim is made that an arbitrarily short deadline can
complete every family. Search can honestly finish before its allocation.

Full mode is guarded before opening data and requires **both** flags below.
The command is for the user later; it was **not executed**:

```bash
python mammography_pipeline.py benchmark \
  --data /path/to/full.csv --data-mode full --allow-full-data \
  --preset extensive --search-method random --budget-seconds 3600 \
  --max-trials 300 --outcome-data-end "$VERIFIED_OUTCOME_END" \
  --reporting-lag-days "$VERIFIED_OR_ASSUMED_REPORTING_LAG_DAYS" \
  --assume-t0-snapshots --targets 90,180,365 \
  --models all-available --feature-sets raw_snapshot,snapshot_engineered,longitudinal_no_labels,all_safe_no_geography,all_safe_no_calendar \
  --cohort-policy horizon --n-jobs 8 --memory-mb 12000 --chunk-size 100000 \
  --output-dir artifacts/full_hour --save-model
```

For several hours use, for example, `--budget-seconds 14400`. Before that run,
verify outcome ascertainment and predictor availability with the source owner.
Do not supply an insurance-eligibility date as a substitute.

The full reader selects approved columns and reads in chunks, **then
materializes and globally sorts the selected frame**. Histories are not computed
independently per chunk: cross-chunk correctness is tested. This is a bounded
in-memory path, **not out-of-core training**. It rejects data exceeding raw-frame
or encoded-matrix guards; these are not a hard process-RSS limit and cannot
promise arbitrary full-cohort feasibility. Allow headroom for raw frames,
history features, sparse encodings and estimator copies on a 32 GB machine.
Dense expansion is checked before conversion; one large candidate runs at a time.
Large native-model matrices and optional estimator internals still need memory
monitoring. Prefer fewer simultaneous features, lower model budgets or a
suitable larger existing CPU environment if guards fail.

For an **already provisioned** suitable Azure CPU host: copy only authorized
inputs through an approved channel, restore this environment, put artifacts on
an access-controlled persistent disk, set explicit CPU/RAM limits, and run the
same CLI. Preserve the input/config/code versions for resume. Inspect actual
quota, RAM, local-disk capacity and permissions yourself; no subscription,
credentials, VM size, price or available quota is assumed. No resources were
created and no paid jobs started. Neither CUDA nor integrated-GPU acceleration
is assumed. Checkpoint storage grows with trials/folds and can exceed the final
model size substantially; plan disk space and remove only known run directories.

Datasets, caches, models, prediction files and generated reports live outside
committed deliverables under ignored `artifacts/`; do not commit or upload
patient-level data. Ignore patterns also cover the sample/full source filenames. The full CSV was
already tracked in the starting repository and was left unchanged; ignore
patterns do not untrack pre-existing files. This implementation made no commits.

## Methodological grounding and limitations

| Source | Finding/principle actually used | Formula or transfer limitation |
|---|---|---|
| Local dictionary, 29 September 2026 | Invitation granularity, selected insurance/anonymization cohort, predictor windows, recorded 90/180/365-day screening procedures | Authoritative for schema; no verified outcome end or backfill latency |
| Local updated project brief | Predict response to invitation; segmentation/support are hypotheses needing experiment | No causal interpretation of predictive explanations |
| [Johns, Moss and Trial Management Group (2010)](https://doi.org/10.1258/jms.2010.009091) | Supplied reference reports prior attendance predicting subsequent uptake in the Age trial | Motivates separately removable historical behavior; not proof of transfer or a causal effect; PubMed fetch was cookie-blocked here |
| [Baskaran et al. (2011)](https://doi.org/10.1109/TITB.2010.2103954) | Supplied verified bibliographic record concerns screening-attendance ML | Background only; inaccessible detailed methods were not used to justify an architecture |
| [Ma et al. (2025)](https://doi.org/10.3389/fmed.2025.1644287) | Supplied study examines contextual/social determinants in a US health system | Available age/territory can be tested; unavailable socioeconomic/access variables are not invented |
| [Vives et al., M-TICS (2024)](https://doi.org/10.1371/journal.pone.0306720) | Primary article accessed: text invitations were noninferior to letters for prior screenees in that trial | Both groups had scheduled appointments and reminders; not equivalent to VZP invitation response, and no effect estimate is transferred |
| [Allgood et al. (2016)](https://doi.org/10.1038/bjc.2015.451) | Search located a randomized postal-reminder trial | Primary webpage returned HTTP 406; no new quantitative claim adopted; scheduled-screen context limits relevance |
| [sklearn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) | Fit learned transforms only in allowed training partitions | This implementation adds explicit invitation/label-availability dates |
| [sklearn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) | Sequential splitting API context | A row gap is not a calendar/maturity gap; this workflow uses custom date folds |
| [sklearn calibration](https://scikit-learn.org/stable/modules/calibration.html) | Assess reliability and calibrate on independent predictions | Chronological mature calibration windows replace random internal CV |
| [sklearn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) | Proper probability scores answer a different question from AUC/threshold accuracy | Primary selection is validation log loss; AP must be compared with prevalence |
| [sklearn cyclic features](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html) | Cyclic/calendar encoding patterns | Birthday-month invitation selection can explain observed calendar effects |
| [sklearn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html) | Held-out predictive diagnostics; correlated inputs complicate interpretation | Derived-column group permutations are explicitly diagnostic |
| [SHAP causal caution](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html) | Predictive explanations are not intervention evidence | A patient-randomized support experiment is proposed instead |
| [MLP API](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), [histogram boosting API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) | Bounded iterations; internal early-stopping behavior must be explicit | Installed sklearn 1.8.0 factories are tested with `early_stopping=False` |

Registry formulas for logs, fixed bins, ratios, calendar interactions and snapshot
summaries are **engineering hypotheses**, not published validated mammography
scores. All associations/metrics remain sample-specific until replicated with
known sampling and valid temporal coverage.

Known limits beyond the data blockers: no exact event-time survival model;
the optional interval benchmark is validation-only, not a deployed joint bundle;
no calendar-window or exponentially decayed snapshot histories; no trained
response streaks; no geography-by-engagement expansion; no automatic invariant
feature removal; no tuned outreach threshold; no cluster-adjusted interval for
each tiny segment; no hard worker-process fit cancellation; no continuous
Optuna search; and no optional-estimator integration verification. These should
not be inferred from the presence of registry tags or CLI options. None is a
claim of full-cohort readiness or of improved participation.
