# GitHub Copilot implementation prompt: mammography screening participation

You are the implementation agent for our Rakathon project. Act as a senior Python engineer and an applied machine-learning researcher. Build, run, and verify the project described below in the current repository. Deliver working code, tests, reproducible experiments, and useful documentation, not merely a plan or illustrative snippets.

All code identifiers, comments, docstrings, CLI help, documentation, and reports must be in English. Keep comments concise and explain non-obvious decisions rather than narrating the code. Use the repository's existing conventions and instructions. Preserve unrelated work.

The requirements below are our engineering specification. Proposed features, architectures, and hyperparameter ranges are hypotheses to test, not established findings about our dataset. The local, updated data dictionary takes precedence over the provisional schema described here.

## 1. Objective and scope

Predict whether a woman who receives an invitation will NOT undergo mammographic screening within 90, 180, and 365 days after the invitation. This is participation after an invitation, not a missed appointment at an already scheduled time.

For each invitation, return a named three-value probability vector in exactly this order:

```text
[p_nonattendance_90d, p_nonattendance_180d, p_nonattendance_365d]
```

Three separately optimized binary classifiers are acceptable and should be the initial implementation. Combine them behind one prediction interface. They may use different model families, hyperparameters, preprocessing, and feature sets. Also support comparison of coherent joint approaches where practical.

The immediate priority is reliable probability estimation and strong, interpretable features. The longer-term objective is to discover associations, characterize meaningful patient segments, and formulate communication or support interventions to evaluate in a subsequent experiment. Do not claim that model explanations establish causes or intervention effects.

Deliver the complete core workflow first: schema validation, exploratory analysis, point-in-time features, chronological model selection, model comparison, probability evaluation, persistence, inference, explainability, and tests. Add a concise segment analysis and intervention-experiment proposal after the core works. A web application, deployment, automated outreach, and cloud provisioning are outside this task.

## 2. Non-negotiable development constraints

### Data access

- Work only with the explicitly provided sample CSV and documentation during development.
- The full CSV is too large for this agent session. Do not open, scan, hash, profile, copy, upload, preview, or train on the full file, even partially. Do not run a command that discovers and reads every CSV automatically.
- Reading repository filenames or file-size metadata is allowed. Confirm the sample path from repository context; ask only if it is ambiguous.
- Implement a scalable full-data execution path for the user to run later. Require an explicit `--data-mode full --allow-full-data` combination for that path. This implementation requirement is NOT permission for you to execute it now.
- Do not send patient-level data, identifiers, or sample rows to external services. Literature searches must use generic methodological queries.
- Do not commit datasets, model binaries, patient-level predictions, caches, or generated reports containing row-level data. Add appropriate ignore patterns without overwriting existing ones.
- Synthetic data are allowed for tests; mark them as synthetic and never mix synthetic metrics with empirical sample results.

### Runtime and hardware

- Local machine: Intel Core Ultra 9, 32 GB RAM, integrated GPU. CPU is the default. Do not assume CUDA or usable GPU acceleration.
- Azure is available, but its actual compute, quota, subscriptions, and credentials are unspecified. Prepare portable instructions for an existing suitable Azure CPU environment; do not create resources or start paid jobs.
- Aim for individual quick-mode fits to take seconds or tens of seconds on the sample. Measure rather than promise runtime.
- Provide a serious full-data search configuration with at least a one-hour allocated search budget, configurable to several hours. The one-hour budget is for the overall experiment, not a requirement that every fit lasts one hour.
- Never waste time with sleeps or useless repeated fits just to reach an hour. If meaningful experiments finish earlier, report that fact and actual elapsed time.

### Single-file application

Put all production Python implementation in `mammography_pipeline.py`, with modular functions/classes, type hints, dataclasses where helpful, and:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

Separate test files under `tests/`, dependency files, README, configuration examples, and generated artifacts are allowed. Do not split production logic into multiple Python modules or hide it in notebooks. Importing the main file must not read data, train models, or parse CLI arguments.

## 3. Inspect and understand the data before modeling

Read the updated data-interface document and project brief, then inspect the sample. Produce a schema audit before fitting anything. Use robust local readers for DOCX/PPTX if needed; data-dictionary interpretation happens during development and should not require an LLM at runtime.

The provisional interface describes one row as one invitation at `INVITATION_DATE = T0`, with repeated invitations linked by `PERSON_ID`. The earlier document describes roughly 1.34 million invitations from 2020–2024 and about 508,000 people, but verify actual scope from the current documentation. Do not assert that the sample is representative.

Potential fields include:

| Role | Provisional fields | Interpretation to verify |
|---|---|---|
| Keys | `INVITATION_ID`, `PERSON_ID`, possibly `RC` in documentation only | Technical identifiers; exclude from model inputs. |
| Prediction time | `INVITATION_DATE` | Invitation dispatch date; all features must be available at this time. |
| Invitation history | `PRIOR_INVITATION_COUNT`, `INVITATION_SEQUENCE`, `MONTHS_LAST_INVITATION`, `FIRST_INVITATION` | Check whether counts exclude/include the current invitation and whether they cover history before the extract. |
| Demographics/geography | `AGE`, `AGE_BAND`, `VZP_TERRITORIAL_AREA` | Exact age may not be delivered. Territorial area is not necessarily an administrative district. |
| Insurance | `INSURANCE_DURATION_YEARS`, `INSURANCE_DURATION_BAND`, `INSURANCE_DURATION_DAYS` | May be redundant or nearly constant after cohort filtering. |
| GP registration | `GP_REGISTERED`, `REG_DURATION`, `GP_REG_DURATION_YEARS`, `GP_REG_DURATION_BAND` | Resolve actual column names and units from the updated dictionary. |
| Gynecology registration | `GYN_REGISTERED_FLAG`, `GYN_REG_DURATION_YEARS`, `GYN_REG_DURATION_BAND` | Distinguish registration from observed preventive attendance. |
| Mammography history | `PRIOR_SCREEN_COUNT_5Y`, `MONTHS_SINCE_LAST_SCREEN`, `NEVER_SCREENED_BEFORE_FLAG` | Verify five-year lookback and what a missing recency actually means. |
| Other prevention | `GP_PREVENTION_24M_CNT`, `GYN_PREVENTION_24M_CNT`, `DENTAL_PREVENTION_24M_CNT`, `OTHER_SCREENING_36M_CNT` | Counts from differing lookback windows; not interchangeable event totals. |
| Healthcare use | `HEALTHCARE_CONTACT_12M_CNT` | Earlier dictionary defines contacts using unique provider/day combinations, not necessarily visits or distinct providers. |
| Outcomes | `SCREENING_3M_FLAG`, `SCREENING_6M_FLAG`, `SCREENING_12M_FLAG` | Recorded screening within 90/180/365 days after T0. These are labels, not current predictors. |

Important interpretation requirements:

1. Use only columns actually delivered. The old document includes both delivered and undelivered columns, distinguished visually; plain text extraction can lose this distinction.
2. A field named `NEVER_SCREENED_BEFORE_FLAG` must not automatically be interpreted as lifetime nonparticipation if its source only observes five years. Label outputs as "no screening observed in the documented lookback" when appropriate.
3. No previous row in the extract does not imply first-ever invitation, no prior screening, or no healthcare history.
4. An extraction limited to 2020–2024 cannot reconstruct all earlier invitations. Reconcile source-provided lifetime/history counters with observed extract counts, without overwriting one with the other.
5. Differences between overlapping rolling counts are changes in rolling summaries, not exact counts of new events. Do not infer precise screening dates from coarse recency fields.
6. Invitations reportedly occur around the person's birthday month. Calendar effects can reflect this selection or operational batching, not causal effects of the invitation month.
7. Continuous-insurance eligibility and anonymization filtering define a selected cohort. Report this limitation; do not generalize automatically to all women or all insurers.
8. Insurance eligibility through a date is not proof that outcome ascertainment is complete through that date.
9. The exact clinical/procedure-code definition comes from the current dictionary. Do not invent diagnoses, appointment status, delivery status, preferred communication channels, distance to care, socioeconomic variables, or clinical severity.

Build an explicit mapping from source names to canonical names. Detect normalization collisions, ambiguous aliases, unexpected labels, duplicate columns, invalid dates, unsupported binary encodings, missing identifiers, impossible negative counts, and contradictory records. Support documented `A/N`, `Y/N`, Boolean, or 0/1 encodings, without treating unknown values as zero. Keep IDs as strings to preserve their original representation.

Build model inputs from an audited allowlist, never from "all columns except the selected target." Exclude all three current-invitation labels, every identifier, post-invitation fields, and unknown extra columns unless their meaning and availability are explicitly approved. A label for another horizon is still a label, not a predictor.

Required output: a schema report with role, dtype, units, missingness, cardinality, time reference, source definition, inclusion decision, and unresolved questions for every input column. Critical ambiguity in a target, timestamp, identifier, or temporal-availability rule must fail clearly instead of being guessed. Optional unavailable features should be skipped with an explicit reason.

## 4. Targets, observation masks, and probability coherence

For horizon `h` in `[90, 180, 365]`, define:

```text
y_nonattendance_h = 1 - screening_flag_h
```

Positive class 1 always means nonattendance. Retrieve probabilities using the estimator's class mapping, not an assumed array position. Document label and output semantics in saved artifacts.

The outputs describe nested cumulative events, not three mutually exclusive classes. Therefore they do NOT sum to one. They must satisfy:

```text
1 >= p_nonattendance_90d >= p_nonattendance_180d >= p_nonattendance_365d >= 0
```

Observed attendance labels should satisfy `screening_90d <= screening_180d <= screening_365d`. Validate this wherever relevant labels are observed. Quarantine contradictory rows or stop with diagnostics; do not silently repair labels.

Maintain a separate validity/observation mask per horizon. Never turn missing, immature, or unobserved outcomes into nonattendance. Require a verified outcome-data-end date or explicit documented per-row follow-up information. If this is unavailable, require configuration and do not claim valid temporal evaluation. Do not estimate this date from the latest invitation.

For the baseline with independent classifiers:

- Preserve raw outputs, calibrated outputs if available, and final coherent outputs.
- Implement a deterministic row-wise projection onto the nonincreasing probability cone, such as equal-weight pooled adjacent violators across the three horizons. The projection uses only the predicted vector, never labels or statistics learned from the evaluation set.
- Do not merely sort the vector: sorting reassigns values to different horizon identities.
- Report the frequency and magnitude of raw violations and the change in scoring metrics after projection.
- Calibration can change after projection. Assess the final deployed vector as well as the raw component models. Do not assume calibration is preserved.
- If parameters of a coherence procedure are ever learned, learn them only from permitted development data.

As an optional joint benchmark, derive four interval classes from fully observed, consistent labels: screening within `(0,90]`, `(90,180]`, `(180,365]`, or no recorded screening by 365 days. A four-class probability model gives coherent outputs by summing the appropriate class probabilities. Do not invent exact event times. Use a common fully observed cohort for fair comparison with independent models.

## 5. Temporal evaluation and leakage prevention

### Define availability explicitly

Separate event time, predictor availability time, label availability time, model-fit time, and evaluation time. With only aggregate flags, use the conservative rule:

```text
label_available_at(i,h) = invitation_date(i) + h days + outcome_reporting_lag
```

Use documented reporting delays when known. A zero-lag setting is an explicit assumption, not a verified property. If actual outcome/availability timestamps exist, any less conservative policy must be documented and tested. Without them, do not make positive labels available early while holding negative labels back.

Audit predictor latency too: an event dated before T0 can have been recorded later. When the extract lacks recording timestamps, document that limitation and the assumption behind using retrospectively assembled T0 snapshots. An outcome-reporting lag alone does not solve retrospective predictor backfill.

At any simulated training origin `O`, use supervised training row `i` for horizon `h` only when:

```text
invitation_date(i) < O
label_available_at(i,h) < O
```

Use a conservative exclusive boundary consistently. Evaluation labels must have matured by the verified data-observation cutoff, independently of whether their values happen to be populated.

### Untouched final test

- Default to the latest complete eligible calendar year of invitations with full 365-day follow-up for the common three-horizon evaluation. Prefer an explicit `--test-year` or `--test-start/--test-end` when provided.
- If the data confirm adequate coverage, 2024 is a candidate test year, not an unconditional default.
- Fix model-training origins before the first test invitation. Keep models and learned transforms fixed during the test interval by default.
- Reserve a final test manifest before any target-driven EDA, feature selection, hyperparameter search, calibration-method selection, threshold tuning, or segmentation decisions.
- All experiment selection happens on earlier chronological validation data. Test metrics cannot select the champion or trigger another iteration.
- Freeze all comparison candidates before a final test pass. A final table may include predeclared family champions, but its ordering must not silently replace validation-based selection.
- Structural checks of held-out rows are permitted. Do not inspect test outcomes for modeling decisions.

### Chronological validation

Implement a timestamp-aware, expanding-window split generator with explicit date boundaries and horizon-specific outcome maturity. Do not rely on random `train_test_split`, shuffled K-fold, or an unmodified row-count `TimeSeriesSplit` as proof of temporal validity. A `gap` measured in rows is not a 365-day gap in an invitation dataset.

For every validation fold, train only on labels available before the first validation prediction date. When designing the fold, ensure validation outcomes would be available by the outer model-selection deadline. Apply the same principle to nested early-stopping, calibration, and threshold-selection data.

Support horizon-specific training eligibility to use more recent labels for shorter horizons. Also support a common 365-day-mature training cohort for controlled comparisons. Report which policy each experiment uses. Score three-horizon bundles on a common fully observed validation cohort; also report horizon-specific supplementary scores and sample counts.

Do not assume the immediately preceding calendar year can be used for 365-day model tuning at the start of the test year. Derive feasible windows from availability. If the sample does not span enough time, report insufficient data. Do not fall back to random splitting or label it valid because the pipeline executes.

### Repeated patients and history

The primary estimand is future invitations in the same service, where some patients have been seen before. The same person may appear in earlier training and later test data if every input is valid at the later invitation. Patient overlap alone is not automatically leakage.

- Exclude identifiers and their encodings/hashes from predictors.
- Keep duplicate invitations from crossing splits. Detect same-person/same-time ambiguity before aggregation.
- Build history only from strictly earlier timestamps. All rows at the same timestamp are excluded from each other's history unless a real sequence is documented.
- A fixed model may use patient history available at each test invitation time, including earlier test-period predictor snapshots. This is predeclared sequential feature availability, not model refitting.
- Current and future test targets must never enter features. Historical label-derived features require the additional explicit availability rules below and are disabled by default.
- Add a separately labeled patient-disjoint temporal stress test, when feasible, with no supervised-fit patient overlap. Do not replace the main operational evaluation with it or mix their scores. Describe what history is allowed for held-out people.
- Report performance separately for seen and unseen patients, first versus repeat invitations, and people with sparse observed history.
- For uncertainty estimates, resample patients with all of their invitations together; naive row bootstrap understates dependence. Add temporal-block sensitivity where feasible.

### Learned transformations

Fit imputation, scaling, category vocabulary, rare-category rules, feature selection, data-driven cutpoints, clustering, target/frequency encodings, and calibration only on their allowed training partitions. Build them into pipelines or equivalent fold-scoped components.

Pure row-local transformations and strictly causal history assembly may run over a chronological stream before slicing, provided they do not fit population statistics and pass the future-invariance tests. Do not reset legitimate history at split boundaries.

Disable hidden random internal validation: for example, use `early_stopping=False` for scikit-learn MLP and histogram boosting unless explicit chronology-safe early-stopping data are implemented. Random minibatch order within an already valid training partition is acceptable; random temporal evaluation is not.

## 6. Feature engineering: registry, families, and provenance

Create a registry rather than scattered column lists. Every feature must declare:

```text
name, description, source_columns, formula, units, dtype,
semantic_groups, transformation_tags, required_history,
availability_rule, missing_value_meaning, leakage_risk,
interpretation_caveat, evidence_or_hypothesis, enabled_status
```

Semantic groups and transformation tags may overlap. Deduplicate a selected feature by canonical name, even if several groups include it. Feature-set presets are independently selectable combinations of registry entries; they may overlap too.

Implement reasonable features from the following families when the source information exists. Use vectorized operations, sorted group operations, or as-of joins; avoid quadratic self-merges and row-wise Python over the full dataset.

### A. Audited snapshot baseline

Keep the available, trusted T0 predictors with minimal transformations. This is the control against which every engineered family is measured. Preserve raw continuous measurements where useful; do not replace them entirely with bins.

### B. Demographics and geography

- Age as numeric, documented ordered age band, and categorical age-band representation as alternative encodings.
- Explicit nonlinear age candidates, such as a low-degree term or spline for an interpretable linear model, with fold-local learned knots if used.
- Territorial-area categorical encoding, unknown-area indicator, and selected area-by-age interactions.
- Never reconstruct exact age from coarse bands; if a midpoint approximation is explored, name it as an approximation and handle open-ended bands explicitly.
- Geographic features are proxies for service context. Do not relabel territorial areas as deprivation, travel distance, or rurality without a verified external mapping.

### C. Calendar and cohort

- Invitation month, quarter, day of week if meaningful, and a relative year/time trend.
- Sine/cosine month encoding with a 12-month period, compared with month one-hot encoding.
- Alternative time encodings must be ablated rather than assumed helpful in combination.
- Consider calendar-cohort interactions only when sample support exists.
- Absolute year and region can capture dataset or policy shifts; compare models with and without them.
- COVID-era or policy-period indicators require verified dates and must remain explicit exploratory hypotheses. Do not invent historical change points.

### D. Screening history and recency

- Audited prior five-year screen count; zero/one/multiple categories; `log1p` count.
- Time since last screening, its missingness, and a separate documented absence-of-screening-in-lookback flag.
- Prespecified recency bins and continuous recency variants. Derive any data-driven bins only inside training folds.
- A 24-month reference-gap feature may be explored as a transparent behavioral hypothesis; do not present it as verified eligibility or a clinical recommendation without current program rules and applicable age criteria.
- Interactions between recency, prior screen count, and repeat invitations.

### E. Invitation history

- Source-provided previous invitation count and invitation sequence, after semantics are confirmed.
- Explicit observed-prior-row count in the extract, distinct from source history counts.
- Days/months since previous observed invitation; elapsed time since first observed invitation.
- Distribution of strictly historical inter-invitation intervals: mean, minimum, maximum, standard deviation, and variability when enough observations exist.
- Invitation density per observed history duration with a denominator floor and an insufficient-history indicator.
- Differences between source counts and observed counts as extract-coverage diagnostics. Do not automatically use apparent discrepancies as predictive features without validating their meaning.
- Never use total future-inclusive invitations per patient or the eventual final invitation number.

### F. Preventive behavior and healthcare engagement

- Any GP, gynecology, dental, or other-screening activity in its documented lookback.
- Number of preventive domains with observed activity, with explicit missing-domain handling.
- Counts and `log1p` counts; sensible train-fitted robust scaling for heavy tails.
- Annualized rates using each field's actual 12/24/36/60-month exposure window.
- GP-plus-gynecology prevention, dental prevention as an engagement proxy, and other-screening participation.
- Healthcare-contact intensity and combinations such as high contact with no recorded prevention.
- Differences/contrasts among prevention domains; relative shares only when numerator and denominator definitions are compatible.
- If comparing prevention counts with contacts from another window, annualize and label the result as an intensity ratio, not a probability or a bounded proportion. Protect zero denominators and preserve missingness.
- A transparent count of preventive domains may serve as an engagement score. Do not label an arbitrary weighted score validated or optimize its weights on the test set.

### G. Registration and continuity of care

- GP/gynecology registration flags; both, one, or neither registered; unknown kept distinct.
- Registration durations, documented bands, and optional logarithms.
- Registered-but-no-recorded-prevention combinations.
- Duration differences or ratios only when units and reference dates match.
- Insurance duration and its redundancy with eligibility/cohort dates. Drop invariant features using training data and record why.

### H. Longitudinal predictor snapshots

For repeated patients, use only earlier invitation snapshots:

- Last one or two observed values of selected numeric/Boolean predictors.
- Difference from the last observed value and change per elapsed year, with same-time/zero-interval protection.
- Expanding historical mean, minimum, maximum, variability, and exponentially decayed historical mean where meaningful.
- Earlier-snapshot summaries within 12/24/36-month windows, explicitly named as snapshot summaries rather than raw medical-event counts.
- Previously observed registration/area changes and their support counts.
- Historical transitions in recorded preventive engagement.
- Count and age of available historical snapshots, insufficient-history flags, and time since earliest observed record.

All aggregates must exclude the current row and future rows, and must be invariant to input ordering after deterministic chronological processing. A group total computed over all rows and copied backward is forbidden.

The sample may contain incomplete patient histories. Identify whether it was sampled by row or by person. Do not pretend missing earlier rows never existed. Distinguish "history observed in this input" from "complete history" in reports and metadata. For further downsampling of the available input, prefer seeded sampling of whole patients with all their available rows.

### I. Optional matured historical outcomes

Default: `--historical-outcomes off`.

If explicitly enabled and feasible in deployment, add earlier invitation outcomes only when the earlier invitation's outcome window plus reporting lag ended strictly before the current T0. A simple `groupby().shift()` of labels is insufficient.

Candidate features include previous mature response, count of mature previous invitations, historical response fraction, and consecutive observed mature nonresponses. Preserve unknown/pending outcomes separately; define whether an unknown breaks a streak. Do not count an immature invitation as a failure. For smoothed rates, use a fixed declared prior or an origin-safe fitted prior.

Historical response rates describe observed associations and may involve overlapping invitation windows. Do not attribute one screening event causally to a particular invitation when event-level data do not support that attribution. Keep this family separately removable and compare it with a target-free history model.

### J. Missingness, transformations, and selected interactions

- Stable missingness indicators with source-specific meanings.
- Raw, binary, `log1p`, binned, standardized, cyclic, lag, trend, and interaction tags.
- Selected interactions: age x recency; invitation count x prior screening; GP registration x GP prevention; gynecology registration x gynecology prevention; healthcare contacts x preventive engagement; region x engagement.
- Avoid an uncontrolled polynomial expansion of every variable.
- Arbitrary target encoding is excluded from the default. If added, require chronology-aware out-of-fold construction and a documented unseen-category fallback; ordinary random-fold encoding is not sufficient here.
- Never use future-inclusive cohort response rates, label proxies, person embeddings, ID-derived features, or model-derived pseudo-labels as unvalidated inputs.

## 7. Independently testable feature sets and ablations

Implement named presets, resolved through the registry, including at least:

| Preset | Purpose |
|---|---|
| `minimal` | Small, interpretable subset available at T0. |
| `raw_snapshot` | All audited raw snapshot predictors. |
| `demographics_calendar` | Demographic, territorial, and calendar context. |
| `screening_invitation` | Screening recency/count and invitation history. |
| `prevention_engagement` | Prevention domains and healthcare-contact behavior. |
| `registration_continuity` | Registration and continuity features. |
| `snapshot_engineered` | Raw snapshot plus justified transformations and interactions. |
| `longitudinal_no_labels` | Snapshot engineering plus earlier predictor snapshots. |
| `longitudinal_with_mature_labels` | Optional historical-outcome extension, explicitly enabled. |
| `all_safe` | All available, approved default features, excluding historical outcomes by default. |
| `all_safe_no_geography` | Geography ablation. |
| `all_safe_no_calendar` | Calendar/cohort ablation. |
| `all_safe_no_longitudinal` | Historical-snapshot ablation. |

Provide `--feature-sets`, `--include-groups`, `--exclude-groups`, and `--list-features`. Missing sources must produce an availability report; a preset resolving to zero usable features must fail clearly. Record the exact resolved feature names and a configuration hash in every experiment.

Implement family-wise addition/removal experiments and paired comparisons using the same folds, rows, seeds, and model budget. Distinguish feature-set comparisons holding model settings fixed from comparisons after retuning each set. Report predictive gain, computation cost, and stability. More features are not automatically better.

## 8. Exploratory analysis and data quality outputs

Use development data for target-driven exploration. Produce compact CSV/JSON summaries and saved plots plus a Markdown report:

- Rows, patients, invitations per patient, dates, duplicates, missingness, cardinality, invalid values, and extract-history coverage.
- Outcome prevalence per horizon and eligible cohort; outcome nesting and censoring/maturity audits.
- Counts and participation/nonparticipation rates by period, age band, territory, first/repeated invitation, registration, and preventive activity, with denominators and uncertainty where feasible.
- Numeric distributions, heavy tails, redundant variables, and structural versus unexplained missingness.
- Numeric associations and categorical relationships using appropriate summaries; do not blindly compute Pearson correlation on arbitrary category codes.
- Behavior profiles and selected interactions, with minimum-support rules and uncertainty. If many hypothesis tests are used, report multiplicity handling and emphasize effect sizes over tiny p-values.
- Development-period drift in features, prevalence, and model behavior. Final test drift belongs to the locked evaluation report and cannot drive retuning.
- A clear split manifest with date bounds, fit origin, eligible/excluded counts, label-maturity exclusions, unique patients, and overlap.

If the sample was selected non-randomly or its sampling mechanism is unknown, label prevalence, subgroup rates, and all predictive metrics as sample-specific. No claim of population calibration can be made from an unrepresentative sample alone.

## 9. Model families and preprocessing

Implement runnable factories, compatible preprocessing, search spaces, and comparison support for:

1. `DummyClassifier(strategy="prior")` per horizon as a mandatory baseline.
2. Regularized logistic regression as an interpretable reference.
3. A decision tree with depth/leaf regularization.
4. Random forest.
5. Extra Trees.
6. Histogram gradient boosting.
7. An MLP neural network with multiple architectures.
8. Optional LightGBM, XGBoost, and CatBoost adapters when installed.

Optional dependencies must be imported lazily. Missing an optional package should produce an explicit skipped-model entry, not break basic execution. Requested strict modes may fail with an installation instruction. Missing mandatory dependencies must fail clearly. Do not silently replace a requested family with another family.

Use model-specific preprocessing:

- For linear/MLP models: numeric imputation with missingness handling, fold-local scaling, and sparse one-hot encoding for nominal categories where supported.
- For trees: no unnecessary scaling; handle missing values according to the estimator. Ordinal encoding is acceptable only for genuinely ordered variables or with correct native-categorical handling.
- For native-categorical boosting: an alternative pipeline that preserves documented categories and handles unknown values explicitly. Do not force every model through one-hot encoding if its native categorical support is the experiment.
- Unknown categories, all-missing training columns, rare categories, and train/inference schema differences must be handled predictably.
- Avoid accidental dense expansion. Estimate matrix memory before densifying; use float32 where suitable and document a configurable memory ceiling.
- Class balancing/weights are optional experiments, not a blanket default: the main goal is probability estimation. Preserve real prevalence in validation/calibration/test. Do not apply SMOTE or resampling before splitting.
- Record convergence warnings and incomplete training honestly. Do not suppress meaningful warnings globally.

For the MLP, compare architectures such as `(32,)`, `(64,)`, `(128,)`, `(64,32)`, `(128,64)`, and `(128,64,32)`, activation (`relu`, optionally `tanh`), regularization, learning rate, batch size, and iteration budget. Use small architectures in quick mode. A PyTorch shared three-output or interval-classification network is optional after a working MLP baseline; PyTorch is not required merely to satisfy the neural-network requirement.

## 10. Hyperparameter and feature-set search

Implement an actual experiment runner, not disconnected example functions. Search over:

```text
target horizon x model family x feature set x valid hyperparameters
```

Use a small exhaustive grid for quick mode and budgeted random/Optuna search for standard/extensive runs. A full Cartesian product of all proposed values is not required and should not be attempted blindly. Add conditional search spaces so incompatible options cannot be sampled.

Illustrative search ranges to adapt to sample size and the installed API:

| Family | Candidate settings |
|---|---|
| Logistic regression | `C` from about `1e-3` to `1e2`; compatible L1/L2/elastic-net configurations; optional class weighting. |
| Decision tree | Depth 3/5/8/12 or unrestricted with sufficient regularization; leaf size 10/50/200 or scale-appropriate fractions; criterion alternatives. |
| Random forest / Extra Trees | About 50–500 trees, depth 6/12/24 or regularized unrestricted, leaf size 5/25/100, feature subsampling. |
| Histogram boosting | Learning rate 0.02–0.15; 50–600 iterations; 7/15/31/63 leaves; leaf-size and L2 penalties. |
| LightGBM | Leaves, depth, minimum child size, learning rate, trees, column/row subsampling, L1/L2 penalties; activate row subsampling with its required frequency parameter. |
| XGBoost | CPU histogram algorithm, depth 3–8, child weight, learning rate, trees, row/column sampling, L1/L2 penalties. |
| CatBoost | CPU mode, depth 4–8, iterations, learning rate, leaf regularization; valid categorical input handling. |
| MLP | Architectures above; `alpha` around `1e-5`–`1e-1`; learning rate around `1e-4`–`1e-2`; batch sizes 128/256/512; preset-dependent epochs. |

Quick-mode search must actually include at least two parameter configurations for selected nontrivial families and at least two feature sets, subject to a clearly reported runtime limit. A mandatory neural-network smoke run can be separate from the default quick leaderboard if necessary for timing. Every implemented family must have a runnable bounded smoke test.

Progressively screen broad families/feature sets, then spend more budget on promising combinations. Still allocate a minimum exploration quota to requested families so one slow model or initial winner does not consume the whole search. If the deadline prevents testing a family, say so.

For every trial, save: run/trial IDs, family, horizon, resolved features, preprocessing, hyperparameters, seed, fold dates, cohort policy, row/patient counts, validation metrics, duration, resource limits, warnings, status, and exception/timeout reason. Use append-safe CSV/JSONL or an optional local Optuna database. Checkpoint completed work and support compatible resume.

Any cache containing learned transforms or label-dependent features must be keyed by dataset identity, schema/feature version, fold boundaries, fit origin, target horizon, availability rules, and relevant settings. Never reuse a transform fitted on a later fold in an earlier fold. Only genuinely fit-free point-in-time features may share a cache across compatible folds.

Produce a validation leaderboard, the best tested configuration for each family and horizon, and the selected three-horizon bundle. Use "best tested" rather than claiming a global optimum. If a family has no successful trial, report it as such.

Never rank candidates by their final test score. Never pool scores from different validation rows without identifying the mismatch. Avoid nested CPU oversubscription: default to sequential trials with a bounded number of estimator threads, or divide threads explicitly across concurrent trials.

## 11. Scoring, calibration, and final selection

Because our immediate goal is probabilities, select primarily by chronological-validation log loss, with equal horizon weights by default. Report per-horizon values and the macro mean on common eligible rows. Brier score, calibration, discrimination, runtime, and model simplicity are additional selection evidence. Preserve the exact selection rule in the run manifest.

For each horizon and the combined vector, report:

- Log loss and Brier score; improvement over the training-prevalence baseline on identical rows.
- ROC AUC, average precision for nonattendance, and average precision for attendance using `1-p` and the complemented label.
- Prevalence and the corresponding no-skill AP baseline. Nonattendance may be common, so a high AP alone can be misleading.
- Reliability diagrams and calibration summaries with bin definitions, counts, and uncertainty when feasible. ECE is descriptive and bin-dependent, not the sole optimization target.
- Predicted-probability distribution and raw/final cross-horizon consistency.
- Secondary threshold metrics: precision, recall, F1, balanced accuracy, and confusion matrix. Report a 0.5 reference and, if needed, a threshold selected on development data only.
- Secondary outreach-ranking diagnostics: precision, recall, and lift among the top 5/10/20 percent by nonattendance risk. These are not estimates of who benefits most from intervention.
- Patient-cluster bootstrap confidence intervals for final metrics and paired differences when sample size permits. Flag small subgroups, undefined AUC, degenerate bootstraps, and single-class folds instead of returning misleading numbers.

Calibration must be operationally honest:

1. Always retain an uncalibrated baseline.
2. Offer sigmoid calibration and, only with adequate data, isotonic calibration.
3. Use explicit chronological calibration data and mature labels. Do not accept a calibrator's default random/stratified cross-validation as a safe temporal policy.
4. A simple valid route is a base model trained before a dedicated later calibration window, then a calibrator fitted on that window, with the base model and calibrator frozen together for test. Account for outcome maturity at both stages.
5. Keep hyperparameter/method selection separate from any window used to claim calibration validation. If data are too short for the planned hierarchy, simplify transparently or use no calibration.
6. An optional chronological out-of-time calibration/refit approach is acceptable only if its data flow is explicit and tested. Do not fit a calibrator, refit the underlying model on additional data, and silently assume the old calibrator still applies.
7. Select raw versus calibrated pipelines using permitted development data, not the final test. Record the training-data reduction or additional assumptions introduced by calibration.
8. Apply the chosen coherence rule after component calibration and evaluate that complete deployed mapping.

For `calibration=none`, the selected configuration may be refitted on all eligible pre-test labels at the declared fit origin. For a frozen holdout-calibrated model, retain the specified base/calibration separation. Later retraining for deployment must produce a new artifact and cannot inherit old test metrics as if they evaluated the new fit.

No minimum AUC or accuracy is guaranteed. Weak results must be reported honestly. Sample experiments establish pipeline functionality and provisional findings, not full-cohort performance.

## 12. Explainability and feature relationships

Provide model-specific and model-agnostic explanations with reproducible bounded computation:

- Grouped permutation importance on held-out development data, with repeated permutations and variability. Use feature-family ablation as a complementary measure, especially for correlated predictors.
- Keep one-hot columns grouped under their original field. For derived features, either perturb a valid source and recompute dependent features, or explicitly label transformed-column permutation as a diagnostic that may create unrealistic combinations.
- SHAP for supported finalists on a small, seeded sample; use training-only background data. Limit rows and time, and skip cleanly if unavailable.
- Map encoded feature names back to original/engineered features and semantic families. Aggregate related representations transparently.
- Standardized logistic coefficients with the encoding and reference categories stated; coefficients are conditional associations, not direct causal effects.
- Selected dependence/interaction plots with data support shown. Prefer methods/interpretations that acknowledge correlated features; do not present unrealistic counterfactuals as actionable evidence.
- A few local example explanations without raw identifying data.
- Compare importance and direction across horizons, folds, model families, and ablations. Highlight unstable effects.

Specify exactly what is being explained: an uncalibrated logit, an uncalibrated probability, a calibrated component, or the final coherent vector. Tree SHAP for a base estimator does not automatically explain a calibrator plus cross-horizon projection. If explaining the final wrapper is too expensive, label base-model explanations honestly and provide final-wrapper permutation/ablation diagnostics.

Every reported finding must distinguish:

```text
Observed association in the data
Association used by this fitted predictive model
Tentative behavioral interpretation
Intervention hypothesis requiring a prospective experiment
```

Do not write "feature X causes nonattendance" or recommend manipulating a variable solely because of its SHAP value. Predicting high nonattendance risk is not the same as predicting high treatment benefit.

## 13. Segments and later intervention design

After the validated predictive workflow works, produce a concise exploratory segment report. Start with transparent feature-based rules and optionally a training-fitted shallow surrogate tree or clustering of standardized behavioral features. Avoid clustering patient IDs or using current/future outcomes as segmentation inputs.

Candidate hypotheses include: little observed healthcare contact; regular healthcare use but little prevention; repeated invitations with no mature observed response; prior screening but long recency; registration without recorded prevention. These are starting hypotheses from the brief, not findings to force onto the data.

For each supported segment report its rule, support, horizon-specific predicted/observed risk on allowed evaluation data, defining features, uncertainty, and limitations. Do not call a territorial association a proven accessibility barrier. For an intervention hypothesis, list what additional information is needed, such as channel availability, contact preferences, or appointment-access data.

Draft an A/B-test proposal: patient-level assignment to standard communication versus added support, stratification by segment/risk if appropriate, one predeclared primary horizon, attendance as the outcome, intention-to-treat analysis, handling of repeated invitations, and a power-analysis plan requiring a baseline rate and minimum worthwhile effect. Do not invent an effect size, claim benefit, or implement causal/uplift estimates from data with no intervention assignment. Do not contact anyone.

## 14. CLI and program structure

Use `argparse` subcommands. Provide at least:

```text
inspect       Validate schema and summarize the sample.
eda           Produce permitted development-data analyses.
features      Build/audit features and optionally export an approved local cache.
benchmark     Compare model families, parameters, and feature sets on validation.
train         Train a specified or validation-selected configuration.
evaluate      Evaluate a saved, fixed artifact on a declared eligible holdout.
predict       Score new invitations without requiring target columns.
explain       Explain a saved model using permitted data.
self-test     Run fast built-in diagnostics or dispatch the test suite.
```

Recommended internal boundaries, adapted as needed:

```text
parse_args / resolve_config / validate_paths
load_sample_csv / stream_full_csv / normalize_schema
validate_schema / validate_targets / compute_label_availability
build_row_features / build_patient_history / build_feature_registry
resolve_feature_set / make_temporal_folds / audit_split
build_preprocessor / build_model / build_search_space
run_trial / compare_models / run_feature_ablations
fit_calibrator / project_horizon_probabilities
evaluate_probabilities / bootstrap_by_patient
explain_model / summarize_segments / write_report
save_bundle / load_bundle / predict_bundle / main
```

Expose configuration for data path/mode, dictionary/schema mapping, CSV format, output directory, targets, verified outcome-data-end, reporting lag, test dates, validation dates/folds, common versus horizon-specific training cohorts, history mode, feature sets, model families, search method, budget, seeds, threads, sample cap, calibration, bootstrap/explanation limits, persistence, resume, and verbosity.

Examples of intended usage; make the actual help and implementation agree. Environment values such as `$VERIFIED_OUTCOME_END` must be supplied from the verified dictionary/configuration, not guessed:

```bash
python mammography_pipeline.py inspect --data data/sample.csv --data-mode sample

python mammography_pipeline.py benchmark \
  --data data/sample.csv --data-mode sample --preset quick \
  --outcome-data-end "$VERIFIED_OUTCOME_END" \
  --models logistic,decision_tree,hist_gradient_boosting \
  --feature-sets raw_snapshot,snapshot_engineered \
  --targets 90,180,365 --budget-seconds 120 --n-jobs 4 \
  --output-dir artifacts/quick --save-model

python mammography_pipeline.py benchmark \
  --data data/sample.csv --data-mode sample --preset quick \
  --outcome-data-end "$VERIFIED_OUTCOME_END" \
  --models mlp --feature-sets snapshot_engineered \
  --budget-seconds 90 --output-dir artifacts/mlp_smoke

python mammography_pipeline.py benchmark \
  --data /path/to/full.csv --data-mode full --allow-full-data \
  --preset extensive --budget-seconds 3600 \
  --outcome-data-end "$VERIFIED_OUTCOME_END" \
  --targets 90,180,365 --models all-available \
  --feature-sets raw_snapshot,snapshot_engineered,longitudinal_no_labels,all_safe \
  --n-jobs 8 --output-dir artifacts/full_hour --save-model

python mammography_pipeline.py evaluate \
  --model-path artifacts/quick/model_bundle.joblib \
  --data data/sample.csv --data-mode sample --split test \
  --output-dir artifacts/quick/final_evaluation

python mammography_pipeline.py predict \
  --model-path artifacts/quick/model_bundle.joblib \
  --data data/new_invitations_sample.csv --data-mode sample \
  --history-data data/history_sample.csv --data-as-of "$PREDICTION_DATA_AS_OF" \
  --output predictions.csv

python mammography_pipeline.py self-test
pytest -q
```

These commands are illustrative: replace paths with verified sample paths before executing sample commands. The full-data command is documentation for the user only. `evaluate --split test` must reproduce a saved holdout definition, check training/calibration overlaps, and reject accidental evaluation on training rows unless explicitly labeled as an in-sample diagnostic. Prediction supports optional history input and a declared availability cutoff; it must not require labels for default features.

Persist CSV format and schema configuration so inference does not guess how to parse a file differently. Do not evaluate automatically merely because `benchmark` completed; require the explicit final-evaluation step after freezing choices, or an explicit documented `--evaluate-final` switch.

## 15. Resource presets and scale-up plan

Provide documented presets with overridable limits:

| Preset | Intended scope | Starting resource policy |
|---|---|---|
| `smoke` | End-to-end correctness on synthetic or a very small valid sample. | One tiny configuration per requested family, short fits, minimal plots. |
| `quick` | Sample-only iterative development. | Small architectures/trees, bounded trials, a few feature sets, roughly 60–180 seconds overall by default. |
| `standard` | Broader CPU comparison. | More valid folds/feature sets, tens of budgeted trials, roughly 10–30 minutes if authorized by the user. |
| `extensive` | User-run full-data investigation. | At least 3600 seconds allocated to search by default; more trials, all available requested families, feature ablations, checkpoint/resume. |

These are configuration targets, not measured runtime guarantees. Report loading, feature engineering, search, refit, calibration, explanation, and total wall time separately. State whether a budget applies only to search or to the entire command.

Use a monotonic clock. Implement per-fit bounds where possible and a global deadline that stops launching new work. If a library cannot enforce a hard timeout, report the soft-boundary behavior and bound iterations; use safely managed worker processes if true cancellation is needed. Preserve completed trials after interruption.

Read full data in selected columns/chunks or via an appropriate dataframe engine when useful. Chunk-local `groupby` is not sufficient for cross-chunk histories: implement persistent per-patient state or a correctly sorted/materialized approved intermediate. Do not claim out-of-core correctness without a cross-chunk test.

Default to reasonable thread limits and a memory budget below available RAM. Fit only one large candidate at a time unless parallel memory usage is bounded. Keep sparse matrices sparse; cap dense MLP/native-model matrices; use model-compatible batches or explicit training subsets if necessary. Document sample size per trial and compare shortlisted candidates on a common scale before selection.

Describe local and existing-Azure execution, dependency installation, CPU/RAM tradeoffs, disk needs, safe checkpoint location, and resume commands. Do not promise that the integrated GPU or an unspecified Azure resource will improve training.

## 16. Persistence and prediction contract

Save the complete deployable bundle, not just estimator weights:

- All three fitted pipelines, their horizon and class mappings, and optional joint model.
- Resolved feature definitions, schema aliases, category handling, preprocessing, and model parameters.
- Calibration objects and coherence settings.
- Train/calibration cutoffs, observation-date assumptions, split fingerprints, and dataset/sample provenance.
- Library/Python versions, seed, configuration, code version/hash, and training-time warnings.
- History strategy, required history fields, partial-history policy, and historical-outcome availability rules.

Do not automatically embed the entire patient-history dataset in the bundle. Either accept external past history at inference or persist only explicit bounded state needed by the chosen design, documenting its contents. Enforce train/inference feature parity. A model trained with longitudinal features must not silently receive zeros when history is absent: provide an explicit partial-history mode with appropriate flags or fail clearly.

Prediction must accept rows in arbitrary input order, internally process chronology if needed, and restore original row order. Return IDs as join keys, invitation date, three named final nonattendance probabilities, and optional raw/calibrated vectors, attendance complements, model version, and history-quality flags. Probability rows must remain aligned with input rows.

Reject incompatible schema/artifact versions with actionable errors. Load serialized Python models only from trusted paths and document that joblib/pickle artifacts can execute code. Save files atomically where feasible. Verify save/load equivalence in a fresh process, including custom classes defined in the single executable module; avoid `__main__` serialization traps.

## 17. Tests: mandatory, broad, and meaningful

Implement pytest tests with small synthetic fixtures, parameterization, and Hypothesis/property tests where useful. Keep default tests fast and independent of network, Azure, optional packages, and the full CSV. Mark optional/slow integration tests explicitly. Test contracts and failure behavior rather than merely reproducing implementation expressions.

Cover at least the following areas:

### Schema and targets

- CSV separator/encoding/date handling; IDs stay strings; name normalization collisions fail.
- Correct mappings of documented binary values, and errors for unknown encodings.
- Required versus optional fields; missing/ambiguous target/date definitions.
- Exact duplicates, conflicting invitations, duplicate indices, and same-time patient rows.
- Label inversion, fixed horizon order, contradictory cumulative flags, missing/immature labels.
- Positive-class extraction when class order differs, and explicit handling of a one-class fit.

### Feature correctness

- Hand-calculated small examples for recency, counts, safe ratios, calendar cycles, and selected interactions.
- First observed patient row; no history; partial history; missing values; zero denominators; all-missing columns.
- Correct exclusion of current/same-time/future rows from lags and aggregates.
- Overlapping-window deltas are not mislabeled as new medical events.
- Registry/set overlap does not duplicate columns; unsupported features have explicit reasons.
- Deterministic feature names/order and preprocessing with unseen categories.

### Leakage and temporal invariants

- Appending or altering future patient records cannot change any earlier point-in-time feature vector; keep already-fitted preprocessing fixed for this invariance check.
- Altering current/future targets cannot affect default feature vectors.
- Changing an immature earlier label cannot change features at a time before that label matures.
- A deliberately leaky fixture is detected by an availability/split assertion.
- Train labels mature strictly before each origin; exact 90/180/365-day and reporting-lag boundaries, including leap-year cases, are covered.
- No duplicate invitation crosses splits; same-date groups are handled consistently.
- Learned transforms do not observe validation/test rows; use fit spies or deliberately shifted data to verify this.
- Temporal early stopping, calibration, and threshold selection obey their explicit partitions.
- Changing test labels cannot change selected hyperparameters, features, calibrator choice, or model predictions when rerunning the same development process.
- Primary temporal overlap and optional patient-disjoint evaluation follow different declared contracts.
- Whole-patient sampling preserves all available selected histories; chunked/full feature construction matches on a small fixture spanning chunk boundaries.

### Models, probabilities, and metrics

- Every mandatory family performs a tiny fit/predict run for all three horizons.
- At least two neural-network architectures and two parameter settings are exercised in bounded tests.
- Output shape `(n,3)`, finite values in `[0,1]`, correct horizon order, and final monotonicity.
- Projection preserves already coherent vectors, is idempotent, and handles violations correctly; e.g. `[0.7,0.8,0.4]` projects to `[0.75,0.75,0.4]` under equal weights.
- Optional interval model yields nonnegative interval masses summing to one and the correct cumulative complements.
- Independent three-horizon probabilities are not normalized to sum to one.
- Log loss/Brier/AP use the declared positive class; single-class AUC and empty cohorts are reported as undefined with reasons.
- Bootstrap samples whole patients, preserves within-patient rows, and handles degenerate resamples.

### CLI, orchestration, and artifacts

- Import has no side effects; help and invalid-argument behavior work.
- Sample-only execution never opens a sentinel full-data path; monkeypatch file opening to make violations fail tests.
- Optional dependency absence is handled as documented.
- Fit timeout, failed trial, interrupted run, resumability, and invalid-resume configuration.
- A full tiny path: inspect -> features -> benchmark -> train/save -> fresh-process load -> evaluate -> predict -> explain/report.
- Saved/reloaded predictions match within an explicit tolerance, including custom transformers.
- Inference does not require labels and restores input row order.
- Missing required history and incompatible schema/model versions fail predictably.
- Evaluate rejects training/calibration overlap presented as a holdout.
- Report generation survives undefined metrics, empty supported segments, and unavailable optional explanation tools.

Measure branch coverage for core parsing, temporal splitting, availability, feature engineering, coherence, and persistence. Aim for high coverage, preferably at least 90% in these critical areas when practical, but never add meaningless tests merely to inflate a percentage. Report coverage and gaps honestly. Run the default test suite and a real sample smoke workflow before completion; if the provided sample cannot support valid temporal scoring, report that explicitly and use synthetic fixtures only to verify mechanics.

## 18. Literature and methodological grounding

Use primary studies and official documentation. For each adopted evidence-inspired feature family, record the source, the actual finding, the relevance to our data, and its limitations. Separate literature support from your proposed formula. Do not transfer another study's performance numbers to our dataset or assume the populations and targets match.

Starting sources verified when this prompt was prepared:

1. **Local project brief:** `VZP – Od pozvánky k mamografu.pptx`. Establishes the invitation-response objective, segmentation/support goals, and the need for prospective evaluation of intervention effects. Use the version present in the repository.
2. **Local data dictionary:** `Rakathon_Datove_rozhrani.docx`, earlier version dated 29 September 2026. Establishes row granularity, predictor lookbacks, cohort restrictions, and 90/180/365-day labels. The updated supplied version is authoritative.
3. **Johns, Moss, and Trial Management Group (2010), “Randomized controlled trial of mammographic screening from age 40 ('Age' trial): patterns of screening attendance.”** DOI: [10.1258/jms.2010.009091](https://doi.org/10.1258/jms.2010.009091); [PubMed](https://pubmed.ncbi.nlm.nih.gov/20356944/). Prior attendance predicted subsequent uptake in the studied trial cohort. This motivates testing observed participation history; it does not establish a causal effect or guarantee transfer to VZP invitees.
4. **Baskaran et al. (2011), “Predicting breast screening attendance using machine learning techniques.”** DOI: [10.1109/TITB.2010.2103954](https://doi.org/10.1109/TITB.2010.2103954); [PubMed](https://pubmed.ncbi.nlm.nih.gov/21216721/). A directly relevant early ML attendance-prediction study. Treat it as background; verify accessible methods before adopting a specific architecture or claim. The bibliographic record was verified; detailed full-text methods are not assumed here.
5. **Ma et al. (2025), “Modeling the impact of social determinants on breast cancer screening: a data-driven approach.”** [Primary article](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1644287/full), DOI: 10.3389/fmed.2025.1644287. The study examined screening behavior using demographic, geographic, access, and other information in a US health-system population. It motivates testing available contextual predictors, not inventing unavailable social determinants or equating VZP territories with the study's geographical measures.
6. **scikit-learn: common pitfalls and data leakage.** [Documentation](https://scikit-learn.org/stable/common_pitfalls.html). Supports train-only fitting of learned transformations and avoiding test-driven model choices.
7. **scikit-learn: TimeSeriesSplit.** [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). Useful API context; implement invitation-date and label-availability logic explicitly rather than treating sample-count gaps as elapsed time.
8. **scikit-learn: probability calibration.** [Documentation](https://scikit-learn.org/stable/modules/calibration.html). Supports evaluating reliability and fitting calibrators using appropriate independent predictions. Adapt splitting to this project's temporal setting.
9. **scikit-learn: scoring and evaluation.** [Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html). Probability estimation motivates proper probability scores such as log loss and Brier score; discrimination and decision metrics answer additional questions.
10. **scikit-learn: time-related feature engineering.** [Example](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html). Provides engineering patterns for comparing temporal representations. It is not evidence that seasonality improves mammography participation prediction.
11. **scikit-learn: permutation importance.** [Documentation](https://scikit-learn.org/stable/modules/permutation_importance.html). Supports held-out importance diagnostics and careful interpretation with correlated predictors.
12. **SHAP: “Be careful when interpreting predictive models in search of causal insights.”** [Official explanation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html). Predictive explanations describe learned associations; intervention recommendations need causal assumptions and suitable evidence.
13. **scikit-learn estimator APIs:** [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) and [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html). Verify current preprocessing, class-weight, sparse-input, and early-stopping behavior against the installed version.

If web access is available, add a small number of directly relevant primary studies on participation after screening invitations, prior attendance, preventive engagement, and access barriers. Do not confuse image-based breast-cancer diagnosis models, general appointment no-show models, and invitation-response models. If access is blocked, state the limitation and use only verified supplied references. Never fabricate citations, quotations, findings, or clinical definitions.

## 19. Deliverables and completion criteria

Deliver:

1. One complete production file: `mammography_pipeline.py`.
2. Meaningful tests under `tests/` plus a fast self-test entry point.
3. Minimal reproducible dependency specifications, with optional model/explanation dependencies separated where helpful, and an environment/version record.
4. A README with setup, exact sample commands, one-hour/full-data commands for the user, configuration descriptions, schemas, feature/target semantics, temporal methodology, and limitations.
5. The schema/feature registry, split audit, experiment ledger, validation leaderboard, per-family best tested parameters, ablations, probability metrics/plots, and an explainability report from runs that actually completed.
6. A reproducible model bundle when saving is requested, and an example of the prediction output schema without exposing real identifiers.
7. A compact supported-segment report and future randomized-intervention study proposal, clearly separated from predictive evidence.

Before finishing, check all of the following:

- The implementation runs, not just compiles.
- All three horizon probabilities are present and coherent in final output.
- No full-data access occurred during this agent session.
- Every feature and training label respects its temporal availability.
- The final test did not influence features, model choice, calibration choice, or thresholds.
- Feature-set comparisons and at least small parameter searches actually ran, or have explicit reasons they could not.
- Required model families exist with bounded smoke coverage, including a neural network.
- Save/load/predict work from a fresh process with the single-file design.
- Tests, actual timings, sample limitations, skipped trials, optional-dependency status, and remaining gaps are reported truthfully.
- No fabricated performance, population claim, intervention effect, or literature citation appears in the report.

Work incrementally: inspect the sample/schema, implement the temporal and feature contracts with tests, establish simple baselines, add the search/model families, then add calibration/explanations/segment reporting. Run checks after each meaningful stage. Continue through implementation and verification; do not stop after a plan. Ask only for a genuinely blocking missing source definition or path that cannot be resolved from the supplied files. Make routine engineering choices, record them, and proceed.
