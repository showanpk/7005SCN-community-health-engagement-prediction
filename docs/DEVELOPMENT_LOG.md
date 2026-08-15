# Development Log

This document records the main technical development stages of the research artefact.

## Stage 1 — Initial research view

The first analytical view combined participant demographics, attendance history and assessment information into one retention dataset.

Validation identified several issues:

- 1,633 rows represented only 1,463 unique participants.
- 170 duplicate rows were present.
- a future session dated 30 December 2026 entered the attendance history;
- the outcome used `DaysSinceLastAttendance <= 90`, while the same information was available to potential predictors, creating target leakage;
- `AttendanceRate` had no variation in the extracted dataset;
- implausible health values were present, including extreme weight and BMI values;
- some historical attendance preceded the CRM `CreatedAt` date, showing that registration date was not a reliable anchor for imported historical records.

The original V1 SQL is retained in `sql/01_v1_original_retention_view.sql` as development evidence.

## Stage 2 — Research dataset redesign

The analysis was frozen at 8 August 2026.

Two separate prediction problems were created.

### Engagement30

Predicts whether a participant engages within 30 days of CRM registration.

Safeguards:

- actual attended sessions only;
- cancelled/future sessions excluded;
- one participant per row;
- complete 30-day observation window required;
- contradictory imported histories excluded from this experiment.

Final cohort: 1,355 participants.

### Retention90

Uses the first 30 days after first attendance as the observation window and days 31–90 as the outcome window.

Safeguards:

- actual attended sessions only;
- complete 90-day follow-up required;
- future sessions excluded;
- baseline assessment information limited to the observation period;
- implausible physiological values converted to missing values.

Final cohort: 425 participants.

## Stage 3 — Exploratory data analysis

EDA checked:

- class distributions;
- missing data;
- invalid ages;
- duplicate identifiers;
- baseline assessment availability;
- early engagement patterns.

The engagement outcome was strongly imbalanced.

The retention outcome was comparatively balanced.

## Stage 4 — Feature preparation

Deterministic, target-independent feature preparation was separated from statistical preprocessing.

Key decisions:

- impossible ages were converted to missing rather than deleting participants;
- multi-select referral reasons were converted to binary indicators;
- preferred languages were converted to language indicators;
- obvious spelling/case equivalents were normalised;
- extremely sparse baseline variables were excluded from the primary model;
- redundant `EarlyUniqueSessions30` was removed because it was identical to `EarlySessions30`.

Median imputation, scaling, one-hot encoding and rare-category handling were deliberately kept inside the scikit-learn training pipeline to prevent test-set leakage.

## Stage 5 — Baseline models

Four models were compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

An 80/20 stratified split was used with five-fold stratified cross-validation on the training data.

## Stage 6 — Controlled tuning

Hyperparameters were selected using training-only cross-validation.

- Engagement tuning objective: PR-AUC because the positive class was rare.
- Retention tuning objective: ROC-AUC because the outcome classes were reasonably balanced.

The held-out test set was used only for final evaluation after model selection.

No further preprocessing changes or repeated tuning were made after final held-out evaluation.
