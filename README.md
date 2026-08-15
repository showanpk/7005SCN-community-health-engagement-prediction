# 7005SCN Individual Research Project

## Community Health Engagement and Retention Prediction

This repository contains the coding and results from my MSc Data Science
Individual Research Project at Coventry University.

My research question is:

**Which participant characteristics and machine learning techniques are most
effective in predicting engagement and retention within community health
programmes?**

For this project, I used data from a community health programme and explored
whether machine learning could help identify patterns linked with participant
engagement and retention.

The main tools I used were SQL Server and Python.

## What I did

I first used SQL to prepare the research data from the original CRM data.

During the initial checks I found several data quality problems, including
duplicate records, future attendance dates, unrealistic health values and
features that could cause target leakage.

Because of these issues, I redesigned the research datasets before starting
the final machine learning analysis.

I created two prediction tasks:

### 1. Engagement within 30 days

The first model looks at whether a participant attended a session within
30 days of registration.

Only participants who had enough time to complete the full 30-day period were
included.

### 2. Retention within 90 days

The second model looks at participants who had already attended.

I used information from their first 30 days of participation to predict
whether they would attend again between day 31 and day 90.

Only participants who had a complete 90-day follow-up period were included.

For consistency, I used **8 August 2026** as the fixed analysis date.

## Machine learning models

I compared four classification models:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

I evaluated the models using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

I used an 80/20 training and test split.

Five-fold stratified cross-validation was used on the training data, and I
also carried out controlled hyperparameter tuning.

The held-out test data was kept separate from the tuning process and was used
for the final model evaluation.

## Repository structure

```text
src/
    01_eda_audit.py
    02_prepare_features_final.py
    03_train_final_baseline.py
    04_tune_models.py

sql/
    01_v1_original_retention_view.sql
    02_engagement30_v2.sql
    03_retention90_v2.sql

results/
    final_feature_audit.csv
    baseline/
    tuned/

docs/
    DEVELOPMENT_LOG.md
    DATA_PRIVACY.md
    AI_DISCLOSURE_TEMPLATE.md
    SUBMISSION_CHECKLIST.md