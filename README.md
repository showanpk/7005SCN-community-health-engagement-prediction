# 7005SCN Individual Research Project — Machine Learning Artefact

## Project

**Research question:**  
Which participant characteristics and machine learning techniques are most effective in predicting engagement and retention within community health programmes?

This repository contains the technical artefact for a Coventry University MSc Data Science individual research project.

The artefact implements a reproducible analytical pipeline using SQL and Python to:

- create research-specific analytical datasets from a community health CRM;
- perform data-quality validation and exploratory data analysis;
- prepare target-independent model features;
- compare Logistic Regression, Decision Tree, Random Forest and XGBoost;
- evaluate models using Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC;
- perform controlled hyperparameter tuning using training-only cross-validation.

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

requirements.txt
.gitignore
```

## Research design

Two related classification experiments were produced.

### 1. 30-day engagement

The engagement experiment predicts whether a participant has an actual, non-cancelled attendance within 30 days of registration.

Only participants with a complete 30-day observation window are included.

### 2. 90-day retention

The retention experiment uses information available during the participant's first 30 days of attendance and predicts whether the participant has at least one actual attendance during days 31–90.

Only participants with a complete 90-day follow-up window are included.

The analysis snapshot is fixed at **8 August 2026** for reproducibility.

## Data privacy

Participant-level source and model-ready datasets are intentionally **not included** in this repository.

The source data contains sensitive community-health information and remains inside the authorised research environment. The repository contains only code, SQL definitions, aggregate model results and non-participant-level charts.

See `docs/DATA_PRIVACY.md`.

## Reproducing the analysis

1. Create the secure analytical views using the SQL scripts in `sql/`.
2. Export the two research datasets in the authorised environment as:
   - `MSc_Engagement30_V2.csv`
   - `MSc_Retention90_V2.csv`
3. Place those files beside the Python scripts in a secure local working folder.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run:

```bash
python 01_eda_audit.py
python 02_prepare_features_final.py
python 03_train_final_baseline.py
python 04_tune_models.py
```

## Important

The repository is provided for academic assessment/review. No participant-level data, CRM credentials, connection strings or production configuration are included.

The project report explains the methodological decisions, limitations, ethics, data protection and interpretation of the results in full.
