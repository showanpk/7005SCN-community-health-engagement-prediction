
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 7005SCN MSc Research Project
# Step 4: Controlled Hyperparameter Tuning
#
# Research safeguards:
# - Same fixed 80/20 stratified split as final baseline.
# - Hyperparameters selected ONLY using 5-fold CV on TRAIN data.
# - Held-out TEST data is never used by RandomizedSearchCV.
# - Engagement primary tuning metric = PR-AUC (average precision)
#   because the positive class is rare (~12%).
# - Retention primary tuning metric = ROC-AUC because the classes
#   are reasonably balanced and discrimination is of interest.
# - Final test metrics include all metrics promised in proposal.
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ITER = 20

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "model_ready_final"
BASELINE_DIR = BASE / "model_results_final_baseline"
OUT = BASE / "model_results_tuned"
OUT.mkdir(exist_ok=True)

ENG_FILE = DATA_DIR / "MSc_Engagement30_ModelReady.csv"
RET_FILE = DATA_DIR / "MSc_Retention90_ModelReady.csv"


def build_preprocessor(X, min_category_frequency):
    categorical_cols = (
        X.select_dtypes(
            include=["object", "string", "category"]
        )
        .columns
        .tolist()
    )

    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="missing",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_category_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def base_models_and_spaces(y_train, imbalanced):
    if imbalanced:
        n0 = int((y_train == 0).sum())
        n1 = int((y_train == 1).sum())
        ratio = n0 / n1

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "Decision Tree": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "Random Forest": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "XGBoost": XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=1,
            ),
        }

        spaces = {
            "Logistic Regression": {
                "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                "model__solver": ["liblinear", "lbfgs"],
            },
            "Decision Tree": {
                "model__max_depth": [2, 3, 4, 5, 6, 8, 10, None],
                "model__min_samples_split": [2, 5, 10, 20, 30],
                "model__min_samples_leaf": [1, 2, 5, 10, 20],
                "model__criterion": ["gini", "entropy"],
            },
            "Random Forest": {
                "model__n_estimators": [200, 300, 500, 700],
                "model__max_depth": [4, 6, 8, 10, 14, None],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", 0.5],
                "model__class_weight": ["balanced", "balanced_subsample"],
            },
            "XGBoost": {
                "model__n_estimators": [100, 200, 300, 500],
                "model__max_depth": [2, 3, 4, 5],
                "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
                "model__min_child_weight": [1, 3, 5, 8],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__gamma": [0, 0.1, 0.3],
                "model__scale_pos_weight": [
                    ratio * 0.75,
                    ratio,
                    ratio * 1.25,
                ],
            },
        }

        return models, spaces

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=4000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=1,
        ),
    }

    spaces = {
        "Logistic Regression": {
            "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
            "model__solver": ["liblinear", "lbfgs"],
            "model__class_weight": [None, "balanced"],
        },
        "Decision Tree": {
            "model__max_depth": [2, 3, 4, 5, 6, 8, 10, None],
            "model__min_samples_split": [2, 5, 10, 20, 30],
            "model__min_samples_leaf": [1, 2, 5, 10, 20],
            "model__criterion": ["gini", "entropy"],
            "model__class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "model__n_estimators": [200, 300, 500, 700],
            "model__max_depth": [4, 6, 8, 10, 14, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
        },
        "XGBoost": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [2, 3, 4, 5],
            "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
            "model__min_child_weight": [1, 3, 5, 8],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__gamma": [0, 0.1, 0.3],
        },
    }

    return models, spaces


def save_feature_importance(best_pipeline, model_name, dataset_name):
    prep = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    try:
        feature_names = prep.get_feature_names_out()
    except Exception:
        return

    feature_names = [
        str(x).replace("num__", "").replace("cat__", "")
        for x in feature_names
    ]

    if hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
        metric_name = "AbsoluteCoefficient"
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        metric_name = "FeatureImportance"
    else:
        return

    imp = pd.DataFrame(
        {
            "Feature": feature_names,
            metric_name: values,
        }
    ).sort_values(metric_name, ascending=False)

    safe_model = model_name.lower().replace(" ", "_")

    imp.to_csv(
        OUT / f"{dataset_name}_{safe_model}_tuned_feature_importance.csv",
        index=False,
    )

    top = imp.head(20).sort_values(metric_name)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["Feature"], top[metric_name])
    ax.set_title(
        f"{dataset_name.title()} - Tuned {model_name}\nTop 20 Features"
    )
    ax.set_xlabel(metric_name)
    fig.tight_layout()
    fig.savefig(
        OUT / f"{dataset_name}_{safe_model}_tuned_top_features.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def load_baseline(dataset_name):
    path = BASELINE_DIR / f"{dataset_name}_heldout_test_results.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df["Version"] = "Baseline"
    return df


def run_tuning(
    df,
    target,
    dataset_name,
    min_frequency,
    imbalanced,
    primary_metric,
):
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models, spaces = base_models_and_spaces(
        y_train=y_train,
        imbalanced=imbalanced,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC_AUC": "roc_auc",
        "PR_AUC": "average_precision",
    }

    tuning_rows = []
    test_rows = []
    roc_data = {}
    pr_data = {}

    for model_name, model in models.items():
        print(
            f"\n[{dataset_name.upper()}] Tuning {model_name} "
            f"using {primary_metric}..."
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        X_train,
                        min_frequency,
                    ),
                ),
                ("model", model),
            ]
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=spaces[model_name],
            n_iter=N_ITER,
            scoring=scoring,
            refit=primary_metric,
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            return_train_score=False,
            verbose=0,
        )

        search.fit(X_train, y_train)

        best_index = search.best_index_
        cv_results = search.cv_results_

        tuning_row = {
            "Model": model_name,
            "PrimaryMetric": primary_metric,
            "BestCVPrimaryScore": float(search.best_score_),
            "BestParameters": json.dumps(
                search.best_params_,
                sort_keys=True,
                default=str,
            ),
        }

        for metric in scoring:
            tuning_row[f"BestCV_{metric}"] = float(
                cv_results[f"mean_test_{metric}"][best_index]
            )
            tuning_row[f"BestCV_{metric}_Std"] = float(
                cv_results[f"std_test_{metric}"][best_index]
            )

        tuning_rows.append(tuning_row)

        best_pipe = search.best_estimator_

        pred = best_pipe.predict(X_test)
        prob = best_pipe.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(
            y_test,
            pred,
            labels=[0, 1],
        )

        test_rows.append(
            {
                "Model": model_name,
                "Version": "Tuned",
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(
                    y_test,
                    pred,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_test,
                    pred,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_test,
                    pred,
                    zero_division=0,
                ),
                "ROC_AUC": roc_auc_score(y_test, prob),
                "PR_AUC": average_precision_score(
                    y_test,
                    prob,
                ),
                "TrueNegative": int(cm[0, 0]),
                "FalsePositive": int(cm[0, 1]),
                "FalseNegative": int(cm[1, 0]),
                "TruePositive": int(cm[1, 1]),
            }
        )

        safe_model = model_name.lower().replace(" ", "_")

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Class 0", "Class 1"],
        ).plot(
            ax=ax,
            values_format="d",
            colorbar=False,
        )
        ax.set_title(
            f"{dataset_name.title()} - Tuned {model_name}"
        )
        fig.tight_layout()
        fig.savefig(
            OUT / f"{dataset_name}_{safe_model}_tuned_confusion_matrix.png",
            dpi=200,
        )
        plt.close(fig)

        fpr, tpr, _ = roc_curve(y_test, prob)
        precision, recall, _ = precision_recall_curve(
            y_test,
            prob,
        )

        roc_data[model_name] = (fpr, tpr)
        pr_data[model_name] = (recall, precision)

        save_feature_importance(
            best_pipe,
            model_name,
            dataset_name,
        )

    tuning_df = pd.DataFrame(tuning_rows)
    test_df = pd.DataFrame(test_rows)

    tuning_df.to_csv(
        OUT / f"{dataset_name}_tuning_cv_results.csv",
        index=False,
    )

    test_df.to_csv(
        OUT / f"{dataset_name}_tuned_heldout_test_results.csv",
        index=False,
    )

    baseline = load_baseline(dataset_name)

    if baseline is not None:
        comparison = pd.concat(
            [
                baseline,
                test_df,
            ],
            ignore_index=True,
            sort=False,
        )

        comparison.to_csv(
            OUT / f"{dataset_name}_baseline_vs_tuned.csv",
            index=False,
        )

    # ROC curves for tuned models
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, (fpr, tpr) in roc_data.items():
        auc_val = test_df.loc[
            test_df["Model"] == model_name,
            "ROC_AUC",
        ].iloc[0]

        ax.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC={auc_val:.3f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random",
    )
    ax.set_title(
        f"{dataset_name.title()} - Tuned ROC Curves"
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUT / f"{dataset_name}_tuned_roc_curves.png",
        dpi=200,
    )
    plt.close(fig)

    # PR curves for tuned models
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, (recall, precision) in pr_data.items():
        ap_val = test_df.loc[
            test_df["Model"] == model_name,
            "PR_AUC",
        ].iloc[0]

        ax.plot(
            recall,
            precision,
            label=f"{model_name} (AP={ap_val:.3f})",
        )

    ax.set_title(
        f"{dataset_name.title()} - Tuned Precision-Recall Curves"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUT / f"{dataset_name}_tuned_precision_recall_curves.png",
        dpi=200,
    )
    plt.close(fig)

    print("\nBest cross-validation results:")
    print(
        tuning_df[
            [
                "Model",
                "BestCVPrimaryScore",
                "BestCV_F1",
                "BestCV_Recall",
                "BestCV_ROC_AUC",
                "BestCV_PR_AUC",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nFinal held-out test results:")
    print(
        test_df[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC_AUC",
                "PR_AUC",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    return tuning_df, test_df


eng = pd.read_csv(ENG_FILE)
ret = pd.read_csv(RET_FILE)

print("=" * 76)
print("STEP 4 - CONTROLLED HYPERPARAMETER TUNING")
print("=" * 76)
print(
    "Hyperparameters are selected from training data only "
    "using 5-fold stratified CV."
)

eng_tuning, eng_test = run_tuning(
    df=eng,
    target="Engaged30Label",
    dataset_name="engagement",
    min_frequency=10,
    imbalanced=True,
    primary_metric="PR_AUC",
)

ret_tuning, ret_test = run_tuning(
    df=ret,
    target="Retained90Label",
    dataset_name="retention",
    min_frequency=5,
    imbalanced=False,
    primary_metric="ROC_AUC",
)

print("\n" + "=" * 76)
print("TUNING COMPLETE")
print("=" * 76)
print(
    "Do not change preprocessing rules or tune again based on "
    "the held-out test scores."
)
print(
    "The next step is final model interpretation and dissertation "
    "results/evaluation."
)
print(f"\nOutputs saved to:\n{OUT}")
