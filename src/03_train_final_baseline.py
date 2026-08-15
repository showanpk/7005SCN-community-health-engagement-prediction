
from pathlib import Path
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
    cross_validate,
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
# Step 3C: FINAL baseline comparison after preprocessing freeze
#
# This is the baseline to retain for dissertation comparison.
# No hyperparameter search is performed here.
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "model_ready_final"
OUT = BASE / "model_results_final_baseline"
OUT.mkdir(exist_ok=True)

ENG_FILE = DATA_DIR / "MSc_Engagement30_ModelReady.csv"
RET_FILE = DATA_DIR / "MSc_Retention90_ModelReady.csv"


def build_preprocessor(X, min_category_frequency):
    categorical_cols = (
        X.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
    )

    numeric_cols = [
        c for c in X.columns if c not in categorical_cols
    ]

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


def build_models(y_train, imbalanced):
    if imbalanced:
        negative = int((y_train == 0).sum())
        positive = int((y_train == 1).sum())
        scale_pos_weight = negative / positive

        return {
            "Logistic Regression": LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "Decision Tree": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.80,
                colsample_bytree=0.80,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            ),
        }

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.80,
            colsample_bytree=0.80,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def save_feature_importance(pipe, model_name, dataset):
    prep = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]

    try:
        names = prep.get_feature_names_out()
    except Exception:
        return

    names = [
        str(x).replace("num__", "").replace("cat__", "")
        for x in names
    ]

    if hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
        metric = "AbsoluteCoefficient"
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        metric = "FeatureImportance"
    else:
        return

    importance = pd.DataFrame(
        {"Feature": names, metric: values}
    ).sort_values(metric, ascending=False)

    safe = model_name.lower().replace(" ", "_")

    importance.to_csv(
        OUT / f"{dataset}_{safe}_feature_importance.csv",
        index=False,
    )

    top = importance.head(20).sort_values(metric)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["Feature"], top[metric])
    ax.set_title(
        f"{dataset.title()} - {model_name}\nTop 20 Features"
    )
    ax.set_xlabel(metric)
    fig.tight_layout()
    fig.savefig(
        OUT / f"{dataset}_{safe}_top_features.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_experiment(
    df,
    target,
    dataset,
    min_frequency,
    imbalanced,
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

    preprocessor, numeric_cols, categorical_cols = (
        build_preprocessor(X_train, min_frequency)
    )

    models = build_models(y_train, imbalanced)

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

    cv_rows = []
    test_rows = []
    roc_data = {}
    pr_data = {}

    for model_name, model in models.items():
        print(f"\n[{dataset.upper()}] Training {model_name}...")

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )

        cv_row = {"Model": model_name}

        for metric in scoring:
            vals = scores[f"test_{metric}"]
            cv_row[f"CV_{metric}_Mean"] = float(
                np.mean(vals)
            )
            cv_row[f"CV_{metric}_Std"] = float(
                np.std(vals)
            )

        cv_rows.append(cv_row)

        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(
            y_test,
            pred,
            labels=[0, 1],
        )

        test_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(
                    y_test, pred, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, pred, zero_division=0
                ),
                "F1": f1_score(
                    y_test, pred, zero_division=0
                ),
                "ROC_AUC": roc_auc_score(y_test, prob),
                "PR_AUC": average_precision_score(
                    y_test, prob
                ),
                "TrueNegative": int(cm[0, 0]),
                "FalsePositive": int(cm[0, 1]),
                "FalseNegative": int(cm[1, 0]),
                "TruePositive": int(cm[1, 1]),
            }
        )

        safe = model_name.lower().replace(" ", "_")

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Class 0", "Class 1"],
        ).plot(
            ax=ax,
            values_format="d",
            colorbar=False,
        )

        ax.set_title(f"{dataset.title()} - {model_name}")
        fig.tight_layout()
        fig.savefig(
            OUT / f"{dataset}_{safe}_confusion_matrix.png",
            dpi=200,
        )
        plt.close(fig)

        fpr, tpr, _ = roc_curve(y_test, prob)
        precision, recall, _ = precision_recall_curve(
            y_test, prob
        )

        roc_data[model_name] = (fpr, tpr)
        pr_data[model_name] = (recall, precision)

        save_feature_importance(
            pipe,
            model_name,
            dataset,
        )

    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows)

    cv_df.to_csv(
        OUT / f"{dataset}_cross_validation_results.csv",
        index=False,
    )
    test_df.to_csv(
        OUT / f"{dataset}_heldout_test_results.csv",
        index=False,
    )

    split = pd.DataFrame(
        [
            {
                "Dataset": dataset,
                "Split": "Full",
                "Rows": len(y),
                "Class0": int((y == 0).sum()),
                "Class1": int((y == 1).sum()),
            },
            {
                "Dataset": dataset,
                "Split": "Train",
                "Rows": len(y_train),
                "Class0": int((y_train == 0).sum()),
                "Class1": int((y_train == 1).sum()),
            },
            {
                "Dataset": dataset,
                "Split": "Test",
                "Rows": len(y_test),
                "Class0": int((y_test == 0).sum()),
                "Class1": int((y_test == 1).sum()),
            },
        ]
    )

    # ROC
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
    ax.set_title(f"{dataset.title()} - ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUT / f"{dataset}_roc_curves.png",
        dpi=200,
    )
    plt.close(fig)

    # PR
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
        f"{dataset.title()} - Precision-Recall Curves"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUT / f"{dataset}_precision_recall_curves.png",
        dpi=200,
    )
    plt.close(fig)

    print("\n5-fold CV means:")
    print(
        cv_df[
            [
                "Model",
                "CV_F1_Mean",
                "CV_Recall_Mean",
                "CV_ROC_AUC_Mean",
                "CV_PR_AUC_Mean",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nHeld-out test results:")
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
        ].round(3).to_string(index=False)
    )

    return split, cv_df, test_df, numeric_cols, categorical_cols


eng = pd.read_csv(ENG_FILE)
ret = pd.read_csv(RET_FILE)

print("=" * 72)
print("STEP 3C - FINAL BASELINE MACHINE LEARNING")
print("=" * 72)

eng_split, eng_cv, eng_test, eng_num, eng_cat = run_experiment(
    eng,
    target="Engaged30Label",
    dataset="engagement",
    min_frequency=10,
    imbalanced=True,
)

ret_split, ret_cv, ret_test, ret_num, ret_cat = run_experiment(
    ret,
    target="Retained90Label",
    dataset="retention",
    min_frequency=5,
    imbalanced=False,
)

pd.concat(
    [eng_split, ret_split],
    ignore_index=True,
).to_csv(
    OUT / "train_test_split_summary.csv",
    index=False,
)

pd.DataFrame(
    [
        {
            "Dataset": "Engagement",
            "NumericPredictors": len(eng_num),
            "CategoricalPredictors": len(eng_cat),
        },
        {
            "Dataset": "Retention",
            "NumericPredictors": len(ret_num),
            "CategoricalPredictors": len(ret_cat),
        },
    ]
).to_csv(
    OUT / "feature_type_summary.csv",
    index=False,
)

print("\n" + "=" * 72)
print("FINAL BASELINE COMPLETE")
print("=" * 72)
print(
    "Preprocessing rules are now frozen. "
    "The next stage is controlled hyperparameter tuning."
)
print(f"\nOutputs saved to:\n{OUT}")
