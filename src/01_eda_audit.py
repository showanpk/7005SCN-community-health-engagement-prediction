
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 7005SCN MSc Research Project
# Step 1: Exploratory Data Analysis / Data Quality Audit
#
# This script DOES NOT modify the original CSV files.
# It reads them, checks data quality, and saves summary outputs.
# ============================================================

BASE = Path(__file__).resolve().parent
ENG_FILE = BASE / "MSc_Engagement30_V2.csv"
RET_FILE = BASE / "MSc_Retention90_V2.csv"
OUT = BASE / "eda_outputs"
OUT.mkdir(exist_ok=True)

eng = pd.read_csv(ENG_FILE)
ret = pd.read_csv(RET_FILE)


def missing_table(df):
    result = pd.DataFrame({
        "Column": df.columns,
        "MissingCount": [df[c].isna().sum() for c in df.columns],
    })
    result["MissingPercent"] = (
        result["MissingCount"] / len(df) * 100
    ).round(2)
    return result.sort_values(
        ["MissingPercent", "Column"],
        ascending=[False, True]
    )


def print_dataset_audit(name, df, target):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(
        "Duplicate ResearchParticipantID:",
        int(df["ResearchParticipantID"].duplicated().sum())
    )

    print("\nTarget distribution:")
    counts = df[target].value_counts(dropna=False).sort_index()
    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {label}: {count:,} ({pct:.2f}%)")

    invalid_age = df[
        df["Age"].notna()
        & ((df["Age"] < 0) | (df["Age"] > 110))
    ]

    print(f"\nMissing Age: {df['Age'].isna().sum()}")
    print(f"Invalid Age (<0 or >110): {len(invalid_age)}")

    if len(invalid_age):
        print("Invalid age values:")
        print(
            invalid_age[
                ["ResearchParticipantID", "Age", target]
            ].to_string(index=False)
        )


print_dataset_audit(
    "ENGAGEMENT 30-DAY DATASET",
    eng,
    "Engaged30Label"
)

print_dataset_audit(
    "RETENTION 90-DAY DATASET",
    ret,
    "Retained90Label"
)


# ------------------------------------------------------------
# Missing-value reports
# ------------------------------------------------------------
eng_missing = missing_table(eng)
ret_missing = missing_table(ret)

eng_missing.to_csv(
    OUT / "engagement_missing_values.csv",
    index=False
)

ret_missing.to_csv(
    OUT / "retention_missing_values.csv",
    index=False
)

print("\nTop Engagement missing-value fields:")
print(eng_missing.head(10).to_string(index=False))

print("\nTop Retention missing-value fields:")
print(ret_missing.head(15).to_string(index=False))


# ------------------------------------------------------------
# Important Retention descriptive comparison
# ------------------------------------------------------------
early_cols = [
    "EarlySessions30",
    "EarlyUniqueSessions30",
    "EarlyUniqueActivityTypes30",
    "EarlyActiveWeeks30",
]

retention_early_summary = (
    ret.groupby("Retained90Label")[early_cols]
       .agg(["count", "mean", "median", "std"])
       .round(3)
)

retention_early_summary.to_csv(
    OUT / "retention_early_engagement_summary.csv"
)

print("\nEarly engagement by 90-day retention outcome:")
print(ret.groupby("Retained90Label")[early_cols].mean().round(3))


# ------------------------------------------------------------
# Check redundant early attendance columns
# ------------------------------------------------------------
same_early_sessions = (
    ret["EarlySessions30"].fillna(-999999)
    == ret["EarlyUniqueSessions30"].fillna(-999999)
).all()

print(
    "\nEarlySessions30 identical to EarlyUniqueSessions30:",
    same_early_sessions
)


# ------------------------------------------------------------
# Work on copies only: clean impossible ages for EDA charts
# ------------------------------------------------------------
eng_plot = eng.copy()
ret_plot = ret.copy()

for df in [eng_plot, ret_plot]:
    df.loc[
        (df["Age"] < 0) | (df["Age"] > 110),
        "Age"
    ] = np.nan


# ------------------------------------------------------------
# Charts
# ------------------------------------------------------------
def save_target_chart(df, target, title, filename, labels):
    counts = df[target].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        [labels.get(i, str(i)) for i in counts.index],
        counts.values
    )
    ax.set_title(title)
    ax.set_ylabel("Participants")

    for i, value in enumerate(counts.values):
        ax.text(i, value, str(value), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200)
    plt.close(fig)


save_target_chart(
    eng,
    "Engaged30Label",
    "30-Day Engagement Outcome",
    "engagement_target_distribution.png",
    {0: "Not engaged", 1: "Engaged"}
)

save_target_chart(
    ret,
    "Retained90Label",
    "90-Day Retention Outcome",
    "retention_target_distribution.png",
    {0: "Not retained", 1: "Retained"}
)


# Age distributions
for name, df in [
    ("engagement", eng_plot),
    ("retention", ret_plot),
]:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Age"].dropna(), bins=15)
    ax.set_title(f"Age Distribution - {name.title()} Dataset")
    ax.set_xlabel("Age")
    ax.set_ylabel("Participants")
    fig.tight_layout()
    fig.savefig(
        OUT / f"{name}_age_distribution.png",
        dpi=200
    )
    plt.close(fig)


# Retention by early active weeks
weekly = pd.crosstab(
    ret["EarlyActiveWeeks30"],
    ret["Retained90Label"]
)

weekly_pct = weekly.div(
    weekly.sum(axis=1),
    axis=0
) * 100

weekly.to_csv(
    OUT / "retention_by_early_active_weeks_counts.csv"
)
weekly_pct.round(2).to_csv(
    OUT / "retention_by_early_active_weeks_percent.csv"
)

if 1 in weekly_pct.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        weekly_pct.index.astype(str),
        weekly_pct[1]
    )
    ax.set_title(
        "Retention Rate by Active Weeks in First 30 Days"
    )
    ax.set_xlabel("Early active weeks")
    ax.set_ylabel("Retained at 90 days (%)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(
        OUT / "retention_rate_by_early_active_weeks.png",
        dpi=200
    )
    plt.close(fig)


# ------------------------------------------------------------
# Save a simple audit text file
# ------------------------------------------------------------
summary_lines = [
    "7005SCN MSc Research Dataset - EDA Audit",
    "",
    f"Engagement rows: {len(eng)}",
    f"Engagement duplicate IDs: {eng['ResearchParticipantID'].duplicated().sum()}",
    f"Engaged within 30 days: {int(eng['Engaged30Label'].sum())}",
    f"Not engaged within 30 days: {int((eng['Engaged30Label'] == 0).sum())}",
    "",
    f"Retention rows: {len(ret)}",
    f"Retention duplicate IDs: {ret['ResearchParticipantID'].duplicated().sum()}",
    f"Retained at 90 days: {int(ret['Retained90Label'].sum())}",
    f"Not retained at 90 days: {int((ret['Retained90Label'] == 0).sum())}",
    "",
    f"EarlySessions30 == EarlyUniqueSessions30 for all rows: {same_early_sessions}",
]

(OUT / "eda_audit_summary.txt").write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
print(f"Outputs saved to: {OUT}")
print("Original CSV files were NOT modified.")
