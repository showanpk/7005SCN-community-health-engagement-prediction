
from pathlib import Path
import re
import pandas as pd
import numpy as np

# ============================================================
# 7005SCN MSc Research Project
# Step 2C: FINAL deterministic feature preparation
#
# This is the preprocessing specification to freeze BEFORE
# final tuning. It uses no target information to decide how
# categories are cleaned.
#
# Original research CSVs are NOT modified.
# Statistical preprocessing (median imputation, scaling and
# one-hot encoding) is NOT fitted here; it stays inside the
# training pipeline.
# ============================================================

BASE = Path(__file__).resolve().parent
ENG_FILE = BASE / "MSc_Engagement30_V2.csv"
RET_FILE = BASE / "MSc_Retention90_V2.csv"

OUT = BASE / "model_ready_final"
OUT.mkdir(exist_ok=True)

ENG_OUT = OUT / "MSc_Engagement30_ModelReady.csv"
RET_OUT = OUT / "MSc_Retention90_ModelReady.csv"


def clean_text(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)

    if text in {
        "", "-", "n/a", "n.a", "na", "none", "null",
        "unknown", "not known"
    }:
        return np.nan

    return text


def clean_age(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.mask((s < 0) | (s > 110), np.nan)


# ------------------------------------------------------------
# Occupation: only obvious equivalent spellings/statuses are
# merged. Job titles are otherwise preserved and later rare
# categories are handled inside the TRAINING pipeline.
# ------------------------------------------------------------
def normalise_occupation(value):
    text = clean_text(value)

    if pd.isna(text):
        return np.nan

    homemaker_terms = {
        "housewife", "house wife", "home maker", "homemaker",
        "house maker", "housemaker", "huosemaker", "hosemaker",
        "house ife", "housework", "stay at home mum",
        "stay at home mom"
    }

    unemployed_terms = {
        "unemployed", "un employed", "enemployed",
        "not employed", "not working",
        "not working at the moment/ looking",
        "looking for work", "no", "non"
    }

    carer_terms = {
        "carer", "full time carer", "part time carer",
        "carer for mum", "carer,full time for son"
    }

    if text in homemaker_terms:
        return "homemaker"

    if text in unemployed_terms:
        return "unemployed"

    if "retired" in text or text == "pensioner":
        return "retired"

    if text in {"student", "studying"}:
        return "student"

    if text in carer_terms:
        return "carer"

    return text


def normalise_site(value):
    text = clean_text(value)

    if pd.isna(text):
        return np.nan

    if "alum rock" in text:
        return "alum rock community centre"
    if "calthorpe" in text:
        return "calthorpe wellbeing hub"
    if text == "omnia" or "omnia medical" in text:
        return "omnia medical practice"
    if "parkfield" in text:
        return "parkfield community school"
    if "st paul" in text:
        return "st pauls trust"
    if "handsworth" in text:
        return "handsworth"
    if "ward end" in text:
        return "ward end park"

    return text


def normalise_heard_about(value):
    text = clean_text(value)

    if pd.isna(text):
        return np.nan

    if text in {
        "workwell", "workwell programme",
        "work well", "work well programme"
    }:
        return "workwell"

    if text in {"social media", "facebook"}:
        return "social media"

    if text in {"self-referral", "self referral", "self refferal"}:
        return "self referral"

    if text in {
        "friend", "wife", "through my wife", "daughter",
        "family member attend this site"
    }:
        return "word of mouth"

    return text


# ------------------------------------------------------------
# Health-condition field:
# specific condition descriptions imply "yes".
# "Prefer not to say" remains separate.
# ------------------------------------------------------------
def normalise_health_condition(value):
    text = clean_text(value)

    if pd.isna(text):
        return np.nan

    if text == "no":
        return "no"

    if text == "prefer not to say":
        return "prefer not to say"

    yes_values = {
        "yes",
        "other",
        "progressive/chronic illness",
        "physical disability",
        "mental health condition",
        "neurodivergent (e.g. autism, adhd, dyslexia)",
        "hearing impairment",
        "learning disability",
        "visual impairment",
    }

    if text in yes_values:
        return "yes"

    return text


# ------------------------------------------------------------
# PreferredLanguage is multi-valued in the source.
# Instead of treating "English/Urdu" and "Urdu/English" as
# different categories, derive language indicators.
# ------------------------------------------------------------
LANGUAGE_PATTERNS = {
    "Language_English": [
        r"\benglish\b", r"\benhlish\b", r"\benglih\b"
    ],
    "Language_Urdu": [
        r"\burdu\b"
    ],
    "Language_Arabic": [
        r"\barabic\b", r"\barab\b"
    ],
    "Language_Punjabi": [
        r"\bpunjabi\b", r"\bpanjabi\b", r"\bpunjarbi\b"
    ],
    "Language_Mirpuri": [
        r"\bmirpuri\b"
    ],
    "Language_Bengali": [
        r"\bbengali\b", r"\bbangali\b", r"\bbanglai\b",
        r"\bbangladeshi\b", r"\bbangaldashi\b"
    ],
    "Language_Somali": [
        r"\bsomali\b", r"\bsomalian\b", r"\bsomilian\b"
    ],
    "Language_Pashto": [
        r"\bpashto\b", r"\bpushto\b", r"\bpushtu\b",
        r"\bpashtau\b"
    ],
    "Language_Gujarati": [
        r"\bgujarati\b", r"\bgujurati\b"
    ],
    "Language_Farsi": [
        r"\bfarsi\b", r"\bpersian\b"
    ],
    "Language_Kurdish": [
        r"\bkurdish\b"
    ],
    "Language_OtherKnown": [
        r"\bhindi\b", r"\bchinese\b", r"\bamharic\b",
        r"\bswahili\b", r"\bitalian\b", r"\bgerman\b",
        r"\btigina\b", r"\btigrinya\b", r"\bdenish\b",
        r"\bdanish\b"
    ],
}


def add_language_features(df):
    result = df.copy()

    raw = (
        result["PreferredLanguage"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    matched_any = pd.Series(False, index=result.index)

    for feature, patterns in LANGUAGE_PATTERNS.items():
        pattern = "|".join(patterns)
        result[feature] = raw.str.contains(
            pattern, regex=True, na=False
        ).astype(int)
        matched_any |= result[feature].eq(1)

    result["Language_OtherUnclassified"] = (
        raw.str.strip().ne("") & ~matched_any
    ).astype(int)

    result["PreferredLanguageMissing"] = (
        raw.str.strip().eq("")
    ).astype(int)

    return result.drop(columns=["PreferredLanguage"])


# ------------------------------------------------------------
# ReferralReason is also multi-select/free text.
# ------------------------------------------------------------
REFERRAL_PATTERNS = {
    "Referral_ExerciseMobility": [
        r"exercise", r"mobility", r"physical activ", r"fitness"
    ],
    "Referral_WeightManagement": [
        r"weight", r"obes"
    ],
    "Referral_HealthyEating": [
        r"healthy eating", r"nutrition", r"diet"
    ],
    "Referral_LongTermCondition": [
        r"long term health", r"long-term health",
        r"health condition", r"disabil", r"diabet",
        r"pre[- ]?diabet"
    ],
    "Referral_MentalHealth": [
        r"depress", r"anxiety", r"mental health",
        r"wellbeing", r"well-being"
    ],
    "Referral_Isolation": [
        r"isolation", r"loneliness", r"lonely", r"social isol"
    ],
    "Referral_EmploymentTraining": [
        r"employment", r"training", r"learning", r"education"
    ],
    "Referral_SocialCircumstances": [
        r"social circumstances", r"challenging social",
        r"housing", r"financial", r"benefit"
    ],
}


def add_referral_features(df):
    result = df.copy()

    raw = (
        result["ReferralReason"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    matched_any = pd.Series(False, index=result.index)

    for feature, patterns in REFERRAL_PATTERNS.items():
        pattern = "|".join(patterns)
        result[feature] = raw.str.contains(
            pattern, regex=True, na=False
        ).astype(int)
        matched_any |= result[feature].eq(1)

    result["Referral_Other"] = (
        raw.str.strip().ne("") & ~matched_any
    ).astype(int)

    result["ReferralReasonMissing"] = (
        raw.str.strip().eq("")
    ).astype(int)

    return result.drop(columns=["ReferralReason"])


def normalise_categories(df):
    result = df.copy()

    for col in ["Gender", "Ethnicity"]:
        if col in result.columns:
            result[col] = result[col].map(clean_text)

    if "Occupation" in result.columns:
        result["Occupation"] = result["Occupation"].map(
            normalise_occupation
        )

    if "Site" in result.columns:
        result["Site"] = result["Site"].map(normalise_site)

    if "HeardAboutSaheli" in result.columns:
        result["HeardAboutSaheli"] = result[
            "HeardAboutSaheli"
        ].map(normalise_heard_about)

    if "HasHealthConditionOrDisability" in result.columns:
        result["HasHealthConditionOrDisability"] = result[
            "HasHealthConditionOrDisability"
        ].map(normalise_health_condition)

    return result


def prepare_engagement(df):
    result = df.copy()
    result["Age"] = clean_age(result["Age"])

    result = result.drop(
        columns=[
            "ResearchParticipantID",
            "RegistrationDate",
            "AgeBand",
        ],
        errors="ignore",
    )

    result = add_language_features(result)
    result = add_referral_features(result)
    result = normalise_categories(result)

    for col in ["LivingAlone", "CaringResponsibilities"]:
        if col in result.columns:
            result[col] = pd.to_numeric(
                result[col], errors="coerce"
            )

    target = result.pop("Engaged30Label").astype(int)
    result["Engaged30Label"] = target
    return result


def prepare_retention(df):
    result = df.copy()
    result["Age"] = clean_age(result["Age"])

    result = result.drop(
        columns=[
            "ResearchParticipantID",
            "FirstAttendanceDate",
            "BaselineAssessmentDate",
            "Retention90Status",
            "AgeBand",
            "EarlyUniqueSessions30",
            "BaselineHbA1c",
            "BaselineHeartAge",
            "BaselineActiveDaysPerWeek",
        ],
        errors="ignore",
    )

    result = add_language_features(result)
    result = add_referral_features(result)
    result = normalise_categories(result)

    for col in ["LivingAlone", "CaringResponsibilities"]:
        if col in result.columns:
            result[col] = pd.to_numeric(
                result[col], errors="coerce"
            )

    target = result.pop("Retained90Label").astype(int)
    result["Retained90Label"] = target
    return result


eng_raw = pd.read_csv(ENG_FILE)
ret_raw = pd.read_csv(RET_FILE)

eng = prepare_engagement(eng_raw)
ret = prepare_retention(ret_raw)

eng.to_csv(ENG_OUT, index=False)
ret.to_csv(RET_OUT, index=False)


# ------------------------------------------------------------
# Verification / audit
# ------------------------------------------------------------
checks = []

for dataset_name, df, target in [
    ("Engagement", eng, "Engaged30Label"),
    ("Retention", ret, "Retained90Label"),
]:
    checks.append({
        "Dataset": dataset_name,
        "Rows": len(df),
        "Predictors": len(df.columns) - 1,
        "Class0": int((df[target] == 0).sum()),
        "Class1": int((df[target] == 1).sum()),
        "InvalidAgesRemaining": int(
            (
                df["Age"].notna()
                & ((df["Age"] < 0) | (df["Age"] > 110))
            ).sum()
        ),
    })

audit = pd.DataFrame(checks)
audit.to_csv(OUT / "final_feature_audit.csv", index=False)

print("=" * 72)
print("STEP 2C - FINAL FEATURE PREPARATION COMPLETE")
print("=" * 72)
print(audit.to_string(index=False))

print("\nFinal categorical decisions:")
print("  PreferredLanguage -> multi-language binary indicators")
print("  ReferralReason -> multi-reason binary indicators")
print("  Obvious occupation spelling/status equivalents normalised")
print("  Obvious HeardAboutSaheli equivalents normalised")
print("  Specific disability/condition descriptions -> health condition = yes")

print("\nIMPORTANT:")
print("  These rules are target-independent and are now FROZEN.")
print("  Do not change feature cleaning because of model scores.")
print("  Original V2 research CSVs were NOT modified.")
print("  Imputation/scaling/one-hot encoding will be fitted on TRAINING data only.")

print(f"\nSaved to:\n  {OUT}")
