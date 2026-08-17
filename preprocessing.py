"""
Shared preprocessing logic for the HR Analytics job-change model.
Used by both train_model.py (fitting) and app.py (inference), so the
exact same transformations are applied at training and prediction time.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def convert_experience(val):
    if pd.isna(val):
        return np.nan
    if val == "<1":
        return 0
    if val == ">20":
        return 21
    return int(val)


def convert_last_new_job(val):
    if pd.isna(val):
        return np.nan
    if val == "never":
        return 0
    if val == ">4":
        return 5
    return int(val)


def engineer_features(raw_df, numeric_medians=None, fit=False):
    """
    Applies all pre-ColumnTransformer steps to a raw dataframe:
    missingness flags, missing-value fill, type conversions, and
    engineered columns (experience_to_last_job_ratio, cdi_bin).

    raw_df columns expected: city, city_development_index, gender,
    relevent_experience, enrolled_university, education_level,
    major_discipline, experience, company_size, company_type,
    last_new_job, training_hours
    (experience / last_new_job may be given as raw strings or already numeric)

    If fit=True, computes and returns numeric_medians from this data
    (call this on the training set). If fit=False, numeric_medians must
    be supplied (the medians learned during training).
    """
    df = raw_df.copy()

    # missingness flags (captured before filling)
    for col in ["company_size", "company_type"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # categorical missing -> "Unknown"
    categorical_missing_cols = [
        "gender", "enrolled_university", "education_level", "major_discipline",
        "company_size", "company_type",
    ]
    for col in categorical_missing_cols:
        df[col] = df[col].fillna("Unknown")

    # convert experience / last_new_job to numeric if they're still raw strings
    if not pd.api.types.is_numeric_dtype(df["experience"]):
        df["experience"] = df["experience"].apply(convert_experience)
    if not pd.api.types.is_numeric_dtype(df["last_new_job"]):
        df["last_new_job"] = df["last_new_job"].apply(convert_last_new_job)

    numeric_cols = ["city_development_index", "training_hours", "experience", "last_new_job"]
    if fit:
        numeric_medians = df[numeric_cols].median()
    df[numeric_cols] = df[numeric_cols].fillna(numeric_medians)

    # feature engineering
    df["experience_to_last_job_ratio"] = df["experience"] / (df["last_new_job"] + 1)

    cdi_bins = [0, 0.6, 0.8, 1.01]
    cdi_labels = ["low", "mid", "high"]
    df["cdi_bin"] = pd.cut(df["city_development_index"], bins=cdi_bins, labels=cdi_labels).astype(str)

    if fit:
        return df, numeric_medians
    return df


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Maps a categorical column to its relative frequency, learned on fit data only
    (leakage-safe: fits inside the train fold during CV / hyperparameter search)."""
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        counts = X[self.column].value_counts(normalize=True)
        self.freq_map_ = counts.to_dict()
        self.default_ = 0.0  # unseen categories (e.g. new cities) get 0
        return self

    def transform(self, X):
        mapped = X[self.column].map(self.freq_map_).fillna(self.default_)
        return mapped.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.column}_freq"])


ONEHOT_COLS = [
    "enrolled_university", "education_level", "major_discipline",
    "company_size", "company_type", "relevent_experience", "gender", "cdi_bin",
]
PASSTHROUGH_COLS = [
    "city_development_index", "training_hours", "experience", "last_new_job",
    "experience_to_last_job_ratio", "company_size_missing", "company_type_missing",
]
