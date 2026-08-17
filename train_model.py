"""
Trains the job-change prediction model and saves everything the Streamlit
app needs: the fitted pipeline (model.pkl) and a small metadata file
(model_meta.pkl) with the list of known cities and the chosen threshold.

Run this once, locally, with aug_train.csv in the same folder:
    pip install scikit-learn xgboost lightgbm catboost joblib pandas numpy
    python train_model.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve
)

from preprocessing import (
    engineer_features, FrequencyEncoder, ONEHOT_COLS, PASSTHROUGH_COLS
)

HAS_XGB = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    pass

HAS_LGBM = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    pass

HAS_CATBOOST = False
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    pass

RANDOM_STATE = 42
N_ITER = 20
CV_FOLDS = 3

# ---------------------------------------------------------------------------
# Load + preprocess
# ---------------------------------------------------------------------------
train_raw = pd.read_csv("aug_train.csv")
test_ids = train_raw["enrollee_id"]
raw = train_raw.drop(columns=["enrollee_id"])

y = raw["target"].astype(int)
raw_features = raw.drop(columns=["target"])

df, numeric_medians = engineer_features(raw_features, fit=True)
X = df

preprocessor = ColumnTransformer(
    transformers=[
        ("city_freq", FrequencyEncoder(column="city"), ["city"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ONEHOT_COLS),
        ("num", SimpleImputer(strategy="median"), PASSTHROUGH_COLS),
    ]
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train size: {X_train.shape[0]} | Val size: {X_val.shape[0]}")

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

model_configs = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "param_distributions": {
            "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "param_distributions": {
            "model__n_estimators": [200, 300, 500, 800],
            "model__max_depth": [None, 6, 10, 14, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__max_features": ["sqrt", "log2", 0.5],
        },
    },
}

if HAS_XGB:
    model_configs["XGBoost"] = {
        "estimator": XGBClassifier(
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "param_distributions": {
            "model__n_estimators": [200, 300, 500, 800],
            "model__max_depth": [3, 4, 6, 8],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__min_child_weight": [1, 3, 5],
        },
    }

if HAS_LGBM:
    model_configs["LightGBM"] = {
        "estimator": LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        "param_distributions": {
            "model__n_estimators": [200, 300, 500, 800],
            "model__num_leaves": [15, 31, 63, 127],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
        },
    }

if HAS_CATBOOST:
    model_configs["CatBoost"] = {
        "estimator": CatBoostClassifier(auto_class_weights="Balanced", random_state=RANDOM_STATE, verbose=False),
        "param_distributions": {
            "model__iterations": [200, 300, 500, 800],
            "model__depth": [4, 6, 8, 10],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "model__l2_leaf_reg": [1, 3, 5, 7, 9],
        },
    }

# ---------------------------------------------------------------------------
# Train + tune + pick best model
# ---------------------------------------------------------------------------
results = []
best_params_log = {}
chosen_thresholds = {}

for name, cfg in model_configs.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", cfg["estimator"])])
    search = RandomizedSearchCV(
        estimator=pipe, param_distributions=cfg["param_distributions"],
        n_iter=N_ITER, scoring="average_precision", cv=CV_FOLDS,
        random_state=RANDOM_STATE, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    best_params_log[name] = search.best_params_

    y_proba = search.best_estimator_.predict_proba(X_val)[:, 1]
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_val, y_proba)
    f1_curve = np.where(
        (prec_curve + rec_curve) > 0,
        2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-12), 0,
    )
    best_idx = np.argmax(f1_curve[:-1])
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    chosen_thresholds[name] = best_threshold

    y_pred = (y_proba >= best_threshold).astype(int)
    pr_auc = average_precision_score(y_val, y_proba)
    results.append({
        "Model": name, "PR-AUC": pr_auc,
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1": f1_score(y_val, y_pred),
    })
    print(f"{name}: PR-AUC={pr_auc:.4f}, best_threshold={best_threshold:.3f}")

results_df = pd.DataFrame(results).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_threshold = chosen_thresholds[best_model_name]
print(f"\nBest model: {best_model_name} (threshold={best_threshold:.3f})")

# ---------------------------------------------------------------------------
# Retrain best model on full data, save pipeline + metadata
# ---------------------------------------------------------------------------
best_estimator = model_configs[best_model_name]["estimator"]
best_params_final = {k.replace("model__", ""): v for k, v in best_params_log[best_model_name].items()}
best_estimator.set_params(**best_params_final)

final_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", best_estimator)])
final_pipe.fit(X, y)

joblib.dump(final_pipe, "model.pkl")

meta = {
    "model_name": best_model_name,
    "threshold": best_threshold,
    "numeric_medians": numeric_medians.to_dict(),
    "known_cities": sorted(raw_features["city"].unique().tolist()),
}
joblib.dump(meta, "model_meta.pkl")

print("\nSaved: model.pkl, model_meta.pkl")
