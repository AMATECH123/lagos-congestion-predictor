"""
LAGOS TRAFFIC CONGESTION PREDICTOR — FULL ML PIPELINE
=======================================================
Target: congestion_level (Low / Medium / High) — multiclass classification
Run this in Colab or locally. Cells are separated by "# %%" so it pastes
straight into Jupyter/Colab cells if you want.
"""

# %% [1] IMPORTS -------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, ConfusionMatrixDisplay)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib

RANDOM_STATE = 42

# %% [2] LOAD -----------------------------------------------------------------
df = pd.read_csv("lagos_traffic_congestion_complex.csv")
print(df.shape)
df.head()

# %% [3] CLEAN ----------------------------------------------------------------
# One missing travel_time_min -> drop it (it's also a leakage column we'll
# remove anyway, but clean the row so nothing chokes downstream if you keep it
# for EDA).
df = df.dropna(subset=["travel_time_min"]).reset_index(drop=True)

# Parse date -> pull out month/day-of-month; day_of_week already given as text
df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day

# %% [4] LEAKAGE CHECK ----------------------------------------------------------
# congestion_score, avg_speed_kmh, and travel_time_min are the columns most
# likely used to construct congestion_level itself (or are direct proxies for
# it). Verify, then drop them from the feature set.
print(df.groupby("congestion_level")[["congestion_score", "avg_speed_kmh", "travel_time_min"]].describe())

LEAKY_COLS = ["congestion_score", "avg_speed_kmh", "travel_time_min"]
TARGET = "congestion_level"
DROP_COLS = ["date"] + LEAKY_COLS  # keep month/day_of_month, drop raw date

# %% [5] EDA (quick) ------------------------------------------------------------
plt.figure(figsize=(5, 4))
df[TARGET].value_counts().plot(kind="bar", color=["#2e7d32", "#f9a825", "#c62828"])
plt.title("Class balance: congestion_level")
plt.tight_layout()
plt.savefig("class_balance.png", dpi=120)
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x=TARGET, y="vehicle_count_est",
            order=["Low", "Medium", "High"])
plt.title("Vehicle count by congestion level")
plt.tight_layout()
plt.savefig("vehicle_count_by_level.png", dpi=120)
plt.close()

corr_cols = ["hour", "lanes", "length_km", "rain_intensity", "visibility_km",
             "vehicle_count_est", "month", "day_of_month"]
plt.figure(figsize=(8, 6))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature correlation (non-leaky numeric features)")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=120)
plt.close()

# %% [6] FEATURE / TARGET SPLIT ------------------------------------------------
X = df.drop(columns=DROP_COLS + [TARGET])
y = df[TARGET]

CATEGORICAL = ["route", "day_of_week"]
BINARY_FLAGS = ["is_weekend", "is_public_holiday", "is_school_day", "has_toll",
                "has_accident", "has_roadwork", "has_police_checkpoint",
                "has_event_nearby", "fuel_scarcity", "is_market_day_route"]
NUMERIC = ["hour", "lanes", "length_km", "rain_intensity", "visibility_km",
           "vehicle_count_est", "month", "day_of_month"]

print("Feature columns used:", list(X.columns))
assert set(X.columns) == set(CATEGORICAL + BINARY_FLAGS + NUMERIC)

# %% [7] TRAIN / TEST SPLIT (stratified because classes are imbalanced) --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)
print(y_train.value_counts(normalize=True))

# %% [8] PREPROCESSING PIPELINE ------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
        ("bin", "passthrough", BINARY_FLAGS),
    ]
)

# %% [9] MODEL 1 — Logistic Regression (baseline, with SMOTE + class_weight) ---
logreg_pipe = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                random_state=RANDOM_STATE)),
])
logreg_pipe.fit(X_train, y_train)
y_pred_lr = logreg_pipe.predict(X_test)
print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))

# %% [10] MODEL 2 — Random Forest (handles nonlinearity + imbalance natively) --
rf_pipe = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("clf", RandomForestClassifier(n_estimators=300, max_depth=None,
                                    min_samples_leaf=2, class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=-1)),
])
rf_pipe.fit(X_train, y_train)
y_pred_rf = rf_pipe.predict(X_test)
print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))

# %% [11] MODEL 3 — Gradient Boosting (often best for tabular, no imbalance flag) --
gb_pipe = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                        max_depth=3, random_state=RANDOM_STATE)),
])
gb_pipe.fit(X_train, y_train)
y_pred_gb = gb_pipe.predict(X_test)
print("=== Gradient Boosting ===")
print(classification_report(y_test, y_pred_gb))

# %% [12] MODEL COMPARISON (macro-F1 is the right metric for imbalanced multiclass) --
results = {
    "Logistic Regression": f1_score(y_test, y_pred_lr, average="macro"),
    "Random Forest": f1_score(y_test, y_pred_rf, average="macro"),
    "Gradient Boosting": f1_score(y_test, y_pred_gb, average="macro"),
}
results_df = pd.Series(results).sort_values(ascending=False)
print(results_df)

best_model_name = results_df.index[0]
best_pipe = {"Logistic Regression": logreg_pipe,
             "Random Forest": rf_pipe,
             "Gradient Boosting": gb_pipe}[best_model_name]
print(f"\nBest model: {best_model_name} (macro-F1={results_df.iloc[0]:.3f})")

# %% [13] CONFUSION MATRIX for best model --------------------------------------
best_pred = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, best_pred, labels=["Low", "Medium", "High"])
disp = ConfusionMatrixDisplay(cm, display_labels=["Low", "Medium", "High"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout()
plt.savefig("confusion_matrix_best_model.png", dpi=120)
plt.close()

# %% [14] FEATURE IMPORTANCE (if tree-based model won) -------------------------
if best_model_name in ("Random Forest", "Gradient Boosting"):
    ohe_cols = best_pipe.named_steps["preprocess"].named_transformers_["cat"] \
        .get_feature_names_out(CATEGORICAL)
    feature_names = NUMERIC + list(ohe_cols) + BINARY_FLAGS
    importances = best_pipe.named_steps["clf"].feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    fi.plot(kind="barh")
    plt.title(f"Top 15 feature importances — {best_model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.close()
    print(fi)

# %% [15] HYPERPARAMETER TUNING (grid search on the winning model family) -----
# Example shown for Random Forest — swap in whichever model won above.
param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [10, 12, 15],       # capped depth keeps the saved file small
    "clf__min_samples_leaf": [3, 5],      # larger leaves = fewer nodes = smaller file
}
# NOTE: uncapped max_depth (None) previously produced a 237MB model file —
# too large for GitHub (25MB limit) and slow to load in a Streamlit app.
# Capping depth + leaf size + estimator count keeps the file in the few-MB
# range with only a small macro-F1 tradeoff (~0.83 -> ~0.82).
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(rf_pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV macro-F1:", grid.best_score_)

tuned_pred = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, tuned_pred))

# %% [16] SAVE THE FINAL MODEL --------------------------------------------------
# compress=3 shrinks the file further (~30-50% smaller) with negligible
# load-time cost. Combined with the capped tree depth above, this keeps
# the saved model well under GitHub's 25MB per-file limit.
joblib.dump(grid.best_estimator_, "lagos_traffic_model.joblib", compress=3)
import os
size_mb = os.path.getsize("lagos_traffic_model.joblib") / 1e6
print(f"Saved model to lagos_traffic_model.joblib ({size_mb:.1f} MB)")

# %% [17] INFERENCE FUNCTION FOR DEPLOYMENT (e.g. behind a Flask API / n8n) ----
def predict_congestion(record: dict, model_path="lagos_traffic_model.joblib"):
    """
    record: dict with the same raw feature columns as training data
            (route, day_of_week, hour, is_weekend, is_public_holiday,
             is_school_day, lanes, length_km, has_toll, rain_intensity,
             visibility_km, has_accident, has_roadwork, has_police_checkpoint,
             has_event_nearby, fuel_scarcity, is_market_day_route,
             vehicle_count_est, month, day_of_month)
    """
    model = joblib.load(model_path)
    row = pd.DataFrame([record])
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    classes = model.named_steps["clf"].classes_ if hasattr(model, "named_steps") else model.classes_
    return {"prediction": pred, "probabilities": dict(zip(classes, proba.round(3)))}

# Example call:
# sample = X_test.iloc[0].to_dict()
# print(predict_congestion(sample))
