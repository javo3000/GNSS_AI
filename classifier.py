# -*- coding: utf-8 -*-
"""
GNSS Signal Reliability Classifier
===================================
Classifies GNSS positioning signals into 3 reliability classes:
  0 = Reliable, 1 = Moderate, 2 = Unreliable

Uses features from BeiDou satellite observations (SNR, PDOP, satellite count, 
position uncertainty, standard deviations, etc.)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------
# 1. LOAD & COMBINE DATASETS
# ----------------------------------------------
print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

files = [
    "features_JFNG_10DEC.csv",
    "features_JFNG_18DEC.csv",
    "features_URUM_10DEC.csv",
]

dfs = []
for f in files:
    df = pd.read_csv(f, index_col=0)
    dfs.append(df)
    print(f"  Loaded {f}: {df.shape[0]} rows")

data = pd.concat(dfs, ignore_index=True)
print(f"\n  Combined dataset: {data.shape[0]} rows x {data.shape[1]} columns")

# ----------------------------------------------
# 2. FEATURE ENGINEERING & SELECTION
# ----------------------------------------------
print("\n" + "=" * 60)
print("PREPARING FEATURES")
print("=" * 60)

# Columns to drop:
#   - station_date, split: identifiers, not features
#   - solution_quality, age, ratio: constant (zero variance)
#   - true_h_error_m: label leakage (ground truth used to derive target)
drop_cols = ["station_date", "split", "solution_quality", "age", "ratio", "true_h_error_m"]

# Target
target_col = "reliability_class"

# Build feature matrix and target vector
X = data.drop(columns=drop_cols + [target_col])
y = data[target_col]

feature_names = list(X.columns)
print(f"  Features ({len(feature_names)}): {feature_names}")
print(f"\n  Target distribution:")
for cls, count in y.value_counts().sort_index().items():
    print(f"    Class {cls}: {count} ({count/len(y)*100:.1f}%)")

# ----------------------------------------------
# 3. TRAIN / TEST SPLIT
# ----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {X_train.shape[0]} samples")
print(f"  Test:  {X_test.shape[0]} samples")

# ----------------------------------------------
# 4. SCALE FEATURES
# ----------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------
# 5. DEFINE MODELS
# ----------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    ),
}

# ----------------------------------------------
# 6. TRAIN, EVALUATE & COMPARE
# ----------------------------------------------
print("\n" + "=" * 60)
print("TRAINING & EVALUATION")
print("=" * 60)

results = {}

for name, model in models.items():
    print(f"\n{'-' * 50}")
    print(f"  Model: {name}")
    print(f"{'-' * 50}")

    # Use scaled data for LR and SVM, raw for tree-based
    if name in ("Logistic Regression", "SVM (RBF)"):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_data = X_train_scaled
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_data = X_train

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # Cross-validation (5-fold stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, cv_data, y_train, cv=cv, scoring="f1_macro")

    results[name] = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
    }

    print(f"  Accuracy:          {acc:.4f}")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"  F1 (weighted):     {f1_weighted:.4f}")
    print(f"  CV F1 (macro):     {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Reliable", "Moderate", "Unreliable"]))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                  Predicted")
    print(f"                  Rel   Mod   Unr")
    for i, label in enumerate(["Reliable ", "Moderate ", "Unreliable"]):
        print(f"    Actual {label} {cm[i][0]:>5} {cm[i][1]:>5} {cm[i][2]:>5}")

# ----------------------------------------------
# 7. MODEL COMPARISON SUMMARY
# ----------------------------------------------
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"\n  {'Model':<25} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12} {'CV F1 (mean)':>13}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12} {'-'*13}")

best_model = None
best_f1 = 0

for name, r in results.items():
    marker = ""
    if r["f1_macro"] > best_f1:
        best_f1 = r["f1_macro"]
        best_model = name
    print(f"  {name:<25} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} {r['f1_weighted']:>12.4f} {r['cv_f1_mean']:>8.4f} +/- {r['cv_f1_std']:.4f}")

print(f"\n  * Best model: {best_model} (F1 macro = {best_f1:.4f})")

# ----------------------------------------------
# 8. FEATURE IMPORTANCE (Best tree-based model)
# ----------------------------------------------
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)

rf = models["Random Forest"]
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print(f"\n  {'Rank':<6} {'Feature':<25} {'Importance':>12}")
print(f"  {'-'*6} {'-'*25} {'-'*12}")
for rank, idx in enumerate(indices, 1):
    bar = "#" * int(importances[idx] * 50)
    print(f"  {rank:<6} {feature_names[idx]:<25} {importances[idx]:>12.4f}  {bar}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
