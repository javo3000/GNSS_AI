# -*- coding: utf-8 -*-
"""
GNSS Signal Reliability Classifier
===================================
Classifies GNSS positioning signals into 3 reliability classes:
  0 = Reliable, 1 = Moderate, 2 = Unreliable

Uses the exactly 10 specified features from BeiDou satellite observations.
Compares Random Forest, XGBoost, and LightGBM models.
Training set: JFNG (10 & 18 Dec 2023) + URUM (10 Dec 2023)
Testing set: URUM (18 Dec 2023)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------
# 1. LOAD DATASETS
# ----------------------------------------------
print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

train_files = [
    "features_JFNG_10DEC.csv",
    "features_JFNG_18DEC.csv",
    "features_URUM_10DEC.csv"
]
test_file = "features_URUM_18DEC.csv"

# Load training data
train_dfs = []
for f in train_files:
    df = pd.read_csv(f, index_col=0)
    train_dfs.append(df)
    print(f"  Loaded Train File {f}: {df.shape[0]} rows")

train_data = pd.concat(train_dfs, ignore_index=True)
print(f"  Combined Train dataset: {train_data.shape[0]} rows")

# Load test data
test_data = pd.read_csv(test_file, index_col=0)
print(f"  Loaded Test File {test_file}: {test_data.shape[0]} rows")

# ----------------------------------------------
# 2. FEATURE SELECTION & PREPARATION
# ----------------------------------------------
print("\n" + "=" * 60)
print("PREPARING FEATURES")
print("=" * 60)

# Exactly the 10 features specified in Section 4 of the project plan
features = [
    "num_satellites",
    "pdop",
    "sdn",
    "sde",
    "pos_uncertainty",
    "snr_mean",
    "snr_min",
    "snr_max",
    "snr_std",
    "bds_sat_count_obs"
]
target_col = "reliability_class"

print(f"  Features ({len(features)}): {features}")

# Build training set
X_train = train_data[features]
y_train = train_data[target_col]

# Build testing set
X_test = test_data[features]
y_test = test_data[target_col]

print("\n  Train Target Distribution:")
for cls, count in y_train.value_counts().sort_index().items():
    print(f"    Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")

print("\n  Test Target Distribution:")
for cls, count in y_test.value_counts().sort_index().items():
    print(f"    Class {cls}: {count} ({count/len(y_test)*100:.1f}%)")

# ----------------------------------------------
# 3. DEFINE MODELS
# ----------------------------------------------
# Models configured exactly to the parameters in Section 7 of the project plan
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
}

# ----------------------------------------------
# 4. TRAINING & EVALUATION
# ----------------------------------------------
print("\n" + "=" * 60)
print("TRAINING & EVALUATION")
print("=" * 60)

results = {}

for name, model in models.items():
    print(f"\n{'-' * 50}")
    print(f"  Model: {name}")
    print(f"{'-' * 50}")
    
    # Train the model
    if name == "XGBoost":
        # Handle class imbalance using sample weights
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
        
    # Predict on train set to check for overfitting
    y_pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    f1_macro_train = f1_score(y_train, y_pred_train, average="macro")
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    # Recall and Precision for specific classes
    c2_recall = recall_score(y_test, y_pred, labels=[2], average=None)[0]
    c2_precision = precision_score(y_test, y_pred, labels=[2], average=None)[0]
    c0_recall = recall_score(y_test, y_pred, labels=[0], average=None)[0]
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Count dangerous error: True Class 2 predicted as Class 0
    dangerous_errors = cm[2][0] if cm.shape[0] > 2 and cm.shape[1] > 0 else 0
    
    # Normalize feature importances
    imp = model.feature_importances_
    imp_norm = imp / imp.sum() if imp.sum() > 0 else imp
    
    results[name] = {
        "accuracy_train": acc_train,
        "f1_macro_train": f1_macro_train,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "c2_recall": c2_recall,
        "c2_precision": c2_precision,
        "c0_recall": c0_recall,
        "dangerous_errors": dangerous_errors,
        "importances": imp_norm
    }
    
    print(f"  Train Accuracy:    {acc_train:.4f} | Train F1 (macro): {f1_macro_train:.4f}")
    print(f"  Test Accuracy:     {acc:.4f} | Test F1 (macro):  {f1_macro:.4f}")
    print(f"  F1 (weighted):     {f1_weighted:.4f}")
    print(f"  Class 2 Recall:    {c2_recall:.4f}")
    print(f"  Class 2 Precision: {c2_precision:.4f}")
    print(f"  Class 0 Recall:    {c0_recall:.4f}")
    print(f"  Dangerous Errors (True 2 -> Pred 0): {dangerous_errors}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Reliable", "Moderate", "Unreliable"]))
    
    print(f"  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Rel   Mod   Unr")
    for i, label in enumerate(["Reliable ", "Moderate ", "Unreliable"]):
        row_vals = cm[i] if i < cm.shape[0] else [0, 0, 0]
        # Pad row_vals to length 3 in case class is missing
        row_padded = list(row_vals) + [0] * (3 - len(row_vals))
        print(f"    Actual {label} {row_padded[0]:>5} {row_padded[1]:>5} {row_padded[2]:>5}")

# ----------------------------------------------
# 5. MODEL COMPARISON SUMMARY (Section 8 format)
# ----------------------------------------------
print("\n" + "=" * 60)
print("MODEL COMPARISON (SECTION 8 TABLE)")
print("=" * 60)

print(f"\n  {'Model':<20} | {'Train Acc':<10} | {'Test Acc':<10} | {'Class 2 Recall':<15} | {'Class 2 Precision':<18} | {'Class 0 Recall':<15} | {'Dangerous Errors':<16}")
print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*15}-+-{'-'*18}-+-{'-'*15}-+-{'-'*16}")

best_model = None
best_c2_recall = -1
best_accuracy = -1

for name, r in results.items():
    print(f"  {name:<20} | {r['accuracy_train']:<10.4f} | {r['accuracy']:<10.4f} | {r['c2_recall']:<15.4f} | {r['c2_precision']:<18.4f} | {r['c0_recall']:<15.4f} | {r['dangerous_errors']:<16}")
    
    # Selection priority: 1) Class 2 recall, 2) Accuracy
    if r['c2_recall'] > best_c2_recall:
        best_c2_recall = r['c2_recall']
        best_accuracy = r['accuracy']
        best_model = name
    elif abs(r['c2_recall'] - best_c2_recall) < 1e-5:
        if r['accuracy'] > best_accuracy:
            best_accuracy = r['accuracy']
            best_model = name

print(f"\n  * Recommended model: {best_model} (Class 2 Recall = {best_c2_recall:.4f}, Accuracy = {best_accuracy:.4f})")

# ----------------------------------------------
# 6. FEATURE IMPORTANCE RANKING COMPARISON
# ----------------------------------------------
print("\n" + "=" * 60)
# Check if feature importances agree across models
print("FEATURE IMPORTANCES BY MODEL")
print("=" * 60)

imp_df = pd.DataFrame(index=features)
for name, r in results.items():
    imp_df[name] = r["importances"]

imp_df["Average"] = imp_df.mean(axis=1)
imp_df = imp_df.sort_values(by="Average", ascending=False)

print("\n  Sorted by Average Importance (All normalized to sum to 1.0):")
print(imp_df.to_string(formatters={c: '{:,.4f}'.format for c in imp_df.columns}))

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
