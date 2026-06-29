# -*- coding: utf-8 -*-
"""
GNSS Reliability Visualization Generator
========================================
Generates:
1. Confusion matrices (3-panel plot: RF, XGBoost, LightGBM)
2. Normalized Feature Importance comparison plot
Saves directly into the Brain artifacts folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix
import os

# Paths
brain_dir = r"C:\Users\JAVO\.gemini\antigravity-ide\brain\30548af6-2e05-4d74-81d8-314985bccfde"
cm_path = os.path.join(brain_dir, "confusion_matrices.png")
feat_path = os.path.join(brain_dir, "feature_importances.png")

# Load datasets
train_files = ["features_JFNG_10DEC.csv", "features_JFNG_18DEC.csv", "features_URUM_10DEC.csv"]
test_file = "features_URUM_18DEC.csv"

train_dfs = [pd.read_csv(f, index_col=0) for f in train_files]
train_data = pd.concat(train_dfs, ignore_index=True)
test_data = pd.read_csv(test_file, index_col=0)

features = [
    "num_satellites", "pdop", "sdn", "sde", "pos_uncertainty",
    "snr_mean", "snr_min", "snr_max", "snr_std", "bds_sat_count_obs"
]
target_col = "reliability_class"

X_train = train_data[features]
y_train = train_data[target_col]
X_test = test_data[features]
y_test = test_data[target_col]

# Define models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="mlogloss", random_state=42, n_jobs=-1
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.1, class_weight="balanced", random_state=42, verbosity=-1, n_jobs=-1
    )
}

results = {}

# Train and collect confusion matrices & importances
for name, model in models.items():
    if name == "XGBoost":
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize importances
    imp = model.feature_importances_
    imp_norm = imp / imp.sum()
    
    results[name] = {
        "cm": cm,
        "importances": imp_norm
    }

# ------------------------------------------------------------
# PLOT 1: Confusion Matrices (3-panel)
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
classes = ["Reliable", "Moderate", "Unreliable"]

for ax, (name, r) in zip(axes, results.items()):
    cm = r["cm"]
    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{name} Confusion Matrix", fontsize=14, fontweight='bold', pad=12)
    
    # Tick marks
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=11)
    
    # Label cell values
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')
            
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=8)
    ax.set_ylabel('True Label', fontsize=12, labelpad=8)
    
# Add single unified colorbar on the right side
fig.subplots_adjust(right=0.88, wspace=0.3)
cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.savefig(cm_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved confusion matrices to {cm_path}")

# ------------------------------------------------------------
# PLOT 2: Feature Importances
# ------------------------------------------------------------
imp_df = pd.DataFrame(index=features)
for name, r in results.items():
    imp_df[name] = r["importances"]
imp_df["Average"] = imp_df.mean(axis=1)
imp_df = imp_df.sort_values(by="Average", ascending=True) # Ascending for horizontal bar chart

plt.figure(figsize=(10, 6.5))
y_pos = np.arange(len(features))
height = 0.25

plt.barh(y_pos - height, imp_df["Random Forest"], height, label="Random Forest", color="#3182bd")
plt.barh(y_pos, imp_df["XGBoost"], height, label="XGBoost", color="#e6550d")
plt.barh(y_pos + height, imp_df["LightGBM"], height, label="LightGBM", color="#31a354")

plt.yticks(y_pos, imp_df.index, fontsize=11)
plt.xlabel("Normalized Importance Value", fontsize=12, labelpad=8)
plt.title("BDS Signal Reliability Feature Importance Comparison", fontsize=14, fontweight='bold', pad=12)
plt.legend(fontsize=11, loc="lower right")
plt.grid(axis='x', linestyle='--', alpha=0.5)

plt.savefig(feat_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved feature importances to {feat_path}")
