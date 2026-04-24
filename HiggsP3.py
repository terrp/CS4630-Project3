import os
import time
import argparse
import pandas as pd
import numpy as np

# Preprocessing & Evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# Phase 3a Classifiers
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

RANDOM_STATE = 42

# ─── 0. Setup Output Directory ───────────────────────────────────────────────
# Ensure the outputs folder exists before we try to save anything to it
os.makedirs('outputs', exist_ok=True)
METRICS_FILE = 'outputs/phase3a_metrics.csv'

# ─── 1. Data Loading & Preprocessing ─────────────────────────────────────────
print("Loading 200k subsample from HIGGS.csv.gz...")
# Col 0 is the label (signal=1, background=0), Cols 1-28 are features.
df = pd.read_csv('HIGGS.csv.gz', header=None, nrows=200000)

X_raw = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

print("Performing Train/Test Split (80/20)...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Scaling features using StandardScaler...")
scaler = StandardScaler()
# IMPORTANT: Fit the scaler ONLY on the training data to prevent data leakage!
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)


# ─── 2. Universal Evaluation Function ────────────────────────────────────────
def evaluate_model(model_name, y_true, y_pred, y_probs, train_time, inference_time):
    """Calculates metrics, prints to console, and appends to a CSV in outputs/."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)

    print(f"\n{'=' * 40}")
    print(f"Results for: {model_name}")
    print(f"{'=' * 40}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print("-" * 40)
    print(f"Train Time:     {train_time:.3f}s")
    print(f"Inference Time: {inference_time:.3f}s")
    print(f"{'=' * 40}\n")

    # Save to CSV in the outputs folder
    result_dict = {
        'Model': [model_name],
        'Accuracy': [acc],
        'F1_Score': [f1],
        'ROC_AUC': [roc_auc],
        'PR_AUC': [pr_auc],
        'Train_Time_s': [train_time],
        'Inference_Time_s': [inference_time]
    }
    result_df = pd.DataFrame(result_dict)

    # Append to CSV if it exists, otherwise write new file with headers
    if not os.path.isfile(METRICS_FILE):
        result_df.to_csv(METRICS_FILE, index=False)
    else:
        result_df.to_csv(METRICS_FILE, mode='a', header=False, index=False)

    print(f"Metrics appended to {METRICS_FILE}")


# ─── 3. Command-Line Model Execution ─────────────────────────────────────────
if __name__ == "__main__":
    # Set up the argument parser for PyCharm terminal execution
    parser = argparse.ArgumentParser(description="Run Phase 3a ML Models on HIGGS Dataset")
    parser.add_argument('--model', type=str, required=True,
                        choices=['linear_svm', 'rbf_svm', 'knn', 'dt', 'rf', 'xgb', 'all'],
                        help="Select the model to train and evaluate")
    args = parser.parse_args()

    # Define all Phase 3a models
    # Note: LinearSVC is wrapped in CalibratedClassifierCV to enable predict_proba for AUC metrics.
    classifiers = {
        'linear_svm': ('Linear SVM', CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE, dual=False))),
        'rbf_svm': ('RBF-kernel SVM', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)),
        'knn': ('k-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        'dt': ('Decision Tree', DecisionTreeClassifier(random_state=RANDOM_STATE)),
        'rf': ('Random Forest', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)),
        'xgb': ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE))
    }

    # Determine which models to run based on the command line argument
    models_to_run = classifiers.keys() if args.model == 'all' else [args.model]

    for m_key in models_to_run:
        model_name, model_obj = classifiers[m_key]
        print(f"\n>>> Initializing and Training {model_name}...")

        # Time the training phase
        t0_train = time.perf_counter()
        model_obj.fit(X_train, y_train)
        train_time = time.perf_counter() - t0_train

        # Time the inference phase
        t0_inf = time.perf_counter()
        y_pred = model_obj.predict(X_test)
        y_probs = model_obj.predict_proba(X_test)[:, 1]
        inference_time = time.perf_counter() - t0_inf

        # Run evaluation and save to outputs/
        evaluate_model(
            model_name=model_name,
            y_true=y_test,
            y_pred=y_pred,
            y_probs=y_probs,
            train_time=train_time,
            inference_time=inference_time
        )