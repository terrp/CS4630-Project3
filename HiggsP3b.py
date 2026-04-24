"""
HiggsP3b.py — Phase 3b: Integrating Supervised + Unsupervised Learning
CS 4630 · Group 2

Part A: Train classifiers on PCA-10 reduced features, compare vs raw-feature results from 3a.
Part B: Add K-Means cluster label (k=2) as an additional feature and retrain.

Usage:
    python HiggsP3b.py --part a          # PCA preprocessing comparison
    python HiggsP3b.py --part b          # Cluster label as feature
    python HiggsP3b.py --part all        # Run both sequentially

Outputs (appended to outputs/):
    outputs/phase3b_pca_metrics.csv      # Part A results
    outputs/phase3b_cluster_metrics.csv  # Part B results
"""

import os
import time
import argparse
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV

# Classifiers (same set as Phase 3a)
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

RANDOM_STATE = 42
os.makedirs('outputs', exist_ok=True)

# ─── 0. Shared constants ──────────────────────────────────────────────────────
PCA_PICKLE      = 'data/processed/pca_results.pkl'   # from Project 2
RAW_DATA_PATH   = 'data/raw/HIGGS.csv.gz'
NROWS           = 200_000

PARTA_FILE = 'outputs/phase3b_pca_metrics.csv'
PARTB_FILE = 'outputs/phase3b_cluster_metrics.csv'

# ─── 1. Load raw data & produce consistent train/test split ──────────────────
print("Loading 200k subsample from HIGGS.csv.gz...")
df = pd.read_csv(RAW_DATA_PATH, header=None, nrows=NROWS)

X_raw = df.iloc[:, 1:].values   # shape (200000, 28)
y     = df.iloc[:, 0].values    # labels

# Same 80/20 stratified split as Phase 3a — MUST match for fair comparison
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# StandardScaler fitted on raw training data only (no leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)

print(f"Train size: {X_train_raw.shape[0]:,}  |  Test size: {X_test_raw.shape[0]:,}")


# ─── 2. Load PCA results from Project 2 ──────────────────────────────────────
def load_pca_data(n_components=10):
    print(f"Loading PCA-{n_components} data from {PCA_PICKLE}...")
    with open(PCA_PICKLE, 'rb') as f:
        pca_results = pickle.load(f)

    X_pca_full = pca_results[n_components]['data']
    var_ratio   = pca_results[n_components]['explained_variance_ratio']
    print(f"  PCA-{n_components} cumulative variance explained: {var_ratio.sum()*100:.1f}%")

    # Reconstruct exactly which rows survived clean.py's dropna + drop_duplicates
    # so we can extract the matching y labels for the 199,909 PCA rows.
    df_raw = pd.read_csv(RAW_DATA_PATH, header=None, nrows=NROWS)
    df_raw.columns = ['label'] + [f'f{i}' for i in range(1, 29)]
    df_clean = df_raw.dropna().drop_duplicates()

    # df_clean.index gives us the original row positions that survived —
    # use those to pull the correct labels
    y_pca = df_clean['label'].values  # length 199909, aligned with X_pca_full

    assert len(y_pca) == len(X_pca_full), \
        f"Mismatch: y_pca={len(y_pca)}, X_pca={len(X_pca_full)}"

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_pca_full, y_pca,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y_pca
    )

    print(f"  PCA train: {X_train_pca.shape}  |  PCA test: {X_test_pca.shape}")
    return X_train_pca, X_test_pca, y_train_pca, y_test_pca


# ─── 3. Evaluation helper ────────────────────────────────────────────────────
def evaluate_and_save(model_name, feature_set_label, y_true, y_pred, y_probs,
                      train_time, inference_time, output_file):
    acc     = accuracy_score(y_true, y_pred)
    f1      = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc  = average_precision_score(y_true, y_probs)

    print(f"\n{'='*50}")
    print(f"  {model_name}  |  Features: {feature_set_label}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}   F1      : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}   PR-AUC  : {pr_auc:.4f}")
    print(f"  Train    : {train_time:.3f}s   Inference: {inference_time:.3f}s")

    row = pd.DataFrame([{
        'Model'           : model_name,
        'Feature_Set'     : feature_set_label,
        'Accuracy'        : acc,
        'F1_Score'        : f1,
        'ROC_AUC'         : roc_auc,
        'PR_AUC'          : pr_auc,
        'Train_Time_s'    : train_time,
        'Inference_Time_s': inference_time,
    }])
    write_header = not os.path.isfile(output_file)
    row.to_csv(output_file, mode='a', header=write_header, index=False)
    print(f"  Saved → {output_file}")


def train_and_eval(model_name, model_obj, X_tr, X_te, y_tr, y_te,
                   feature_label, output_file):
    """Fit, predict, evaluate, save."""
    print(f"\n>>> Training {model_name} on [{feature_label}]...")
    t0 = time.perf_counter()
    model_obj.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred  = model_obj.predict(X_te)
    y_probs = model_obj.predict_proba(X_te)[:, 1]
    inf_time = time.perf_counter() - t0

    evaluate_and_save(model_name, feature_label, y_te, y_pred, y_probs,
                      train_time, inf_time, output_file)


# ─── 4. Classifier definitions ───────────────────────────────────────────────
# Skip RBF-SVM — it took 2.95 hours in Phase 3a; impractical to repeat twice.
# LinearSVC wrapped in CalibratedClassifierCV for predict_proba support.
def get_classifiers():
    return {
        'Linear SVM'      : CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE, dual=False)),
        'k-NN'            : KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Decision Tree'   : DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest'   : RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }


# ─── 5. Part A — PCA-10 as preprocessing ─────────────────────────────────────
def run_part_a():
    print("\n" + "="*60)
    print("PART A: PCA-10 as Preprocessing for Classification")
    print("="*60)

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = load_pca_data(n_components=10)

    for name, clf in get_classifiers().items():
        train_and_eval(
            model_name=name,
            model_obj=clf,
            X_tr=X_train_pca,
            X_te=X_test_pca,
            y_tr=y_train_pca,
            y_te=y_test_pca,
            feature_label='PCA-10',
            output_file=PARTA_FILE,
        )

    print(f"\nPart A complete. Results saved to {PARTA_FILE}")
    print("Compare these against Phase 3a raw-feature results to see PCA impact.")


# ─── 6. Part B — K-Means cluster label as additional feature ─────────────────
def run_part_b():
    print("\n" + "="*60)
    print("PART B: K-Means Cluster Label as Additional Feature")
    print("="*60)

    # Fit K-Means (k=2) on the scaled training data only — same as Project 2 choice
    print("Fitting K-Means (k=2) on training set...")
    t0 = time.perf_counter()
    km = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_train_scaled)
    print(f"  K-Means fit in {time.perf_counter()-t0:.1f}s  |  iters={km.n_iter_}")

    # Predict cluster membership for train and test
    train_cluster = km.predict(X_train_scaled).reshape(-1, 1)
    test_cluster  = km.predict(X_test_scaled).reshape(-1, 1)

    # Augment feature matrices: 28 scaled features + 1 cluster ID = 29 features
    X_train_aug = np.hstack([X_train_scaled, train_cluster])
    X_test_aug  = np.hstack([X_test_scaled,  test_cluster])
    print(f"  Augmented feature shape: {X_train_aug.shape}  (28 raw + 1 cluster)")

    for name, clf in get_classifiers().items():
        train_and_eval(
            model_name=name,
            model_obj=clf,
            X_tr=X_train_aug,
            X_te=X_test_aug,
            y_tr=y_train,
            y_te=y_test,
            feature_label='Raw+ClusterID',
            output_file=PARTB_FILE,
        )

    print(f"\nPart B complete. Results saved to {PARTB_FILE}")
    print("Compare these against Phase 3a raw-feature results to see cluster label impact.")


# ─── 7. Comparison summary printer ──────────────────────────────────────────
def print_comparison_summary():
    """
    After both parts run, load all three CSVs and print a unified comparison table.
    Requires outputs/phase3a_metrics.csv to already exist from Phase 3a.
    """
    phase3a_file = 'outputs/phase3a_metrics.csv'
    if not all(os.path.isfile(f) for f in [phase3a_file, PARTA_FILE, PARTB_FILE]):
        print("\nCannot print summary — not all three metric files exist yet.")
        return

    raw_df = pd.read_csv(phase3a_file)[['Model','Accuracy','ROC_AUC','Train_Time_s']]
    raw_df['Feature_Set'] = 'Raw-28D'

    pca_df = pd.read_csv(PARTA_FILE)[['Model','Accuracy','ROC_AUC','Train_Time_s','Feature_Set']]
    clu_df = pd.read_csv(PARTB_FILE)[['Model','Accuracy','ROC_AUC','Train_Time_s','Feature_Set']]

    summary = pd.concat([raw_df, pca_df, clu_df], ignore_index=True)
    summary = summary.sort_values(['Model','Feature_Set'])
    summary['Accuracy'] = summary['Accuracy'].map('{:.4f}'.format)
    summary['ROC_AUC']  = summary['ROC_AUC'].map('{:.4f}'.format)
    summary['Train_Time_s'] = summary['Train_Time_s'].map('{:.2f}s'.format)

    print("\n" + "="*70)
    print("FULL COMPARISON SUMMARY — Raw vs PCA-10 vs Raw+ClusterID")
    print("="*70)
    print(summary.to_string(index=False))


# ─── 8. Entry point ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 3b: Supervised + Unsupervised Integration')
    parser.add_argument('--part', type=str, required=True,
                        choices=['a', 'b', 'all'],
                        help='a = PCA preprocessing, b = cluster label feature, all = both')
    args = parser.parse_args()

    if args.part in ('a', 'all'):
        run_part_a()
    if args.part in ('b', 'all'):
        run_part_b()
    if args.part == 'all':
        print_comparison_summary()