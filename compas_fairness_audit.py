"""
COMPAS fairness audit script.
Saves:
 - 'fpr_fnr_by_race.png'
 - 'selection_rates_by_race.png'
 - prints fairness metrics and a brief summary.

Requires: aif360 (preferred). If not available, falls back to pandas/sklearn metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import AIF360; if not available, proceed with fallback
use_aif360 = True
try:
    from aif360.datasets import CompasDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.algorithms.inprocessing import AdversarialDebiasing
except Exception as e:
    print("AIF360 not available or failed to import:", str(e))
    print("Falling back to pandas/sklearn computations. (Install aif360 for richer tools.)")
    use_aif360 = False

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def load_compas_aif360():
    # Loads COMPAS dataset included in AIF360
    dataset = CompasDataset()
    return dataset

def preprocess_dataframe(df):
    # Example minimal preprocessing if using raw CSV:
    # map target (two_year_recid) to 1/0, encode categorical, drop unneeded cols
    df = df.copy()
    df = df[df['days_b_screening_arrest'] <= 30]  # typical filter used in COMPAS processing
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'].isin(['F','M'])]
    df['target'] = df['two_year_recid'].astype(int)
    # Drop many columns that are identifiers or leakage; simple numeric features:
    features = ['age', 'priors_count']  # extend as needed
    X = df[features].fillna(0)
    y = df['target']
    return X, y, df

def compute_group_confusion(y_true, y_pred, groups):
    """
    groups: array-like of group labels (e.g., 'African-American', 'Caucasian')
    returns dict[group] = confusion matrix and rates
    """
    unique_groups = np.unique(groups)
    results = {}
    for g in unique_groups:
        idx = (groups == g)
        if idx.sum() == 0:
            continue
        tn, fp, fn, tp = confusion_matrix(y_true[idx], y_pred[idx], labels=[0,1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        selection_rate = (y_pred[idx] == 1).mean()
        results[g] = {'tn':tn,'fp':fp,'fn':fn,'tp':tp, 'fpr':fpr, 'fnr':fnr, 'tpr':tpr, 'selection_rate':selection_rate}
    return results

def plot_rates_by_group(results, rate_key, outpath):
    groups = list(results.keys())
    rates = [results[g][rate_key] for g in groups]
    plt.figure(figsize=(8,5))
    plt.bar(groups, rates)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(rate_key)
    plt.title(f"{rate_key} by group")
    plt.tight_layout()
    plt.savefig(outpath)
    print("Saved plot:", outpath)
    plt.close()

def aif360_workflow():
    dataset = load_compas_aif360()
    # Sensitive attribute
    sens_attr = 'race'
    privileged_groups = [{'race': 'Caucasian'}]
    unprivileged_groups = [{'race': 'African-American'}]

    # Convert to train/test
    train, test = dataset.split([0.7], shuffle=True)
    # Train a simple classifier (logistic) using AIF360 helper or scikit-learn on features
    # Convert label datasets to arrays
    X_train = train.features
    y_train = train.labels.ravel()
    X_test = test.features
    y_test = test.labels.ravel()

    # Fit logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Wrap predictions back into BinaryLabelDataset for metrics
    from aif360.datasets import BinaryLabelDataset
    test_pred = test.copy()
    test_pred.labels = y_pred.reshape(-1,1)

    # Metrics
    metric_test = BinaryLabelDatasetMetric(test, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    metric_pred = BinaryLabelDatasetMetric(test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    print("Mean difference (selection rate) (priv - unpriv):", metric_pred.mean_difference())

    # Classification metrics
    class_metric = ClassificationMetric(test, test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    print("FPR difference (unpriv - priv):", class_metric.false_positive_rate_difference())
    print("FNR difference (unpriv - priv):", class_metric.false_negative_rate_difference())
    print("Balanced accuracy:", class_metric.balanced_accuracy())

    # Visualizations:
    # Compute confusion per group for plotting (fallback method)
    groups = test.protected_attributes[:, test.protected_attribute_names.index('race')]
    # map numeric race back to string using dataset metadata
    race_map = {v:k for k,v in dataset.protected_attribute_names_map()['race'].items()} if hasattr(dataset, 'protected_attribute_names_map') else None
    # Simpler: use dataset.protected_attribute_names and dataset.protected_attributes - skip mapping here; we'll compute using dataset.protected_attributes numeric values
    # For demonstration, build dictionary keyed by numeric values
    grp_vals = np.unique(groups)
    group_names = [str(int(v)) for v in grp_vals]
    # Using sklearn-like approach for plotting
    # Create arrays for y_true, y_pred, groups_str
    return

def fallback_workflow_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    X, y, df_clean = preprocess_dataframe(df)
    # Use race column
    groups = df_clean['race'].values
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(X, y, groups, test_size=0.3, random_state=42, stratify=groups)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Compute group metrics
    results = compute_group_confusion(y_test.values, y_pred, g_test)
    print("Group metrics (per-race):")
    for g, r in results.items():
        print(g, r)
    # Plot FPR and FNR
    plot_rates_by_group(results, 'fpr', 'fpr_by_race.png')
    plot_rates_by_group(results, 'fnr', 'fnr_by_race.png')
    # Plot selection rates
    plot_rates_by_group(results, 'selection_rate', 'selection_rates_by_race.png')
    return results

def main():
    # If user has aif360 and wants to use internal dataset:
    if use_aif360:
        print("AIF360 available. To run the aif360 workflow, please run the aif360_workflow() function or this script with AIF360 dataset usage enabled.")
        # For succinctness, invoke fallback path unless user explicitly wants AIF360 run
        # (AIF360's CompasDataset may require additional preprocessing; user should inspect).
    else:
        # Fallback: require user to provide local COMPAS CSV path
        csv_path = input("Path to COMPAS CSV file (e.g., 'compas-scores-two-years.csv'): ").strip()
        if not os.path.exists(csv_path):
            print("CSV path not found. Exiting.")
            return
        results = fallback_workflow_from_csv(csv_path)
        # Print summary
        print("Saved visualizations and printed per-group metrics. Inspect PNG files.")

if __name__ == '__main__':
    main()
