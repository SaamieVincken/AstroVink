"""
Metrics include standard classification trackers.

All functions treat label 0 as the positive class (Lens).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute accuracy, precision, recall, f1 (macro), and optionally AUC.
    """
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }

    if y_prob is not None:
        y_true_np = np.asarray(y_true)
        binary_true = (y_true_np == 0).astype(int)
        results["auc"] = float(roc_auc_score(binary_true, y_prob))

    return results
