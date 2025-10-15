import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(y_true, y_pred):
    """
    Computes core classification metrics for lens detection.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (0 = Lens, 1 = NoLens).
    y_pred : array-like
        Predicted probabilities for the 'Lens' class.

    Returns
    -------
    dict
        Dictionary containing AUC, AP, accuracy, precision, recall, and F1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_lens = 1 - y_true  # match convention: Lens = class 0

    auc = roc_auc_score(y_lens, y_pred)
    ap = average_precision_score(y_lens, y_pred)

    pred_binary = (y_pred >= 0.5).astype(int)
    precision = precision_score(y_lens, pred_binary, zero_division=0)
    recall = recall_score(y_lens, pred_binary, zero_division=0)
    f1 = f1_score(y_lens, pred_binary, zero_division=0)
    accuracy = accuracy_score(y_lens, pred_binary)

    return {
        "AUC": auc,
        "AP": ap,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }
