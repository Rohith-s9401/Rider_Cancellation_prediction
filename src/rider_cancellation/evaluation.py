"""Imbalanced classification metrics and validation-only decision selection."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score)


def choose_threshold(y, probability, false_negative_cost=10., false_positive_cost=1.):
    if false_negative_cost <= 0 or false_positive_cost <= 0:
        raise ValueError("Both illustrative costs must be positive")
    precision, recall, thresholds = precision_recall_curve(y, probability)
    positives = np.asarray(y).sum()
    tp = recall[:-1] * positives
    fp = tp / np.maximum(precision[:-1], 1e-15) - tp
    costs = false_negative_cost * (positives - tp) + false_positive_cost * fp
    # Include the valid 'intervene on nobody' policy.
    thresholds = np.r_[thresholds, np.nextafter(1., 2.)]
    costs = np.r_[costs, false_negative_cost * positives]
    best = int(np.argmin(costs))
    return float(thresholds[best])


def metrics(y, probability, threshold):
    y = np.asarray(y)
    p = np.clip(np.asarray(probability), 0, 1)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    k = max(1, int(np.ceil(len(y) * .05)))
    # Stable input order breaks probability ties deterministically.
    top = np.argsort(-p, kind="stable")[:k]
    base = float(y.mean())
    return {"average_precision": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "brier_score": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "alert_rate": float(pred.mean()), "threshold": float(threshold),
            "precision_at_top_5_percent": float(y[top].mean()),
            "recall_at_top_5_percent": float(y[top].sum() / y.sum()),
            "lift_at_top_5_percent": float(y[top].mean() / base),
            "cancellation_rate": base,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
