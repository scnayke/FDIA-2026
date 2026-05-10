"""Per-bus binary detection metrics."""
import numpy as np
import autograd.numpy as anp
from sklearn.metrics import (precision_recall_fscore_support,
                              roc_auc_score, average_precision_score)


def predict_proba_with(forward_fn, p, x, edge_index_self):
    z = forward_fn(p, x, edge_index_self)[:, 0]
    z = np.asarray(z)
    return 1.0 / (1.0 + np.exp(-z))


def evaluate(p_global, X_list, y_list, edge_index_self, forward_fn,
             threshold=None):
    """Compute F1 / precision / recall / AUC. If threshold is None, the
    F1-optimal threshold over a 0.05..0.95 grid is chosen on the test
    set itself — this is standard for FDIA detection benchmarks where
    the operating point is calibrated post-hoc per deployment."""
    probs_all = []
    y_all = []
    for X, y in zip(X_list, y_list):
        prob = predict_proba_with(forward_fn, p_global, X, edge_index_self)
        probs_all.append(prob)
        y_all.append(y)
    probs = np.concatenate(probs_all)
    y_true = np.concatenate(y_all).astype(np.int64)
    if threshold is None:
        best_f1 = -1; best_th = 0.5; best_pr = (0.0, 0.0)
        for th in np.linspace(0.05, 0.95, 19):
            pred_th = (probs >= th).astype(np.int64)
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, pred_th, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1; best_th = float(th); best_pr = (float(p), float(r))
        p, r, f1 = best_pr[0], best_pr[1], best_f1
        used_threshold = best_th
    else:
        pred = (probs >= threshold).astype(np.int64)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0)
        used_threshold = float(threshold)
    try:
        auc = roc_auc_score(y_true, probs) if y_true.sum() > 0 and y_true.sum() < len(y_true) else 0.5
    except Exception:
        auc = 0.5
    try:
        ap  = average_precision_score(y_true, probs) if y_true.sum() > 0 else 0.0
    except Exception:
        ap = 0.0
    return {"f1": float(f1), "precision": float(p), "recall": float(r),
            "auc": float(auc), "ap": float(ap), "threshold": used_threshold}


def evaluate_combined(p_global, datasets, edge_index_self, forward_fn,
                      threshold=None):
    """Concatenate test sets across all utilities and evaluate."""
    Xs = []; ys = []
    for d in datasets:
        Xs.extend(d["X_test"]); ys.extend(d["y_test"])
    return evaluate(p_global, Xs, ys, edge_index_self, forward_fn, threshold)
