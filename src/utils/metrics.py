import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)


def evaluate_all(model, X_te, y_te, metrics_list, threshold: float = 0.5):
    """
    Evaluate a binary classifier on a test set.

    Supported `metrics_list` keys:
      - "roc_auc": threshold-free AUC-ROC (rank separation).
      - "average_precision": PR AUC; robust to class imbalance.
      - "f1": harmonic mean of precision & recall at `threshold`.
      - "accuracy": share of correct predictions at `threshold`.
      - "precision": P(y=1 | pred=1) at `threshold`.
      - "recall": P(pred=1 | y=1) at `threshold`.
      - "lift": precision in top 10% by score / base rate.

    Returns a dict with requested metrics and always includes:
      - "threshold": the decision cut-off used.
      - "positives_rate": mean(preds) at that threshold.
    """
    # Get scores robustly
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, "decision_function"):
        # scale decision_function to [0,1] for AUC/PR consistency
        z = model.decision_function(X_te)
        p = (z - z.min()) / (z.max() - z.min() + 1e-9)
    else:
        # fall back to predictions (degenerate)
        p = model.predict(X_te).astype(float)

    preds = (p >= threshold).astype(int)
    out = {}
    # ROC AUC: threshold-free ranking quality
    if "roc_auc" in metrics_list:
        out["roc_auc"] = roc_auc_score(y_te, p)

    # AP (PR AUC): better when classes are imbalanced
    if "average_precision" in metrics_list:
        out["average_precision"] = average_precision_score(y_te, p)

    # F1: balance of precision & recall at the threshold
    if "f1" in metrics_list:
        out["f1"] = f1_score(y_te, preds)

    # Accuracy: share of correct predictions
    if "accuracy" in metrics_list:
        out["accuracy"] = accuracy_score(y_te, preds)

    # Precision: P(y=1 | pred=1)
    if "precision" in metrics_list:
        out["precision"] = precision_score(y_te, preds)

    # Recall: P(pred=1 | y=1)
    if "recall" in metrics_list:
        out["recall"] = recall_score(y_te, preds)

    # Lift@10%: precision in top decile / baseline prevalence
    if "lift" in metrics_list:
        n = len(p)
        k = max(1, int(np.ceil(n * 0.10)))
        if k > 0 and n > 0:
            # use positional indexing to avoid pandas index alignment issues
            order = np.argsort(p)
            top_indices = order[-k:]
            y_true = np.asarray(y_te)
            precision_top_10 = float(np.mean(y_true[top_indices]))
            baseline = float(np.mean(y_true)) if y_true.size > 0 else np.nan
            lift = precision_top_10 / baseline if baseline > 0 else np.nan
            out["lift"] = lift
        else:
            out["lift"] = np.nan

    out["threshold"] = threshold
    out["positives_rate"] = float(np.mean(preds))
    return out
