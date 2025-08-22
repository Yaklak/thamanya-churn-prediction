from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np

def evaluate_all(model, X_te, y_te, metrics_list, threshold: float = 0.5):
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
    if "roc_auc" in metrics_list: out["roc_auc"] = roc_auc_score(y_te, p)
    if "average_precision" in metrics_list: out["average_precision"] = average_precision_score(y_te, p)
    if "f1" in metrics_list: out["f1"] = f1_score(y_te, preds)
    out["threshold"] = threshold
    out["positives_rate"] = float(np.mean(preds))
    return out