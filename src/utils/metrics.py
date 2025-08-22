from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
def evaluate_all(model, X_te, y_te, metrics_list):
    p = model.predict_proba(X_te)[:,1]
    preds = (p >= 0.5).astype(int)
    out = {}
    if "roc_auc" in metrics_list: out["roc_auc"] = roc_auc_score(y_te, p)
    if "average_precision" in metrics_list: out["average_precision"] = average_precision_score(y_te, p)
    if "f1" in metrics_list: out["f1"] = f1_score(y_te, preds)
    return out
