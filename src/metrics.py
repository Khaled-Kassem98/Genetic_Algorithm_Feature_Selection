import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve

def summary(y_true, y_prob, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

def curves(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    pr_p, pr_r, _ = precision_recall_curve(y_true, y_prob)
    cm = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
    return {"roc": (fpr, tpr), "pr": (pr_r, pr_p), "cm": cm}
