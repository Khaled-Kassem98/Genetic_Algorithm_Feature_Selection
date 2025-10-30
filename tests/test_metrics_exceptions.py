import numpy as np
from src.metrics import summary

def test_summary_auc_exception_path():
    y_true = np.array([0,0,0,0])   # single-class true labels → roc_auc should raise
    y_prob = np.array([0.1,0.2,0.3,0.4])
    y_pred = (y_prob>=0.5).astype(int)
    s = summary(y_true, y_prob, y_pred)
    assert np.isnan(s["roc_auc"])
