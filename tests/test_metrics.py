import numpy as np
from src.metrics import summary, curves

def test_summary_and_curves_binary():
    y_true = np.array([0,1,1,0,1,0,1,1])
    y_prob = np.array([0.1,0.8,0.6,0.4,0.9,0.2,0.55,0.7])
    y_pred = (y_prob>=0.5).astype(int)
    s = summary(y_true, y_prob, y_pred)
    assert set(s.keys()) == {"accuracy","precision","recall","f1","roc_auc"}
    f = curves(y_true, y_prob)
    assert "roc" in f and "pr" in f and "cm" in f
