import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.preprocess import split_cols, make_preprocess
from src.model import train_eval_logreg
import numpy as np, random, os
np.random.seed(0)
random.seed(0)
os.environ["PYTHONHASHSEED"] = "0"
def test_logreg_train_eval_runs():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "x1": rng.normal(size=60),
        "x2": rng.normal(size=60),
        "grp": np.where(rng.random(60) < 0.3, "G1", "G2"),
    })
    df["y"] = (df["x1"] + 0.5 * (df["grp"] == "G1").astype(int) + rng.normal(scale=0.3, size=60) > 0).astype(int)

    X, y, num_cols, cat_cols = split_cols(df, "y")
    ct = make_preprocess(num_cols, cat_cols)
    Xt = ct.fit_transform(X)

    clf, (Xtr, Xte, ytr, yte) = train_eval_logreg(Xt, y, feature_mask=None, C=1.0, max_iter=200, random_state=42, test_size=0.2)
    assert isinstance(clf, LogisticRegression)
    assert Xtr.shape[1] == Xte.shape[1] == Xt.shape[1]
    assert len(ytr) + len(yte) == len(y)
