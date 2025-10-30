import numpy as np
import pandas as pd
from src.preprocess import split_cols, make_preprocess
from src.model import train_eval_logreg

def test_train_eval_with_feature_mask():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x1": rng.normal(size=50),
        "x2": rng.normal(size=50),
        "c": np.where(rng.random(50) < 0.5, "A", "B"),
    })
    df["y"] = (df["x1"] + (df["c"] == "A").astype(int) + rng.normal(scale=0.2, size=50) > 0).astype(int)

    X, y, num, cat = split_cols(df, "y")
    ct = make_preprocess(num, cat)
    Xt = ct.fit_transform(X)

    # keep only first two transformed features
    mask = np.zeros(Xt.shape[1], dtype=bool)
    mask[:2] = True

    clf, (Xtr, Xte, ytr, yte) = train_eval_logreg(Xt, y, feature_mask=mask, C=1.0, max_iter=200, random_state=0, test_size=0.2)
    assert Xtr.shape[1] == 2 and Xte.shape[1] == 2
    assert set(getattr(clf, "classes_", [])) == {0, 1}
