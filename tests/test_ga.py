import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.preprocess import split_cols, make_preprocess
from src.ga import ga_feature_select
import numpy as np, random, os
np.random.seed(0)
random.seed(0)
os.environ["PYTHONHASHSEED"] = "0"
def test_ga_returns_valid_mask_and_score():
    rng = np.random.default_rng(2)
    n = 120
    df = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": np.where(rng.random(n) < 0.5, "X", "Y"),
    })
    # signal in a and c
    df["y"] = np.where(df["a"] + (df["c"] == "X").astype(int)*0.8 + rng.normal(scale=0.6, size=n) > 0, "P", "N")

    X, y, num_cols, cat_cols = split_cols(df, "y")
    ct = make_preprocess(num_cols, cat_cols)
    Xt = ct.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=0.25, stratify=y, random_state=42)

    def build_clf():
        return LogisticRegression(C=1.0, max_iter=200)

    cfg = dict(
        population_size=20, generations=5, tournament_k=3,
        crossover_prob=0.8, mutation_prob=0.1, elitism=1,
        max_features=None, random_state=42, metric="accuracy"
    )
    mask, best, scores = ga_feature_select(Xtr, ytr, Xte, yte, build_clf, cfg)

    assert mask.dtype == bool and mask.shape[0] == Xtr.shape[1]
    assert mask.sum() >= 1
    assert 0.0 <= best <= 1.0
    assert len(scores) == cfg["population_size"]
