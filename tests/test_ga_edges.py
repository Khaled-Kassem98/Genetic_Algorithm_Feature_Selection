import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.preprocess import split_cols, make_preprocess
from src.ga import ga_feature_select, evaluate_mask

def _toy_mc():
    rng = np.random.default_rng(0); n=120
    df = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": np.where(rng.random(n)<0.5, "X","Y"),
    })
    # three classes to hit multi-class guard
    y = np.where(df["a"]>0.8, "C2", np.where(df["a"]<-0.8, "C0", "C1"))
    return df.drop(columns=[]), pd.Series(y, name="y")

def _prep(df, yname="y"):
    X, y, num, cat = split_cols(df, yname)
    ct = make_preprocess(num, cat)
    Xt = ct.fit_transform(X)
    return Xt, y

def _build():
    return LogisticRegression(C=1.0, max_iter=200)

def test_evaluate_mask_zero_features_returns_zero():
    df = pd.DataFrame({"x":[0,1,2,3], "y":[0,1,0,1]})
    Xt, y = _prep(df)
    Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=0.5, stratify=y, random_state=0)
    mask = np.zeros(Xt.shape[1], dtype=bool)
    score = evaluate_mask(mask, Xtr, ytr, Xte, yte, _build, metric="accuracy", pos_label=None)
    assert score == 0.0

def test_ga_missing_pos_label_makes_f1_zero():
    rng = np.random.default_rng(1); n=80
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = (df["x"]>0).astype(int)
    Xt, y = _prep(df)
    Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=0.25, stratify=y, random_state=0)
    cfg = dict(population_size=10, generations=2, tournament_k=3, crossover_prob=0.8, mutation_prob=0.1,
               elitism=1, max_features=None, random_state=0, metric="f1", pos_label=None)
    _, best, _ = ga_feature_select(Xtr, ytr, Xte, yte, _build, cfg)
    assert best == 0.0

def test_ga_multiclass_auc_returns_zero():
    df, y = _toy_mc()
    df = df.assign(y=y.values)
    Xt, y = _prep(df, "y")
    Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=0.3, stratify=y, random_state=0)
    cfg = dict(population_size=8, generations=2, tournament_k=2, crossover_prob=0.9, mutation_prob=0.2,
               elitism=1, max_features=None, random_state=0, metric="roc_auc", pos_label="C1")
    _, best, _ = ga_feature_select(Xtr, ytr, Xte, yte, _build, cfg)
    assert best == 0.0
