import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.preprocess import split_cols, make_preprocess
from src.ga import ga_feature_select

def _toy():
    rng = np.random.default_rng(0); n=160
    df = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": np.where(rng.random(n)<0.5, "X","Y"),
    })
    df["y"] = np.where(df["a"] + (df["c"]=="X").astype(int)*0.8 + rng.normal(scale=0.6, size=n) > 0, "P","N")
    X,y,num,cat = split_cols(df,"y")
    ct = make_preprocess(num,cat)
    Xt = ct.fit_transform(X)
    return Xt,y

def _build():
    return LogisticRegression(C=1.0, max_iter=200)

def _run(metric, pos_label=None):
    Xt,y = _toy()
    Xtr,Xte,ytr,yte = train_test_split(Xt,y,test_size=0.25, stratify=y, random_state=42)
    cfg = dict(population_size=10,generations=3,tournament_k=3,crossover_prob=0.8,mutation_prob=0.1,elitism=1,
               max_features=None, random_state=42, metric=metric, pos_label=pos_label)
    return ga_feature_select(Xtr,ytr,Xte,yte,_build,cfg)

def test_ga_accuracy():
    mask, best, scores = _run("accuracy")
    assert mask.sum()>=1 and 0<=best<=1 and len(scores)==10

def test_ga_f1_with_pos_label():
    mask, best, _ = _run("f1", pos_label="P")
    assert 0<=best<=1

def test_ga_roc_auc_with_pos_label():
    mask, best, _ = _run("roc_auc", pos_label="P")
    assert 0.5<=best<=1  # weak lower bound for separable toy data

def test_ga_metric_missing_pos_label_returns_zero():
    _, best, _ = _run("f1", pos_label=None)
    assert best == 0.0 or best <= 0.01
