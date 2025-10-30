import pandas as pd, numpy as np
from src.preprocess import split_cols, make_preprocess, get_feature_names

def test_feature_names_num_only():
    df = pd.DataFrame({"x1":[1,2,3], "x2":[0.1,0.2,0.3], "y":[0,1,0]})
    X,y,num,cat = split_cols(df,"y")
    ct = make_preprocess(num, cat)
    _ = ct.fit_transform(X)
    names = get_feature_names(ct)
    assert all(n in names for n in ["num__x1","num__x2"]) or len(names)==2

def test_feature_names_cat_only():
    df = pd.DataFrame({"c":["a","b","a"], "y":[0,1,0]})
    X,y,num,cat = split_cols(df,"y")
    ct = make_preprocess(num, cat)
    _ = ct.fit_transform(X)
    names = get_feature_names(ct)
    assert any("c" in n for n in names)

def test_empty_branch_is_skipped():
    df = pd.DataFrame({"c":["a","b","a"], "y":[0,1,0]})
    X,y,num,cat = split_cols(df,"y")
    # ensure no numeric branch added → previously caused NotFittedError
    ct = make_preprocess(num, cat)
    _ = ct.fit_transform(X)
    names = get_feature_names(ct)
    assert len(names) >= 1
