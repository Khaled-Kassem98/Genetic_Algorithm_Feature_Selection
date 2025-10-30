import pandas as pd
import numpy as np
import types
from sklearn.compose import ColumnTransformer
from src.preprocess import split_cols, make_preprocess, get_feature_names

def test_get_feature_names_fallback(monkeypatch):
    df = pd.DataFrame({
        "num1": [1.0, 2.5, 3.2],
        "cat1": ["a", "b", "a"],
        "y": [0, 1, 0],
    })
    X, y, num, cat = split_cols(df, "y")
    ct = make_preprocess(num, cat)
    _ = ct.fit_transform(X)

    # Force fallback by breaking ct.get_feature_names_out
    def boom(*args, **kwargs):
        raise AttributeError("boom")
    monkeypatch.setattr(ColumnTransformer, "get_feature_names_out", boom, raising=True)

    names = get_feature_names(ct)
    assert isinstance(names, list)
    assert len(names) >= 2
    assert any("cat1" in n for n in names)
