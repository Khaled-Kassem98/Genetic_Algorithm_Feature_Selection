import pandas as pd
import numpy as np
from src.preprocess import split_cols, make_preprocess, get_feature_names
import numpy as np, random, os
np.random.seed(0)
random.seed(0)
os.environ["PYTHONHASHSEED"] = "0"
def test_preprocess_shapes_and_names():
    # tiny mixed dataset
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "num1": rng.normal(size=40),
        "num2": rng.normal(size=40),
        "cat1": np.where(rng.random(40) < 0.5, "A", "B"),
        "y":   np.where(rng.random(40) < 0.5, "neg", "pos"),
    })
    X, y, num_cols, cat_cols = split_cols(df, "y")
    ct = make_preprocess(num_cols, cat_cols)
    Xt = ct.fit_transform(X)
    names = get_feature_names(ct)

    assert Xt.shape[0] == len(df)
    assert Xt.shape[1] == len(names) > 0
    # one-hot expands cat1 into 2 columns
    assert set(num_cols) == {"num1", "num2"}
    assert any("cat1" in n for n in names)
