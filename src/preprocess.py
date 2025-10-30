import pandas as pd
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def split_cols(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols

def _onehot():
    # sklearn>=1.2 uses sparse_output; older uses sparse
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def make_preprocess(num_cols: List[str], cat_cols: List[str]):
    num_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", _onehot())
    ]
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline(cat_steps)
    # Only add branches that have columns
    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", num_pipe, num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers, remainder="drop")

def get_feature_names(ct: ColumnTransformer):
    """Return transformed feature names. Skips empty branches. Works across sklearn versions."""
    # Newer sklearn has this built-in
    try:
        return ct.get_feature_names_out().tolist()
    except Exception:
        pass

    names = []
    for name, trans, cols in ct.transformers_:
        # Skip dropped or empty branches
        if name == "remainder" or trans == "drop" or len(cols) == 0:
            continue

        # If it's a Pipeline, try its last step first
        if hasattr(trans, "named_steps"):
            last = trans.steps[-1][1]
            try:
                sub = last.get_feature_names_out(cols)
                names.extend(sub.tolist())
                continue
            except Exception:
                # fall back to pipeline.get_feature_names_out or raw cols
                try:
                    sub = trans.get_feature_names_out(cols)
                    names.extend(sub.tolist())
                    continue
                except Exception:
                    names.extend(list(cols))
                    continue

        # Plain transformer
        try:
            sub = trans.get_feature_names_out(cols)
            names.extend(sub.tolist())
        except Exception:
            names.extend(list(cols))

    return names
