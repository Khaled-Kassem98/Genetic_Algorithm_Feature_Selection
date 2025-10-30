import pandas as pd, yaml, pathlib

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_target(df, target):
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in columns")
