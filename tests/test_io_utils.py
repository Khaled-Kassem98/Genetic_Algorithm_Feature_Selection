import pandas as pd, yaml, pathlib
from src.io_utils import load_csv, load_cfg, ensure_target

def test_load_csv_and_cfg_and_target(tmp_path):
    p = tmp_path/"t.csv"
    pd.DataFrame({"a":[1,2], "y":[0,1]}).to_csv(p, index=False)
    assert load_csv(str(p)).shape == (2,2)

    cfgp = tmp_path/"config.yaml"
    cfgp.write_text(yaml.safe_dump({"data":{"target":"y"}}), encoding="utf-8")
    cfg = load_cfg(str(cfgp))
    assert cfg["data"]["target"]=="y"

    df = pd.read_csv(p)
    ensure_target(df, "y")
