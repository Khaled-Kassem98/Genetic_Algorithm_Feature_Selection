import pandas as pd
import pytest
from src.io_utils import ensure_target

def test_ensure_target_raises_when_missing():
    df = pd.DataFrame({"a":[1,2]})
    with pytest.raises(ValueError):
        ensure_target(df, "y")
