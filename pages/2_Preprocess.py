import streamlit as st
from sklearn.model_selection import train_test_split
from src.preprocess import split_cols, make_preprocess, get_feature_names
from src.io_utils import load_cfg
from src.ui import inject_watermark
from pathlib import Path

inject_watermark(
    Path("logo") / "logo.svg",
                  size="100vmin", opacity=0.08,
                  position="center 55%"
                  )

st.title("Preprocess")
cfg = load_cfg()
df = st.session_state.get("df"); target = st.session_state.get("target")
if df is None or target is None:
    st.stop()

X, y, num_cols, cat_cols = split_cols(df, target)
ct = make_preprocess(num_cols, cat_cols)
Xt = ct.fit_transform(X)
feat_names = get_feature_names(ct)

st.session_state["preprocess"] = {"ct": ct, "feat_names": feat_names}
st.write("Numerical:", ",".join(num_cols))
st.write("Categorical:", ",".join(cat_cols))
st.write("Transformed shape:", Xt.shape)
st.dataframe(X.head(20))
st.caption("Above: raw features. Encoded and scaled in the model step.")

