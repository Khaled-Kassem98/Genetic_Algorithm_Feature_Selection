import streamlit as st, pandas as pd
from pathlib import Path
from src.io_utils import load_csv, load_cfg
from src.ui import inject_watermark


inject_watermark(
    Path("logo") / "logo.svg",
                  size="100vmin", opacity=0.08,
                  position="center 55%"
                  )

st.title("Data & Preview")
cfg = load_cfg()
st.session_state.setdefault("df", None)
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = load_csv(cfg["paths"]["default_csv"])
st.session_state["df"] = df
st.write("Shape:", df.shape)
st.dataframe(df.head(cfg["ui"]["preview_rows"]))
st.session_state["target"] = st.selectbox("Target column", options=df.columns, index=list(df.columns).index(cfg["data"]["target"]) if cfg["data"]["target"] in df.columns else 0)
