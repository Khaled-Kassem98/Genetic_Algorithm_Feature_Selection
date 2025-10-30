import streamlit as st
from pathlib import Path
from src.ui import inject_watermark
import base64

st.set_page_config(
    page_title="LogReg + GA",
    layout="wide"
)

inject_watermark(
    Path("logo") / "logo.svg",
                  size="100vmin", opacity=0.08,
                  position="center 55%"
                  )

logo_path = Path(__file__).parent / "logo" / "logo.svg"
if logo_path.exists():
    svg = logo_path.read_text(encoding="utf-8")
    # make text visible on dark UI
    svg = svg.replace('#0f172a', '#e5e7eb')  # wordmark color
    b64 = base64.b64encode(svg.encode("utf-8")).decode()
    st.sidebar.markdown(
        f'<div style="text-align:center;padding:8px 0;">'
        f'<img src="data:image/svg+xml;base64,{b64}" style="max-width:200px;width:100%;" alt="logo"/></div>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.warning("assets/logo.svg not found")

st.title("Logistic Regression with GA Feature Selection")
st.write("Use the pages on the left.")
