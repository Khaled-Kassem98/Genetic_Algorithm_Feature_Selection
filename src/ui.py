# src/ui.py
from pathlib import Path
import base64, re
import streamlit as st

def build_watermark_css(svg_path: str | Path, *, size="50vmin",
                        position="center 55%", opacity=0.08) -> str:
    """Return CSS+HTML for a fixed SVG overlay watermark."""
    p = Path(svg_path)
    if not p.exists():
        return ""
    svg = p.read_text(encoding="utf-8")
    svg = svg.replace("#0f172a", "#e5e7eb")             # light wordmark on dark
    svg = re.sub(r"<svg\b", f'<svg opacity="{opacity}"', svg, count=1)
    b64 = base64.b64encode(svg.encode("utf-8")).decode()
    return f"""
    <style>
      #__wm_overlay__ {{
        position: fixed;
        inset: 0;
        background-image: url("data:image/svg+xml;base64,{b64}");
        background-repeat: no-repeat;
        background-position: {position};
        background-size: {size};
        pointer-events: none;
        z-index: 9999;
        mix-blend-mode: normal;
      }}
      /* keep app above page background layers */
      .block-container {{ position: relative; z-index: 1; }}
      header, [data-testid="stSidebar"] {{ position: relative; z-index: 2; }}
    </style>
    <div id="__wm_overlay__"></div>
    """

def inject_watermark(svg_path: str | Path, *, size="50vmin",
                     position="center 55%", opacity=0.08):
    css = build_watermark_css(svg_path, size=size, position=position, opacity=opacity)
    if css:
        st.markdown(css, unsafe_allow_html=True)

# Backward compatibility
def set_svg_watermark(svg_path: str | Path, *, size="50vmin",
                      position="center 55%", opacity=0.08):
    inject_watermark(svg_path, size=size, position=position, opacity=opacity)
