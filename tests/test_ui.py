import types
from pathlib import Path
from src.ui import set_svg_watermark

class DummySt:
    def __init__(self): self.markdowns=[]
    def markdown(self, s, unsafe_allow_html=False): self.markdowns.append((s, unsafe_allow_html))

def test_set_svg_watermark(tmp_path, monkeypatch):
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><text fill="#0f172a">X</text></svg>'
    p = tmp_path/"logo.svg"; p.write_text(svg, encoding="utf-8")
    dummy = DummySt()
    import src.ui as ui
    monkeypatch.setattr(ui, "st", dummy)
    set_svg_watermark(p, size="10vmin", position="center", opacity=0.1)
    assert dummy.markdowns and "data:image/svg+xml;base64" in dummy.markdowns[0][0]

