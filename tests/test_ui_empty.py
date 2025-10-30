from src.ui import build_watermark_css
def test_build_watermark_css_missing_file():
    css = build_watermark_css("does_not_exist.svg")
    assert css == ""
