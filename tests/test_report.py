from pathlib import Path

from gotrain.report.fixture import build_fixture_home
from gotrain.report.html import render_dashboard
from gotrain.report.png import encode_png
import numpy as np


def test_png_encoder_produces_valid_signature():
    data = encode_png(np.zeros((4, 5, 3), dtype=np.uint8))
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and b"IHDR" in data and b"IEND" in data


def test_fixture_dashboard_renders_all_sections_without_external_resources(tmp_path):
    home = build_fixture_home(tmp_path / "fixture", with_agent=True, with_final_eval=True, log=lambda s: None)
    out = render_dashboard(home)
    txt = Path(out).read_text()
    for needle in ("SYNTHETIC FIXTURE", "Task overview", "Learning curves", "Selected checkpoint", "Model registry", "verified: model and preprocessing reproduced", "Test results", "screening, partial data", "simulated failure", "data:image/png;base64"):
        assert needle in txt, needle
    assert "<script" not in txt
    assert "http" not in txt.replace("http://www.w3.org/2000/svg", "")


def test_empty_home_renders_missing_markers(tmp_path):
    from gotrain.paths import Home

    out = render_dashboard(Home(tmp_path / "empty").ensure())
    txt = Path(out).read_text()
    assert "Missing results for required tasks" in txt and "m-chesapeake" in txt
    assert "No agent sessions" in txt and "No model versions" in txt
