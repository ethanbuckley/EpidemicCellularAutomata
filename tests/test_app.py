"""
test_app.py — the Streamlit dashboard loads without raising.

A headless smoke test using streamlit's AppTest: it runs app.py end-to-end
(loading data/results.json, rendering all three tabs, and executing one live
interactive simulation) and asserts the script finishes without an uncaught
exception. This guards against the class of breakage a deprecated Streamlit API
caused before (use_container_width, replaced by width="stretch").
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    return at


def test_dashboard_runs_without_exception(app):
    assert not app.exception, f"app.py raised: {list(app.exception)}"


def test_dashboard_renders_title(app):
    assert any(t.value == "SEIQR Epidemic Cellular Automaton" for t in app.title)
