"""Tests for the Streamlit UI – basic rendering and button states."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestUIRendering:
    """Tests for basic Streamlit UI components using AppTest."""

    def test_app_loads_without_crash(self) -> None:
        """The app should load and render the title without crashing."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "ui.py"))
        at.run(timeout=10)
        assert not at.exception
        # Title should be present
        assert any(
            "Autonomous Research" in str(el.value)
            for el in at.title
        )

    def test_audit_button_present(self) -> None:
        """The 'Audit & Verify' button should be rendered."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "ui.py"))
        at.run(timeout=10)
        assert not at.exception
        button_labels = [b.label for b in at.button]
        assert any("Audit" in label for label in button_labels)

    def test_example_button_present(self) -> None:
        """The example button should be rendered."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "ui.py"))
        at.run(timeout=10)
        assert not at.exception
        button_labels = [b.label for b in at.button]
        assert any("Example" in label for label in button_labels)

    def test_text_area_present(self) -> None:
        """The text input area should be rendered."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "ui.py"))
        at.run(timeout=10)
        assert not at.exception
        assert len(at.text_area) > 0
