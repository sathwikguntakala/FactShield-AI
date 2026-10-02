"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Skip tests marked @pytest.mark.live when no API key is present."""
    import os

    if not os.environ.get("GEMINI_API_KEY"):
        skip_live = pytest.mark.skip(reason="No GEMINI_API_KEY set")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)


class FakeDDGS:
    """Fake DDGS client for offline testing."""

    def __init__(self, results: list[dict[str, str]] | None = None) -> None:
        self._results = results or [
            {
                "title": "Test Result 1",
                "href": "https://example.com/page1",
                "body": "This is a test snippet for result 1.",
            },
            {
                "title": "Test Result 2",
                "href": "https://example.org/page2",
                "body": "Another test snippet with different info.",
            },
            {
                "title": "Test Result 3",
                "href": "https://official.gov/page3",
                "body": "Official government source snippet.",
            },
        ]
        self.call_log: list[dict[str, Any]] = []

    def text(self, keywords: str, max_results: int = 4, **kwargs: Any) -> list[dict[str, str]]:
        """Return fake search results and log the call."""
        self.call_log.append({"keywords": keywords, "max_results": max_results})
        return self._results[:max_results]


class FakeRaisingDDGS:
    """DDGS that always raises, simulating network failure."""

    def text(self, keywords: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Always raise an exception."""
        raise ConnectionError("Simulated network failure")
