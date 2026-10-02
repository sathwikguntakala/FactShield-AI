"""Tests for src/tools/search_tool.py – JSON formatting, deduplication, error handling."""

from __future__ import annotations

import json
from typing import Any

from src.tools.search_tool import normalize_url, web_search
from tests.conftest import FakeDDGS, FakeRaisingDDGS


# ---------------------------------------------------------------------------
# normalize_url
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    """Tests for URL normalization."""

    def test_strips_fragment(self) -> None:
        assert normalize_url("https://example.com/page#section") == "https://example.com/page"

    def test_strips_trailing_slash(self) -> None:
        assert normalize_url("https://example.com/page/") == "https://example.com/page"

    def test_lowercases_host(self) -> None:
        assert normalize_url("https://Example.COM/Page") == "https://example.com/Page"

    def test_strips_www(self) -> None:
        assert normalize_url("https://www.example.com/page") == "https://example.com/page"

    def test_strips_utm_params(self) -> None:
        result = normalize_url("https://example.com/page?utm_source=test&id=5")
        assert "utm_source" not in result
        assert "id=5" in result

    def test_empty_url(self) -> None:
        assert normalize_url("") == ""

    def test_same_url_different_formats(self) -> None:
        url1 = normalize_url("https://www.Example.COM/page/")
        url2 = normalize_url("https://example.com/page")
        assert url1 == url2


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


class TestWebSearch:
    """Tests for the web_search function."""

    def test_returns_valid_json(self) -> None:
        result = web_search("test query", _ddgs=FakeDDGS())
        data = json.loads(result)
        assert "query" in data
        assert "timestamp" in data
        assert "results" in data
        assert "result_count" in data

    def test_result_structure(self) -> None:
        result = web_search("test query", max_results=2, _ddgs=FakeDDGS())
        data = json.loads(result)
        assert data["result_count"] <= 2
        for r in data["results"]:
            assert "title" in r
            assert "url" in r
            assert "snippet" in r
            assert "domain" in r

    def test_empty_query(self) -> None:
        result = web_search("   ", _ddgs=FakeDDGS())
        data = json.loads(result)
        assert data["result_count"] == 0
        assert "error" in data

    def test_deduplication(self) -> None:
        dupes = FakeDDGS(
            results=[
                {"title": "A", "href": "https://example.com/page", "body": "one"},
                {"title": "B", "href": "https://www.example.com/page/", "body": "two"},
            ]
        )
        result = web_search("test", _ddgs=dupes)
        data = json.loads(result)
        assert data["result_count"] == 1

    def test_max_results_clamping(self) -> None:
        ddgs = FakeDDGS()
        web_search("test", max_results=0, _ddgs=ddgs)
        assert ddgs.call_log[0]["max_results"] == 1

        ddgs2 = FakeDDGS()
        web_search("test", max_results=100, _ddgs=ddgs2)
        assert ddgs2.call_log[0]["max_results"] == 10

    def test_error_handling_no_raise(self) -> None:
        """web_search must never raise, even on connection failure."""
        result = web_search("test query", _ddgs=FakeRaisingDDGS())
        data = json.loads(result)
        assert data["result_count"] == 0
        assert "error" in data

    def test_snippet_truncation(self) -> None:
        long_snippet = "x" * 500
        ddgs = FakeDDGS(
            results=[
                {"title": "A", "href": "https://example.com/long", "body": long_snippet}
            ]
        )
        result = web_search("test", _ddgs=ddgs)
        data = json.loads(result)
        assert len(data["results"][0]["snippet"]) <= 401  # 400 + "…"

    def test_query_length_capping(self) -> None:
        long_query = "a" * 500
        ddgs = FakeDDGS()
        web_search(long_query, _ddgs=ddgs)
        assert len(ddgs.call_log[0]["keywords"]) <= 300
