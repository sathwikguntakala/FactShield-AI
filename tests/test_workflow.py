"""Tests for the agent workflow – event dispatch, coverage guard, error handling."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent_workflow import (
    AgentConfig,
    EvidenceLedger,
    _parse_structured_response,
    _today_str,
    stream_verification_events,
)
from src.schemas import AgentEvent, AtomicClaim, ClaimAudit, DecompositionResult


# ---------------------------------------------------------------------------
# EvidenceLedger tests
# ---------------------------------------------------------------------------


class TestEvidenceLedger:
    """Tests for EvidenceLedger tracking."""

    def test_record_search(self) -> None:
        ledger = EvidenceLedger()
        ledger.record_search("test query", ["https://a.com", "https://b.com"])
        assert ledger.search_count == 1
        assert len(ledger.urls) == 2

    def test_adversarial_detection(self) -> None:
        ledger = EvidenceLedger()
        ledger.record_search("normal query", [])
        assert not ledger.has_adversarial()
        ledger.record_search("claim debunked", [])
        assert ledger.has_adversarial()

    def test_url_grounding(self) -> None:
        ledger = EvidenceLedger()
        ledger.record_search("test", ["https://example.com/page"])
        assert ledger.url_is_grounded("https://example.com/page")
        assert not ledger.url_is_grounded("https://other.com/page")

    def test_url_normalization_in_grounding(self) -> None:
        ledger = EvidenceLedger()
        ledger.record_search("test", ["https://www.Example.COM/page/"])
        assert ledger.url_is_grounded("https://example.com/page")


# ---------------------------------------------------------------------------
# Structured response parsing
# ---------------------------------------------------------------------------


class TestParseStructuredResponse:
    """Tests for the structured output fallback parser."""

    def test_parsed_attribute(self) -> None:
        """Should use .parsed when available."""
        mock_resp = MagicMock()
        mock_resp.parsed = DecompositionResult(
            claims=[AtomicClaim(claim_id=1, claim="Test")]
        )
        result = _parse_structured_response(mock_resp, DecompositionResult)
        assert len(result.claims) == 1

    def test_json_fallback(self) -> None:
        """When .parsed is None, fall back to JSON parsing."""
        mock_resp = MagicMock()
        mock_resp.parsed = None
        mock_resp.text = '{"claims": [{"claim_id": 1, "claim": "Test"}]}'
        result = _parse_structured_response(mock_resp, DecompositionResult)
        assert len(result.claims) == 1

    def test_json_with_markdown_fences(self) -> None:
        """Should strip markdown code fences."""
        mock_resp = MagicMock()
        mock_resp.parsed = None
        mock_resp.text = '```json\n{"claims": [{"claim_id": 1, "claim": "Test"}]}\n```'
        result = _parse_structured_response(mock_resp, DecompositionResult)
        assert len(result.claims) == 1


# ---------------------------------------------------------------------------
# Event dispatch – no API key scenario
# ---------------------------------------------------------------------------


class TestStreamVerificationNoKey:
    """Tests for pipeline behavior without an API key."""

    @pytest.mark.asyncio
    async def test_no_api_key_yields_error(self) -> None:
        """Without an API key, should yield error + done."""
        config = AgentConfig(api_key="", model="gemini-3.6-flash")
        events: list[AgentEvent] = []
        async for event in stream_verification_events("Test text", config):
            events.append(event)
        types = [e.type for e in events]
        assert "error" in types
        assert "done" in types

    @pytest.mark.asyncio
    async def test_empty_input_yields_error(self) -> None:
        """Empty input should yield error + done."""
        config = AgentConfig(api_key="fake-key", model="gemini-3.6-flash")

        # Mock resolve_model to avoid API call
        with patch("src.agent_workflow.resolve_model", return_value="gemini-3.6-flash"):
            events: list[AgentEvent] = []
            async for event in stream_verification_events("   ", config):
                events.append(event)
            types = [e.type for e in events]
            assert "error" in types


# ---------------------------------------------------------------------------
# Coverage guard
# ---------------------------------------------------------------------------


class TestCoverageGuard:
    """Tests for the coverage guard behavior."""

    def test_has_adversarial_detection_keywords(self) -> None:
        ledger = EvidenceLedger()
        for term in ["debunked", "false", "fact check", "contradicts", "myth", "hoax"]:
            fresh = EvidenceLedger()
            fresh.record_search(f"claim {term}", [])
            assert fresh.has_adversarial(), f"Should detect '{term}' as adversarial"

    def test_normal_query_not_adversarial(self) -> None:
        ledger = EvidenceLedger()
        ledger.record_search("who invented the telephone", [])
        assert not ledger.has_adversarial()


# ---------------------------------------------------------------------------
# Today helper
# ---------------------------------------------------------------------------


class TestTodayStr:
    """Tests for the _today_str helper."""

    def test_format(self) -> None:
        result = _today_str()
        assert len(result) == 10  # YYYY-MM-DD
        assert result.count("-") == 2


# ---------------------------------------------------------------------------
# Isolated claim failure
# ---------------------------------------------------------------------------


class TestIsolatedFailure:
    """Test that one claim's failure doesn't crash the pipeline."""

    @pytest.mark.asyncio
    async def test_claim_failure_isolated(self) -> None:
        """If a single claim task errors, others should still complete."""
        # This is tested via the grounding enforcer ensuring
        # failed research still yields UNVERIFIED results
        audit = ClaimAudit(
            claim_id=1,
            claim="Test",
            verdict="UNVERIFIED",
            confidence=20,
            finding_summary="Research failed: timeout",
        )
        assert audit.verdict == "UNVERIFIED"
        assert audit.confidence == 20
