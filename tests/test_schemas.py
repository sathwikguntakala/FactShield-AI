"""Tests for src/schemas.py – validation bounds, markdown generation, escaping."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    AgentEvent,
    AtomicClaim,
    ClaimAudit,
    DecompositionResult,
    FactCheckReport,
)


# ---------------------------------------------------------------------------
# AtomicClaim validation
# ---------------------------------------------------------------------------


class TestAtomicClaim:
    """Tests for AtomicClaim validation rules."""

    def test_valid_claim(self) -> None:
        c = AtomicClaim(claim_id=1, claim="The sky is blue.")
        assert c.claim_id == 1
        assert c.claim == "The sky is blue."

    def test_claim_id_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            AtomicClaim(claim_id=0, claim="Invalid ID")

    def test_claim_id_negative(self) -> None:
        with pytest.raises(ValidationError):
            AtomicClaim(claim_id=-1, claim="Negative ID")

    def test_empty_claim_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AtomicClaim(claim_id=1, claim="")

    def test_whitespace_stripping(self) -> None:
        c = AtomicClaim(claim_id=1, claim="  hello world  ")
        assert c.claim == "hello world"


# ---------------------------------------------------------------------------
# ClaimAudit validation
# ---------------------------------------------------------------------------


class TestClaimAudit:
    """Tests for ClaimAudit validation rules."""

    def test_valid_verdicts(self) -> None:
        for verdict in ("VERIFIED", "FALSE", "MISLEADING", "UNVERIFIED"):
            ca = ClaimAudit(
                claim_id=1, claim="test", verdict=verdict, confidence=50
            )
            assert ca.verdict == verdict

    def test_invalid_verdict(self) -> None:
        with pytest.raises(ValidationError):
            ClaimAudit(claim_id=1, claim="test", verdict="MAYBE", confidence=50)

    def test_confidence_bounds(self) -> None:
        # Lower bound
        ca = ClaimAudit(claim_id=1, claim="test", verdict="VERIFIED", confidence=0)
        assert ca.confidence == 0

        # Upper bound
        ca = ClaimAudit(claim_id=1, claim="test", verdict="VERIFIED", confidence=100)
        assert ca.confidence == 100

    def test_confidence_too_high(self) -> None:
        with pytest.raises(ValidationError):
            ClaimAudit(claim_id=1, claim="test", verdict="VERIFIED", confidence=101)

    def test_confidence_too_low(self) -> None:
        with pytest.raises(ValidationError):
            ClaimAudit(claim_id=1, claim="test", verdict="VERIFIED", confidence=-1)


# ---------------------------------------------------------------------------
# FactCheckReport markdown generation
# ---------------------------------------------------------------------------


class TestFactCheckReport:
    """Tests for deterministic markdown generation."""

    def test_empty_claims_table(self) -> None:
        report = FactCheckReport(original_statement="test")
        table = report.generate_claims_table()
        assert "No claims" in table

    def test_claims_table_content(self) -> None:
        report = FactCheckReport(
            original_statement="test",
            audited_claims=[
                ClaimAudit(
                    claim_id=1,
                    claim="Earth orbits the Sun",
                    verdict="VERIFIED",
                    confidence=90,
                    primary_sources=["https://nasa.gov/earth"],
                ),
            ],
        )
        table = report.generate_claims_table()
        assert "VERIFIED" in table
        assert "90%" in table
        assert "Earth orbits the Sun" in table

    def test_pipe_escaping_in_table(self) -> None:
        """Pipe characters in claims must be escaped for valid Markdown tables."""
        report = FactCheckReport(
            original_statement="test",
            audited_claims=[
                ClaimAudit(
                    claim_id=1,
                    claim="A|B means A or B",
                    verdict="VERIFIED",
                    confidence=80,
                ),
            ],
        )
        table = report.generate_claims_table()
        # Raw pipe should not appear inside the claim column
        lines = table.strip().split("\n")
        data_line = lines[-1]  # Last row is our data row
        # The claim text should have escaped pipe
        assert "A|B" not in data_line or "&#124;" in data_line

    def test_to_markdown_includes_sections(self) -> None:
        report = FactCheckReport(
            original_statement="test statement",
            executive_summary="This is a summary.",
            audited_claims=[
                ClaimAudit(
                    claim_id=1,
                    claim="Test claim",
                    verdict="UNVERIFIED",
                    confidence=30,
                    finding_summary="Not enough evidence.",
                    analysis="Searched but found nothing.",
                ),
            ],
            final_takeaway="Be cautious.",
        )
        md = report.to_markdown()
        assert "Executive Summary" in md
        assert "Atomic Claims Audit Table" in md
        assert "Detailed Breakdown" in md
        assert "Final Takeaway" in md
        assert "Be cautious." in md

    def test_detailed_breakdown_urls(self) -> None:
        report = FactCheckReport(
            original_statement="test",
            audited_claims=[
                ClaimAudit(
                    claim_id=1,
                    claim="Test",
                    verdict="FALSE",
                    confidence=85,
                    primary_sources=["https://source.com"],
                    contradictions=["https://contra.com"],
                ),
            ],
        )
        breakdown = report.generate_detailed_breakdown()
        assert "https://source.com" in breakdown
        assert "https://contra.com" in breakdown


# ---------------------------------------------------------------------------
# DecompositionResult
# ---------------------------------------------------------------------------


class TestDecompositionResult:
    """Tests for DecompositionResult."""

    def test_empty_claims_ok(self) -> None:
        dr = DecompositionResult(claims=[])
        assert dr.claims == []

    def test_with_claims(self) -> None:
        dr = DecompositionResult(
            claims=[AtomicClaim(claim_id=1, claim="Test")]
        )
        assert len(dr.claims) == 1


# ---------------------------------------------------------------------------
# AgentEvent
# ---------------------------------------------------------------------------


class TestAgentEvent:
    """Tests for AgentEvent."""

    def test_basic_event(self) -> None:
        e = AgentEvent(type="stage", data="Starting")
        assert e.type == "stage"
        assert e.data == "Starting"

    def test_event_with_dict_data(self) -> None:
        e = AgentEvent(type="search", data={"query": "test", "count": 3})
        assert e.data["count"] == 3
