"""Tests for the grounding enforcer logic in agent_workflow.py."""

from __future__ import annotations

from src.agent_workflow import EvidenceLedger, enforce_grounding
from src.schemas import ClaimAudit


def _make_ledger(queries: list[str] | None = None, urls: list[str] | None = None) -> EvidenceLedger:
    """Helper to build a pre-populated ledger."""
    ledger = EvidenceLedger()
    for q in queries or []:
        ledger.record_search(q, urls or [])
    return ledger


# ---------------------------------------------------------------------------
# Grounding strip tests
# ---------------------------------------------------------------------------


class TestGroundingEnforcer:
    """Tests for the enforce_grounding function."""

    def test_strips_ungrounded_source_urls(self) -> None:
        """URLs not in the ledger must be removed from primary_sources."""
        ledger = _make_ledger(
            queries=["test", "fact check", "official"],
            urls=["https://real.com/page"],
        )
        audit = ClaimAudit(
            claim_id=1,
            claim="Test claim",
            verdict="VERIFIED",
            confidence=90,
            primary_sources=["https://real.com/page", "https://fake.com/hallucinated"],
        )
        result = enforce_grounding(audit, ledger)
        assert "https://real.com/page" in result.primary_sources
        assert "https://fake.com/hallucinated" not in result.primary_sources
        assert "Auditor note" in result.analysis

    def test_strips_ungrounded_contradiction_urls(self) -> None:
        ledger = _make_ledger(
            queries=["test", "debunked", "source"],
            urls=["https://grounded.org/contra"],
        )
        audit = ClaimAudit(
            claim_id=1,
            claim="Test",
            verdict="FALSE",
            confidence=80,
            contradictions=["https://grounded.org/contra", "https://invented.com"],
        )
        result = enforce_grounding(audit, ledger)
        assert len(result.contradictions) == 1
        assert "https://grounded.org/contra" in result.contradictions

    def test_no_searches_downgrades_to_unverified(self) -> None:
        """No searches performed → UNVERIFIED, confidence ≤30."""
        ledger = EvidenceLedger()  # No searches
        audit = ClaimAudit(
            claim_id=1,
            claim="Test",
            verdict="VERIFIED",
            confidence=95,
        )
        result = enforce_grounding(audit, ledger)
        assert result.verdict == "UNVERIFIED"
        assert result.confidence <= 30

    def test_verified_needs_two_domains(self) -> None:
        """VERIFIED requires ≥2 grounded sources across ≥2 domains."""
        ledger = _make_ledger(
            queries=["test", "fact check", "official"],
            urls=["https://example.com/a"],
        )
        audit = ClaimAudit(
            claim_id=1,
            claim="Test",
            verdict="VERIFIED",
            confidence=90,
            primary_sources=["https://example.com/a"],
        )
        result = enforce_grounding(audit, ledger)
        # Only 1 source from 1 domain → downgrade
        assert result.verdict == "UNVERIFIED"
        assert result.confidence <= 50

    def test_verified_with_two_domains_passes(self) -> None:
        """VERIFIED with 2+ sources across 2+ domains should pass."""
        ledger = EvidenceLedger()
        ledger.record_search("test", ["https://source-a.com/page"])
        ledger.record_search("fact check", ["https://source-b.org/page"])
        ledger.record_search("official", [])

        audit = ClaimAudit(
            claim_id=1,
            claim="Test",
            verdict="VERIFIED",
            confidence=85,
            primary_sources=["https://source-a.com/page", "https://source-b.org/page"],
        )
        result = enforce_grounding(audit, ledger)
        assert result.verdict == "VERIFIED"
        assert result.confidence == 85

    def test_false_without_sources_downgrades(self) -> None:
        """FALSE without any grounded sources → UNVERIFIED."""
        ledger = _make_ledger(
            queries=["test", "debunked", "official"],
            urls=[],
        )
        audit = ClaimAudit(
            claim_id=1,
            claim="Test",
            verdict="FALSE",
            confidence=80,
        )
        result = enforce_grounding(audit, ledger)
        assert result.verdict == "UNVERIFIED"
        assert result.confidence <= 50

    def test_misleading_without_sources_downgrades(self) -> None:
        """MISLEADING without grounded sources → UNVERIFIED."""
        ledger = _make_ledger(
            queries=["test", "debunked", "official"],
            urls=[],
        )
        audit = ClaimAudit(
            claim_id=1,
            claim="Test",
            verdict="MISLEADING",
            confidence=70,
        )
        result = enforce_grounding(audit, ledger)
        assert result.verdict == "UNVERIFIED"

    def test_confidence_threshold_enforcement(self) -> None:
        """Downgraded confidence must respect the threshold."""
        ledger = EvidenceLedger()  # zero searches
        audit = ClaimAudit(
            claim_id=1,
            claim="Test",
            verdict="VERIFIED",
            confidence=99,
        )
        result = enforce_grounding(audit, ledger)
        assert result.confidence <= 30

    def test_clean_audit_passes_through(self) -> None:
        """A properly grounded audit should pass through unchanged."""
        ledger = EvidenceLedger()
        ledger.record_search("test query", ["https://a.com/1", "https://b.org/2"])
        ledger.record_search("debunked test", ["https://c.edu/3"])
        ledger.record_search("official", [])

        audit = ClaimAudit(
            claim_id=1,
            claim="Test",
            verdict="VERIFIED",
            confidence=80,
            finding_summary="Found evidence.",
            analysis="Multiple sources agree.",
            primary_sources=["https://a.com/1", "https://b.org/2"],
        )
        result = enforce_grounding(audit, ledger)
        assert result.verdict == "VERIFIED"
        assert result.confidence == 80
        assert len(result.primary_sources) == 2
