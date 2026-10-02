"""Pydantic v2 schemas for the fact-checking pipeline.

All domain types are defined here: claims, verdicts, audit records,
reports and streaming events.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

Verdict = Literal["VERIFIED", "FALSE", "MISLEADING", "UNVERIFIED"]

VERDICT_EMOJI: dict[str, str] = {
    "VERIFIED": "✅",
    "FALSE": "❌",
    "MISLEADING": "⚠️",
    "UNVERIFIED": "❓",
}


class AtomicClaim(BaseModel):
    """A single, independently verifiable factual assertion."""

    model_config = ConfigDict(str_strip_whitespace=True)

    claim_id: int = Field(ge=1, description="1-based ordinal identifier")
    claim: str = Field(min_length=1, description="Self-contained factual claim")


class DecompositionResult(BaseModel):
    """Output of the Decomposer agent."""

    model_config = ConfigDict(str_strip_whitespace=True)

    claims: list[AtomicClaim] = Field(default_factory=list)


class ClaimAudit(BaseModel):
    """Adjudication result for a single claim."""

    model_config = ConfigDict(str_strip_whitespace=True)

    claim_id: int = Field(ge=1)
    claim: str
    verdict: Verdict
    confidence: int = Field(ge=0, le=100)
    finding_summary: str = ""
    analysis: str = ""
    primary_sources: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


class FactCheckReport(BaseModel):
    """Complete report encompassing all audited claims."""

    model_config = ConfigDict(str_strip_whitespace=True)

    original_statement: str
    executive_summary: str = ""
    audited_claims: list[ClaimAudit] = Field(default_factory=list)
    final_takeaway: str = ""

    # ------------------------------------------------------------------
    # Deterministic Markdown generation
    # ------------------------------------------------------------------

    @staticmethod
    def _escape_md(text: str) -> str:
        """Escape pipe characters for Markdown table cells."""
        return html.escape(text).replace("|", "&#124;").replace("\n", " ")

    def generate_claims_table(self) -> str:
        """Build a Markdown table of verdicts – pure code, no LLM."""
        if not self.audited_claims:
            return "_No claims were extracted._\n"

        header = "| # | Claim | Verdict | Confidence | Sources |\n"
        sep = "|---|-------|---------|------------|---------|\n"
        rows: list[str] = []
        for ca in self.audited_claims:
            emoji = VERDICT_EMOJI.get(ca.verdict, "")
            sources_count = len(ca.primary_sources)
            contra_count = len(ca.contradictions)
            src_text = f"{sources_count} src"
            if contra_count:
                src_text += f", {contra_count} contra"
            claim_text = self._escape_md(
                ca.claim[:120] + ("…" if len(ca.claim) > 120 else "")
            )
            rows.append(
                f"| {ca.claim_id} | {claim_text} "
                f"| {emoji} {ca.verdict} | {ca.confidence}% | {src_text} |"
            )
        return header + sep + "\n".join(rows) + "\n"

    def generate_detailed_breakdown(self) -> str:
        """Render per-claim detail blocks – pure code, no LLM."""
        if not self.audited_claims:
            return ""

        parts: list[str] = []
        for ca in self.audited_claims:
            emoji = VERDICT_EMOJI.get(ca.verdict, "")
            block = f"### Claim {ca.claim_id}: {self._escape_md(ca.claim)}\n\n"
            block += f"**Verdict:** {emoji} {ca.verdict} ({ca.confidence}% confidence)\n\n"
            if ca.finding_summary:
                block += f"**Summary:** {ca.finding_summary}\n\n"
            if ca.analysis:
                block += f"**Analysis:** {ca.analysis}\n\n"
            if ca.primary_sources:
                block += "**Sources:**\n"
                for url in ca.primary_sources:
                    block += f"- {url}\n"
                block += "\n"
            if ca.contradictions:
                block += "**Contradicting sources:**\n"
                for url in ca.contradictions:
                    block += f"- {url}\n"
                block += "\n"
            parts.append(block)
        return "\n".join(parts)

    def to_markdown(self) -> str:
        """Full Markdown report suitable for display or download."""
        md = "# 🛡️ Fact-Check Report\n\n"
        if self.executive_summary:
            md += f"## Executive Summary\n\n{self.executive_summary}\n\n"
        md += "## Atomic Claims Audit Table\n\n"
        md += self.generate_claims_table()
        md += "\n## Detailed Breakdown\n\n"
        md += self.generate_detailed_breakdown()
        if self.final_takeaway:
            md += f"## Final Takeaway\n\n{self.final_takeaway}\n\n"
        md += (
            f"\n---\n_Report generated at "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n"
        )
        return md


# ---------------------------------------------------------------------------
# Streaming event
# ---------------------------------------------------------------------------


class AgentEvent(BaseModel):
    """Single event emitted by the verification pipeline.

    Types: stage, claims, search, claim_done, token, report, error, done.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    type: str
    data: Any = None
