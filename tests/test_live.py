"""Live integration tests – only run when GEMINI_API_KEY is available.

Run with: pytest tests/test_live.py -m live -v
"""

from __future__ import annotations

import asyncio
import os

import pytest

from src.agent_workflow import AgentConfig, stream_verification_events
from src.schemas import AgentEvent


pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_live_simple_claim() -> None:
    """Test end-to-end pipeline with a simple factual claim."""
    config = AgentConfig.from_env()
    assert config.api_key, "GEMINI_API_KEY required for live tests"

    text = "The Earth orbits the Sun."
    events: list[AgentEvent] = []
    async for event in stream_verification_events(text, config):
        events.append(event)

    types = [e.type for e in events]
    assert "claims" in types
    assert "done" in types

    # Should have at least one claim_done event
    claim_dones = [e for e in events if e.type == "claim_done"]
    assert len(claim_dones) >= 1


@pytest.mark.asyncio
async def test_live_no_claims() -> None:
    """Test that non-factual input yields zero claims gracefully."""
    config = AgentConfig.from_env()
    assert config.api_key, "GEMINI_API_KEY required for live tests"

    text = "What a beautiful day! I think the weather will be nice tomorrow."
    events: list[AgentEvent] = []
    async for event in stream_verification_events(text, config):
        events.append(event)

    types = [e.type for e in events]
    assert "done" in types
