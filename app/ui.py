"""Streamlit UI for the Autonomous Research & Fact-Checking Engine.

Drives the async verification pipeline and renders live progress,
streaming markdown, and downloadable reports.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Path isolation: ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from src.agent_workflow import AgentConfig, stream_verification_events
from src.schemas import VERDICT_EMOJI, FactCheckReport

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FactShield AI",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Example text
# ---------------------------------------------------------------------------

EXAMPLE_TEXT = (
    "The Great Wall of China is visible from space with the naked eye. "
    "Albert Einstein failed mathematics in school. "
    "Humans only use 10% of their brains. "
    "Lightning never strikes the same place twice."
)


def _set_example() -> None:
    """Callback to populate the text area with example text."""
    st.session_state["claims_input"] = EXAMPLE_TEXT


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

    st.markdown(f"**Model:** `{model_name}`")

    if api_key:
        st.success("🔑 API Key: configured", icon="✅")
    else:
        st.warning("🔑 API Key: not set", icon="⚠️")
        st.caption("Set `GEMINI_API_KEY` in your `.env` file or environment.")

    max_claims = int(os.environ.get("MAX_CLAIMS", "8"))
    max_parallel = int(os.environ.get("MAX_PARALLEL", "2"))
    search_budget = int(os.environ.get("SEARCH_BUDGET_PER_CLAIM", "6"))

    st.markdown(f"**Max claims:** {max_claims}")
    st.markdown(f"**Parallel workers:** {max_parallel}")
    st.markdown(f"**Search budget/claim:** {search_budget}")

    st.divider()
    st.caption(
        "This tool performs automated fact-checking using AI and web search. "
        "Results should be independently verified for critical decisions."
    )

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("🛡️ Autonomous Research & Fact-Checking Engine")
st.markdown(
    "Paste any passage below. The engine will decompose it into atomic claims, "
    "adversarially research each one, and deliver a grounded audit report."
)

# Input area
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_area(
        "Enter text to fact-check:",
        height=150,
        key="claims_input",
        placeholder="Paste a statement, article excerpt, or social media post here…",
    )
with col2:
    st.button("📝 Load Example", on_click=_set_example, use_container_width=True)

# Audit button
run_button = st.button(
    "Audit & Verify",
    type="primary",
    use_container_width=True,
    disabled=not user_input,
)


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------


def _run_pipeline(text: str) -> None:
    """Execute the async pipeline synchronously in Streamlit."""
    config = AgentConfig.from_env()

    # Metrics for summary
    claims_found: int = 0
    searches_done: int = 0
    claims_done_list: list[dict] = []
    full_report_md: str = ""
    full_report_data: dict | None = None

    # Streaming containers
    status_container = st.status("🔄 Starting verification pipeline…", expanded=True)
    report_placeholder = st.empty()
    streamed_tokens: list[str] = []

    async def _collect_events() -> None:
        nonlocal claims_found, searches_done, full_report_md, full_report_data

        async for event in stream_verification_events(text, config):
            if event.type == "stage":
                status_container.update(label=f"🔄 {event.data}", state="running")
                status_container.write(f"📋 {event.data}")

            elif event.type == "claims":
                claims_found = len(event.data)
                status_container.write(f"📊 Extracted **{claims_found}** atomic claims")

            elif event.type == "search":
                searches_done += 1
                d = event.data
                status_container.write(
                    f"🔍 Claim {d['claim_id']}: "
                    f"`{d['query'][:50]}…` → {d['result_count']} results"
                )

            elif event.type == "claim_done":
                d = event.data
                emoji = VERDICT_EMOJI.get(d.get("verdict", ""), "")
                claims_done_list.append(d)
                status_container.write(
                    f"{emoji} Claim {d['claim_id']}: **{d['verdict']}** "
                    f"({d['confidence']}%)"
                )

            elif event.type == "token":
                streamed_tokens.append(str(event.data))
                report_placeholder.markdown("".join(streamed_tokens))

            elif event.type == "report":
                full_report_md = str(event.data)

            elif event.type == "error":
                status_container.write(f"⚠️ {event.data}")

            elif event.type == "done":
                full_report_data = event.data
                status_container.update(
                    label="✅ Verification complete!", state="complete"
                )

    asyncio.run(_collect_events())

    # Post-run: metrics and downloads
    if claims_done_list:
        st.divider()

        # Verdict tally
        m1, m2, m3, m4 = st.columns(4)
        verdicts = [d.get("verdict", "") for d in claims_done_list]
        m1.metric("✅ Verified", verdicts.count("VERIFIED"))
        m2.metric("❌ False", verdicts.count("FALSE"))
        m3.metric("⚠️ Misleading", verdicts.count("MISLEADING"))
        m4.metric("❓ Unverified", verdicts.count("UNVERIFIED"))

        st.markdown(f"**Total searches performed:** {searches_done}")

        # Render the full markdown report
        if full_report_md:
            with st.expander("📄 Full Report", expanded=False):
                st.markdown(full_report_md)

        # Download buttons
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "📥 Download Report (.md)",
                data=full_report_md or "".join(streamed_tokens),
                file_name="factcheck_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl2:
            json_data = json.dumps(full_report_data, indent=2, ensure_ascii=False) if full_report_data else "{}"
            st.download_button(
                "📥 Download Report (.json)",
                data=json_data,
                file_name="factcheck_report.json",
                mime="application/json",
                use_container_width=True,
            )


if run_button and user_input:
    _run_pipeline(user_input)
