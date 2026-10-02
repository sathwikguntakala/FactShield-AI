"""Web search tool using DuckDuckGo (ddgs).

Synchronous, thread-safe, never raises. Always returns valid JSON text.
External dependency (DDGS) is injectable for offline testing.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urldefrag, urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_INTERVAL_SECONDS: float = 0.25
MAX_QUERY_LENGTH: int = 300
MAX_SNIPPET_LENGTH: int = 400
MIN_RESULTS: int = 1
MAX_RESULTS: int = 10
RATE_LIMIT_RETRIES: int = 3
RATE_LIMIT_BACKOFF: float = 2.0

# ---------------------------------------------------------------------------
# DDGS protocol for dependency injection
# ---------------------------------------------------------------------------


class DDGSProtocol(Protocol):
    """Minimal interface required from a DDGS-like object."""

    def text(self, keywords: str, max_results: int, **kwargs: Any) -> list[dict[str, Any]]:
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication.

    Strips fragment, trailing slash, lowercases scheme/host,
    removes www. prefix, drops common tracking params.
    """
    if not url:
        return ""
    url = url.strip()
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/") or ""
    # Drop common tracking query params
    query = parsed.query
    if query:
        params = [
            p
            for p in query.split("&")
            if not re.match(r"^(utm_|fbclid|gclid|ref=)", p, re.IGNORECASE)
        ]
        query = "&".join(params)
    result = f"{scheme}://{host}{path}"
    if query:
        result += f"?{query}"
    return result


def _clean_query(query: str) -> str:
    """Strip whitespace and cap at MAX_QUERY_LENGTH chars."""
    cleaned = query.strip()
    if len(cleaned) > MAX_QUERY_LENGTH:
        cleaned = cleaned[:MAX_QUERY_LENGTH]
    return cleaned


def _clamp_results(n: int) -> int:
    """Clamp max_results to [MIN_RESULTS, MAX_RESULTS]."""
    return max(MIN_RESULTS, min(n, MAX_RESULTS))


def _get_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        host = urlparse(url).hostname or ""
        host = host.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _truncate_snippet(snippet: str) -> str:
    """Truncate snippet to MAX_SNIPPET_LENGTH."""
    if len(snippet) > MAX_SNIPPET_LENGTH:
        return snippet[:MAX_SNIPPET_LENGTH] + "…"
    return snippet


# ---------------------------------------------------------------------------
# Thread-safe search with pacing
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_last_call_time: float = 0.0


def _get_default_ddgs() -> DDGSProtocol:
    """Import and return a DDGS instance. Handles package rename."""
    try:
        from ddgs import DDGS  # type: ignore[import-untyped]
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore[import-untyped]
    return DDGS()


def web_search(
    query: str,
    max_results: int = 4,
    *,
    _ddgs: DDGSProtocol | None = None,
) -> str:
    """Search the web via DuckDuckGo. Returns valid JSON text, never raises.

    Args:
        query: Search query string.
        max_results: Number of results to request (clamped 1-10).
        _ddgs: Injectable DDGS-like object for testing.

    Returns:
        JSON string with keys: query, timestamp, results, result_count, error.
    """
    global _last_call_time

    cleaned_query = _clean_query(query)
    clamped_max = _clamp_results(max_results)
    timestamp = datetime.now(timezone.utc).isoformat()

    if not cleaned_query:
        return json.dumps(
            {
                "query": query,
                "timestamp": timestamp,
                "results": [],
                "result_count": 0,
                "error": "Empty query after cleaning",
            }
        )

    # Enforce minimum interval between calls (thread-safe)
    with _lock:
        now = time.monotonic()
        elapsed = now - _last_call_time
        if elapsed < MIN_INTERVAL_SECONDS:
            time.sleep(MIN_INTERVAL_SECONDS - elapsed)
        _last_call_time = time.monotonic()

    ddgs_client = _ddgs or _get_default_ddgs()

    raw_results: list[dict[str, Any]] = []
    last_error: str | None = None

    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            raw_results = ddgs_client.text(cleaned_query, max_results=clamped_max)
            last_error = None
            break
        except Exception as exc:
            last_error = f"Search attempt {attempt + 1} failed: {exc}"
            logger.warning(last_error)
            if attempt < RATE_LIMIT_RETRIES - 1:
                time.sleep(RATE_LIMIT_BACKOFF * (attempt + 1))

    # Deduplicate by normalized URL
    seen_urls: set[str] = set()
    results: list[dict[str, str]] = []

    for item in raw_results or []:
        url = item.get("href") or item.get("url") or item.get("link") or ""
        norm = normalize_url(url)
        if not norm or norm in seen_urls:
            continue
        seen_urls.add(norm)
        results.append(
            {
                "title": str(item.get("title", "")),
                "url": url,
                "snippet": _truncate_snippet(str(item.get("body") or item.get("snippet") or "")),
                "domain": _get_domain(url),
            }
        )

    payload: dict[str, Any] = {
        "query": cleaned_query,
        "timestamp": timestamp,
        "results": results,
        "result_count": len(results),
    }
    if last_error and not results:
        payload["error"] = last_error

    return json.dumps(payload, ensure_ascii=False)
