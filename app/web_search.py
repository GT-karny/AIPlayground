import json
import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

from ddgs import DDGS
from ddgs.exceptions import DDGSException

logger = logging.getLogger(__name__)

# --- Tool Schema (OpenAI function calling format) ---

SEARCH_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. "
            "Use this when the user asks about recent events, facts you are unsure about, "
            "or anything that requires up-to-date information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-12)",
                    "minimum": 1,
                    "maximum": 12,
                },
            },
            "required": ["query"],
        },
    },
}

TOOLS: list[dict[str, Any]] = [SEARCH_TOOL_DEFINITION]

# --- Rate Limiting ---

MIN_SEARCH_INTERVAL = 0.5  # seconds
_last_search_time: float = 0.0

# --- Constants ---

DEFAULT_MAX_RESULTS = 8
MAX_RESULTS_LIMIT = 12
MAX_SNIPPET_LENGTH = 300
SEARCH_TIMEOUT = 10
SEARCH_REGION = "jp-jp"

TRUSTED_DOMAIN_KEYWORDS = (
    "wikipedia.org",
    "go.jp",
    "ac.jp",
    "or.jp",
    "co.jp",
    "gov",
    "edu",
    "nhk.or.jp",
    "imdb.com",
    "britannica.com",
)

LOW_TRUST_DOMAIN_KEYWORDS = (
    "pinterest.",
    "vk.com",
    "tumblr.com",
)


# --- Sanitization ---


def _sanitize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    if len(text) > MAX_SNIPPET_LENGTH:
        text = text[:MAX_SNIPPET_LENGTH] + "..."
    return text.strip()


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _trust_level(domain: str) -> str:
    if not domain:
        return "unknown"
    if any(k in domain for k in LOW_TRUST_DOMAIN_KEYWORDS):
        return "low"
    if any(k in domain for k in TRUSTED_DOMAIN_KEYWORDS):
        return "high"
    return "medium"


def _trust_score(trust_level: str) -> int:
    if trust_level == "high":
        return 3
    if trust_level == "medium":
        return 1
    if trust_level == "low":
        return -3
    return 0


# --- Search Execution ---


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    global _last_search_time

    max_results = max(1, min(max_results, MAX_RESULTS_LIMIT))

    now = time.monotonic()
    elapsed = now - _last_search_time
    if _last_search_time > 0 and elapsed < MIN_SEARCH_INTERVAL:
        return json.dumps(
            {"error": f"検索間隔が短すぎます。{MIN_SEARCH_INTERVAL}秒以上空けてください。"},
            ensure_ascii=False,
        )
    _last_search_time = now

    try:
        results = DDGS(timeout=SEARCH_TIMEOUT).text(
            query,
            region=SEARCH_REGION,
            safesearch="moderate",
            max_results=max_results,
        )
    except DDGSException as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return json.dumps({"error": f"検索に失敗しました: {e}"}, ensure_ascii=False)
    except Exception as e:
        logger.warning("Unexpected search error: %s", e)
        return json.dumps({"error": f"検索エラー: {e}"}, ensure_ascii=False)

    if not results:
        return json.dumps(
            {"results": [], "message": "検索結果が見つかりませんでした。"},
            ensure_ascii=False,
        )

    formatted: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for r in results:
        url = r.get("href", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        domain = _extract_domain(url)
        trust_level = _trust_level(domain)
        formatted.append(
            {
                "title": _sanitize_text(r.get("title", "")),
                "url": url,
                "domain": domain,
                "trust_level": trust_level,
                "snippet": _sanitize_text(r.get("body", "")),
                "_score": _trust_score(trust_level),
            }
        )

    formatted.sort(key=lambda x: x["_score"], reverse=True)
    strong_results = [r for r in formatted if r["trust_level"] != "low"]
    weak_results = [r for r in formatted if r["trust_level"] == "low"]
    ranked = strong_results if len(strong_results) >= 3 else strong_results + weak_results

    for r in ranked:
        r.pop("_score", None)

    return json.dumps(
        {
            "query": query,
            "results": ranked[:max_results],
        },
        ensure_ascii=False,
    )


# --- Tool Dispatch ---

TOOL_FUNCTIONS = {
    "web_search": web_search,
}


def execute_tool_call(function_name: str, arguments: dict) -> str:
    func = TOOL_FUNCTIONS.get(function_name)
    if func is None:
        raise ValueError(f"Unknown tool function: {function_name}")
    try:
        return func(**arguments)
    except TypeError as e:
        raise ValueError(f"Invalid tool arguments: {e}") from e
