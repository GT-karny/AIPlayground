import json
import logging
import re
import time
from typing import Any

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
            },
            "required": ["query"],
        },
    },
}

TOOLS: list[dict[str, Any]] = [SEARCH_TOOL_DEFINITION]

# --- Rate Limiting ---

MIN_SEARCH_INTERVAL = 2.0  # seconds
_last_search_time: float = 0.0

# --- Constants ---

DEFAULT_MAX_RESULTS = 5
MAX_SNIPPET_LENGTH = 200
SEARCH_TIMEOUT = 10


# --- Sanitization ---


def _sanitize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    if len(text) > MAX_SNIPPET_LENGTH:
        text = text[:MAX_SNIPPET_LENGTH] + "..."
    return text.strip()


# --- Search Execution ---


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    global _last_search_time

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
            region="wt-wt",
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

    formatted = []
    for r in results:
        formatted.append(
            {
                "title": _sanitize_text(r.get("title", "")),
                "url": r.get("href", ""),
                "snippet": _sanitize_text(r.get("body", "")),
            }
        )

    return json.dumps({"results": formatted}, ensure_ascii=False)


# --- Tool Dispatch ---

TOOL_FUNCTIONS = {
    "web_search": web_search,
}


def execute_tool_call(function_name: str, arguments: dict) -> str:
    func = TOOL_FUNCTIONS.get(function_name)
    if func is None:
        raise ValueError(f"Unknown tool function: {function_name}")
    return func(**arguments)
