import json
import logging
import re
import time
from collections import defaultdict
from typing import Any
from urllib.parse import urlparse

from ddgs import DDGS

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
MAX_QUERY_VARIANTS = 3

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
    "github.com",
    "arxiv.org",
    "who.int",
    "un.org",
)

LOW_TRUST_DOMAIN_KEYWORDS = (
    "pinterest.",
    "vk.com",
    "tumblr.com",
)

OFFICIAL_DOMAIN_KEYWORDS = (
    "official",
    "gov",
    "edu",
    "go.jp",
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
    if any(k in domain for k in OFFICIAL_DOMAIN_KEYWORDS):
        return "very_high"
    if any(k in domain for k in TRUSTED_DOMAIN_KEYWORDS):
        return "high"
    return "medium"


def _trust_score(trust_level: str) -> int:
    if trust_level == "very_high":
        return 5
    if trust_level == "high":
        return 3
    if trust_level == "medium":
        return 1
    if trust_level == "low":
        return -3
    return 0


# --- Search Execution ---


def _build_query_variants(query: str) -> list[str]:
    raw = query.strip()
    if not raw:
        return []

    candidates = [
        raw,
        f"{raw} 公式",
        f"{raw} 一次情報",
    ]
    variants: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        variants.append(normalized)
        if len(variants) >= MAX_QUERY_VARIANTS:
            break
    return variants


def _search_once(query: str, max_results: int) -> list[dict[str, Any]]:
    try:
        results = DDGS(timeout=SEARCH_TIMEOUT).text(
            query,
            region=SEARCH_REGION,
            safesearch="moderate",
            max_results=max_results,
        )
    except Exception as e:
        logger.warning("Search failed for query '%s': %s", query, e)
        return []
    return list(results or [])


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

    query_variants = _build_query_variants(query)
    if not query_variants:
        return json.dumps({"error": "検索クエリが空です。"}, ensure_ascii=False)

    per_query_results = max(3, min(max_results, 6))
    aggregated: list[dict[str, Any]] = []
    for variant in query_variants:
        aggregated.extend(_search_once(variant, per_query_results))

    if not aggregated:
        return json.dumps(
            {"results": [], "message": "検索結果が見つかりませんでした。"},
            ensure_ascii=False,
        )

    merged_by_url: dict[str, dict[str, Any]] = {}
    hit_counts: dict[str, int] = defaultdict(int)

    for row in aggregated:
        url = row.get("href", "")
        if not url:
            continue
        hit_counts[url] += 1
        if url in merged_by_url:
            continue

        domain = _extract_domain(url)
        trust_level = _trust_level(domain)
        merged_by_url[url] = {
            "title": _sanitize_text(row.get("title", "")),
            "url": url,
            "domain": domain,
            "trust_level": trust_level,
            "snippet": _sanitize_text(row.get("body", "")),
            "_score": _trust_score(trust_level),
        }

    formatted = list(merged_by_url.values())
    for row in formatted:
        row["_score"] += min(hit_counts[row["url"]], 3) - 1

    formatted.sort(key=lambda x: x["_score"], reverse=True)
    strong_results = [r for r in formatted if r["trust_level"] != "low"]
    weak_results = [r for r in formatted if r["trust_level"] == "low"]
    ranked = strong_results if len(strong_results) >= 3 else strong_results + weak_results

    for row in ranked:
        row.pop("_score", None)

    return json.dumps(
        {
            "query": query,
            "query_variants": query_variants,
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
