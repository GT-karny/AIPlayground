import json
import logging
from collections import OrderedDict
from typing import Any

from openai import OpenAI

from app import config

logger = logging.getLogger(__name__)

ALLOWED_MODES = {
    "chat",
    "factual_balanced",
    "factual_strict",
    "needs_clarification",
}

ROUTER_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are an intent router for a Japanese assistant.\n"
        "Classify the latest user message using recent conversation context.\n"
        "Modes:\n"
        "- chat\n"
        "- factual_balanced\n"
        "- factual_strict\n"
        "- needs_clarification\n"
        "If the latest message is short or elliptical (e.g. 主人公誰？), resolve it with context.\n"
        "Return ONLY valid JSON in this schema:\n"
        "{"
        '"mode":"chat|factual_balanced|factual_strict|needs_clarification",'
        '"confidence":0.0,'
        '"reason":"short Japanese reason",'
        '"clarification_prompt":"2-3 choice Japanese question or empty string",'
        '"rewritten_user_message":"standalone Japanese user intent sentence"'
        "}\n"
        "No markdown. No extra keys."
    ),
}


class IntentRouter:
    def __init__(self, client: OpenAI):
        self.client = client
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_limit = 128

    def _remember(self, key: str, value: dict[str, Any]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            raise ValueError("empty router response")

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start < 0 or end <= start:
                raise
            return json.loads(raw[start : end + 1])

    @staticmethod
    def _normalize(data: dict[str, Any], user_message: str) -> dict[str, Any]:
        mode = str(data.get("mode", "")).strip()
        if mode not in ALLOWED_MODES:
            raise ValueError(f"invalid mode: {mode}")

        confidence_raw = data.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError) as e:
            raise ValueError("invalid confidence") from e
        confidence = min(1.0, max(0.0, confidence))

        reason = str(data.get("reason", "")).strip()
        clarification_prompt = str(data.get("clarification_prompt", "")).strip()
        rewritten = str(data.get("rewritten_user_message", "")).strip() or user_message.strip()

        return {
            "mode": mode,
            "confidence": confidence,
            "reason": reason,
            "clarification_prompt": clarification_prompt,
            "rewritten_user_message": rewritten,
        }

    @staticmethod
    def _fallback(user_message: str) -> dict[str, Any]:
        text = (user_message or "").strip()
        if not text:
            return {
                "mode": "needs_clarification",
                "confidence": 0.4,
                "reason": "入力が空",
                "clarification_prompt": (
                    "もう少し詳しく教えてください。どう進めますか？\n"
                    "1. 要点だけ知りたい\n"
                    "2. 背景から詳しく知りたい\n"
                    "3. まず何を決めるべきか知りたい"
                ),
                "rewritten_user_message": "",
            }
        if len(text) <= 4:
            return {
                "mode": "needs_clarification",
                "confidence": 0.45,
                "reason": "入力が短すぎる",
                "clarification_prompt": (
                    "意図を確認したいです。どの形で進めますか？\n"
                    "1. 質問内容を具体化する\n"
                    "2. こちらで候補を出して選ぶ\n"
                    "3. まず概要だけ聞く"
                ),
                "rewritten_user_message": text,
            }
        return {
            "mode": "factual_balanced",
            "confidence": 0.35,
            "reason": "ルーター失敗時の既定",
            "clarification_prompt": "",
            "rewritten_user_message": text,
        }

    @staticmethod
    def _build_router_context(history: list[dict[str, str]], user_message: str) -> str:
        lines = []
        for row in history[-6:]:
            role = row.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = (row.get("content", "") or "").strip().replace("\n", " ")
            if not content:
                continue
            lines.append(f"{role}: {content[:220]}")
        lines.append(f"user: {(user_message or '').strip()}")
        return "\n".join(lines)

    def classify(self, user_message: str, history: list[dict[str, str]]) -> dict[str, Any]:
        cache_key = f"{(user_message or '').strip()}||{len(history)}"
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        router_context = self._build_router_context(history, user_message)
        messages = [
            ROUTER_SYSTEM_MESSAGE,
            {"role": "user", "content": router_context},
        ]

        try:
            response = self.client.chat.completions.create(
                model=config.ROUTER_MODEL_NAME,
                messages=messages,
                temperature=config.ROUTER_TEMPERATURE,
                max_tokens=config.ROUTER_MAX_TOKENS,
            )
            choices = getattr(response, "choices", None)
            if not choices:
                raise RuntimeError("router returned no choices")
            content = choices[0].message.content or ""
            parsed = self._extract_json(content)
            result = self._normalize(parsed, user_message)
        except Exception as e:
            logger.warning("Intent router failed: %s", e)
            result = self._fallback(user_message)

        self._remember(cache_key, result)
        return result
