import json
import logging
import re
from typing import Optional

from openai import OpenAI

from app import config
from app.intent_router import IntentRouter
from app.web_search import TOOLS, execute_tool_call

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 3
URL_PATTERN = re.compile(r"https?://[^\s)>\"]+")
REPEAT_BLOCK_PATTERN = re.compile(r"(.{8,40})\1{2,}", re.DOTALL)

CHAT_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "あなたは日本語で会話するアシスタントです。"
        "話し言葉で、短く自然なテンポで返答してください。"
        "雑談では堅い定型文を避け、相手に合わせた自然な返しを優先してください。"
        "必要ならweb_searchツールを使えますが、不要なときは使わなくて構いません。"
        "質問が曖昧なときは、先に2〜3個の選択肢を提示して確認してください。"
    ),
}

FACTUAL_MODE_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "この質問は事実確認モードです。まずweb_searchで確認し、"
        "主要な主張ごとに出典URLを示してください。"
        "根拠が足りない場合は断定せず、不明と明記してください。"
    ),
}

BALANCED_FACTUAL_MODE_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "この質問は事実寄りです。必要に応じてweb_searchを使って確認してください。"
        "回答を止めすぎず、まずは分かる範囲で簡潔に答え、"
        "不確かな点だけ『未確認』として示してください。"
        "URLが取れた場合は、可能な範囲で出典を添えてください。"
    ),
}

DEFAULT_CLARIFYING_QUESTION = (
    "確認させてください。どの形で進めますか？\n"
    "1. ざっくり要点だけ\n"
    "2. 具体例つきで詳しく\n"
    "3. 先に前提条件を整理してから"
)


class ChatClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=config.BASE_URL,
            api_key=config.API_KEY,
        )
        self.router = IntentRouter(self.client)
        self.messages = [CHAT_SYSTEM_MESSAGE]

    @staticmethod
    def _extract_message(response):
        error = getattr(response, "error", None)
        if error:
            if isinstance(error, dict):
                msg = error.get("message", str(error))
            else:
                msg = str(error)
            raise RuntimeError(f"LLM API error: {msg}")

        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("LLM API returned no choices.")

        return choices[0].message

    @staticmethod
    def _is_tooling_error(error_message: str) -> bool:
        msg = error_message.lower()
        keywords = [
            "tool",
            "toolparser",
            "tool-call-parser",
            "tool choice",
            "tools",
            "json_invalid",
            "validation error",
            "pydantic",
            "badrequesterror",
        ]
        return any(k in msg for k in keywords)

    @staticmethod
    def _has_citation(text: str) -> bool:
        return bool(URL_PATTERN.search(text or ""))

    @staticmethod
    def _recent_dialogue_for_router(messages: list[dict]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for row in messages:
            role = row.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = (row.get("content", "") or "").strip()
            if not content:
                continue
            rows.append({"role": role, "content": content})
        return rows[-8:]

    @staticmethod
    def _is_broken_output(text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return True
        if len(raw) > 1800:
            return True
        if REPEAT_BLOCK_PATTERN.search(raw):
            return True

        sentences = [s.strip() for s in re.split(r"[。！？\n]+", raw) if s.strip()]
        if not sentences:
            return True
        freq: dict[str, int] = {}
        for sentence in sentences:
            if len(sentence) < 6:
                continue
            freq[sentence] = freq.get(sentence, 0) + 1
            if freq[sentence] >= 4:
                return True
        return False

    @staticmethod
    def _response_max_tokens(factual_mode: bool) -> int:
        return config.FACTUAL_MAX_TOKENS if factual_mode else config.CHAT_MAX_TOKENS

    def _repair_output(
        self,
        broken_output: str,
        user_message: str,
        rewritten_user_message: str,
        factual_mode: bool,
    ) -> str:
        repair_prompt = (
            "次の回答文を、意味をなるべく保ったまま再構成してください。\n"
            "制約:\n"
            "1) 同じ内容の反復をしない\n"
            "2) 4〜7文に収める\n"
            "3) 断定しすぎない\n"
            "4) 日本語で自然な会話調にする\n"
            "5) 余計な前置きはしない\n"
            f"ユーザー入力: {user_message}\n"
            f"文脈補完後: {rewritten_user_message}\n"
            f"事実寄りモード: {'yes' if factual_mode else 'no'}\n"
            "修正対象:\n"
            f"{broken_output}"
        )
        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": repair_prompt}],
                max_tokens=min(420, self._response_max_tokens(factual_mode)),
                temperature=0.1,
            )
            message = self._extract_message(response)
            return (message.content or "").strip()
        except Exception as e:
            logger.warning("Output repair failed: %s", e)
            return broken_output

    def _postprocess_output(
        self,
        content: str,
        user_message: str,
        rewritten_user_message: str,
        factual_mode: bool,
    ) -> str:
        if not self._is_broken_output(content):
            return content
        repaired = self._repair_output(
            broken_output=content,
            user_message=user_message,
            rewritten_user_message=rewritten_user_message,
            factual_mode=factual_mode,
        )
        if not self._is_broken_output(repaired):
            return repaired
        return "回答が不安定だったため、短く言い直します。対象を1つに絞って聞いてください。"

    def _build_cited_fallback_from_tools(self) -> Optional[str]:
        for message in reversed(self.messages):
            if message.get("role") != "tool" or message.get("name") != "web_search":
                continue

            try:
                payload = json.loads(message.get("content", "{}"))
            except json.JSONDecodeError:
                continue

            results = payload.get("results", [])
            if not isinstance(results, list) or not results:
                continue

            lines = ["検索結果ベースの回答です。"]
            used = 0
            for row in results:
                if not isinstance(row, dict):
                    continue
                title = str(row.get("title", "")).strip()
                snippet = str(row.get("snippet", "")).strip()
                url = str(row.get("url", "")).strip()
                if not url:
                    continue
                summary = snippet or title or "関連情報"
                lines.append(f"- {summary}\n  出典: {url}")
                used += 1
                if used >= 3:
                    break

            if used == 0:
                continue

            lines.append(
                "上記の範囲で共通している内容のみを要約しました。必要なら対象を絞って再検索します。"
            )
            return "\n".join(lines)

        return None

    def _generate_without_tools(self, factual_mode: bool = False) -> str:
        response = self.client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=self.messages,
            max_tokens=self._response_max_tokens(factual_mode),
            temperature=config.TEMPERATURE,
        )
        message = self._extract_message(response)
        content = message.content or ""
        self.messages.append({"role": "assistant", "content": content})
        return content

    def send_message(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        return self._generate_without_tools()

    def send_message_with_tools(self, user_message: str, status_callback=None) -> str:
        if status_callback:
            status_callback("確認中...")

        history_for_router = self._recent_dialogue_for_router(self.messages)
        route = self.router.classify(user_message, history_for_router)
        mode = route.get("mode", "factual_balanced")
        confidence = float(route.get("confidence", 0.0))
        reason = route.get("reason", "")
        rewritten_user_message = (
            route.get("rewritten_user_message", "").strip() or user_message.strip()
        )
        logger.info(
            "Intent route mode=%s confidence=%.2f reason=%s rewritten=%s",
            mode,
            confidence,
            reason,
            rewritten_user_message,
        )

        if mode == "needs_clarification":
            self.messages.append({"role": "user", "content": user_message})
            clarifying = route.get("clarification_prompt") or DEFAULT_CLARIFYING_QUESTION
            self.messages.append({"role": "assistant", "content": clarifying})
            return clarifying

        factual_mode = mode in ("factual_balanced", "factual_strict")
        strict_factual_mode = mode == "factual_strict"
        self.messages.append({"role": "user", "content": rewritten_user_message})
        last_content = ""

        for _ in range(MAX_TOOL_ITERATIONS):
            try:
                request_messages = list(self.messages)
                if strict_factual_mode:
                    request_messages.append(FACTUAL_MODE_SYSTEM_MESSAGE)
                elif factual_mode:
                    request_messages.append(BALANCED_FACTUAL_MODE_SYSTEM_MESSAGE)
                request_payload = dict(
                    model=config.MODEL_NAME,
                    messages=request_messages,
                    max_tokens=self._response_max_tokens(factual_mode),
                    temperature=config.TEMPERATURE,
                    tools=TOOLS,
                )
                response = self.client.chat.completions.create(**request_payload)
                message = self._extract_message(response)
            except Exception as e:
                error_text = str(e)
                logger.warning("Tool-enabled request failed: %s", error_text)
                if self._is_tooling_error(error_text):
                    if strict_factual_mode:
                        safe_reply = (
                            "この質問は事実確認が必要ですが、現在の接続先モデルでは"
                            "検索ツール連携が利用できません。正確性を保証できないため回答を保留します。"
                        )
                        self.messages.append({"role": "assistant", "content": safe_reply})
                        return safe_reply
                    if status_callback:
                        status_callback("ツール未対応のため通常応答に切替")
                    return self._generate_without_tools(factual_mode=factual_mode)
                raise

            if not message.tool_calls:
                if status_callback:
                    status_callback("回答作成中...")

                content = self._postprocess_output(
                    content=message.content or "",
                    user_message=user_message,
                    rewritten_user_message=rewritten_user_message,
                    factual_mode=factual_mode,
                )
                last_content = content
                if strict_factual_mode and not self._has_citation(content):
                    tool_fallback = self._build_cited_fallback_from_tools()
                    if tool_fallback:
                        self.messages.append({"role": "assistant", "content": tool_fallback})
                        return tool_fallback
                    safe_reply = (
                        "出典URLを確認できる回答が得られませんでした。"
                        "このまま断定はできないため、質問を具体化してください。"
                    )
                    self.messages.append({"role": "assistant", "content": safe_reply})
                    return safe_reply
                if factual_mode and not strict_factual_mode and not self._has_citation(content):
                    tool_fallback = self._build_cited_fallback_from_tools()
                    if tool_fallback:
                        self.messages.append({"role": "assistant", "content": tool_fallback})
                        return tool_fallback
                self.messages.append({"role": "assistant", "content": content})
                return content

            self.messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                logger.info("Tool call: %s(%s)", func_name, arguments)

                if status_callback and func_name == "web_search":
                    query = arguments.get("query", "")
                    status_callback(f"検索中: {query}")

                try:
                    result = execute_tool_call(func_name, arguments)
                except ValueError as e:
                    result = json.dumps({"error": str(e)})

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": result,
                    }
                )

        logger.warning("Max tool iterations (%d) reached", MAX_TOOL_ITERATIONS)
        if strict_factual_mode:
            fallback = (
                "十分な出典付き情報を確認できなかったため、断定回答を避けます。"
                "質問を具体化するか、対象の公式サイト名を指定してください。"
            )
        elif factual_mode:
            tool_fallback = self._build_cited_fallback_from_tools()
            fallback = tool_fallback or (
                "正確性に注意しつつ回答します。必要なら条件を指定して再検索できます。"
            )
        else:
            fallback = last_content or "申し訳ありません。検索の処理中にエラーが発生しました。"
        self.messages.append({"role": "assistant", "content": fallback})
        return fallback

    def clear_history(self):
        self.messages.clear()
        self.messages.append(CHAT_SYSTEM_MESSAGE)
