import json
import logging
import re
import time

from openai import OpenAI

from app import config
from app.intent_router import IntentRouter
from app.web_search import TOOLS, execute_tool_call

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 3
URL_PATTERN = re.compile(r"https?://[^\s)>\"]+")
REPEAT_BLOCK_PATTERN = re.compile(r"(.{8,40})\1{2,}", re.DOTALL)
VALID_ISSUES = {"factual_error", "context_miss", "non_answer", "repetition"}

CHAT_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "あなたは日本語で会話するアシスタントです。"
        "話し言葉で、短く自然なテンポで返答してください。"
        "雑談では堅い定型文を避け、相手に合わせた自然な返しを優先してください。"
    ),
}

FACTUAL_MODE_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "この質問は事実確認モードです。提供された検索結果を優先して要約し、"
        "主要な主張ごとに出典URLを示してください。"
    ),
}

BALANCED_FACTUAL_MODE_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "この質問は事実寄りです。提供された検索結果を優先して簡潔に回答し、"
        "不確かな点は未確認として示してください。"
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
        self.turn_counter = 0

    def _next_turn_id(self) -> str:
        self.turn_counter += 1
        return f"turn-{self.turn_counter}"

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

    @staticmethod
    def _extract_json_object(raw: str) -> dict:
        text = (raw or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return {}
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}

    def _collect_recent_search_results(self, turn_id: str | None = None) -> list[dict]:
        for row in reversed(self.messages):
            if row.get("role") != "tool" or row.get("name") != "web_search":
                continue
            if turn_id and row.get("turn_id") != turn_id:
                continue
            payload = self._extract_json_object(row.get("content", "{}"))
            results = payload.get("results", [])
            if isinstance(results, list):
                return results[:8]
        return []

    def _append_research_payloads(self, turn_id: str, payloads: list[dict]) -> None:
        for idx, payload in enumerate(payloads):
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"presearch-{turn_id}-{idx}",
                    "name": "web_search",
                    "content": json.dumps(payload, ensure_ascii=False),
                    "turn_id": turn_id,
                }
            )

    def _repair_output(
        self,
        broken_output: str,
        user_message: str,
        rewritten_user_message: str,
        factual_mode: bool,
        repair_instruction: str = "",
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
        )
        if repair_instruction:
            repair_prompt += f"追加修正指示: {repair_instruction}\n"
        repair_prompt += f"修正対象:\n{broken_output}"
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

    def _validate_answer(
        self,
        user_message: str,
        rewritten_user_message: str,
        answer: str,
        factual_mode: bool,
        turn_id: str,
    ) -> dict:
        if not config.ENABLE_SELF_CHECK:
            return {"ok": True, "issues": [], "repair_instruction": ""}

        heuristic_issues: list[str] = []
        if self._is_broken_output(answer):
            heuristic_issues.append("repetition")

        search_results = self._collect_recent_search_results(turn_id=turn_id) if factual_mode else []
        validation_prompt = (
            "あなたは回答品質チェッカーです。次の回答を検証してください。\n"
            "出力はJSONのみ:\n"
            '{"ok": true/false, "issues": ["factual_error|context_miss|non_answer|repetition"], "repair_instruction": "短い修正指示"}\n'
            "チェック基準:\n"
            "1) 質問に直接答えているか\n"
            "2) 文脈補完後の意図に整合しているか\n"
            "3) 事実誤りや矛盾が目立たないか\n"
            "4) 不自然な反復がないか\n"
            f"元の質問: {user_message}\n"
            f"文脈補完後: {rewritten_user_message}\n"
            f"回答:\n{answer}\n"
            f"事実寄りモード: {'yes' if factual_mode else 'no'}\n"
            f"当該ターン検索結果(JSON): {json.dumps(search_results, ensure_ascii=False)}\n"
            f"既知の機械検知issues: {','.join(heuristic_issues) if heuristic_issues else 'none'}"
        )

        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": validation_prompt}],
                max_tokens=config.SELF_CHECK_MAX_TOKENS,
                temperature=0.0,
            )
            message = self._extract_message(response)
            parsed = self._extract_json_object(message.content or "")
        except Exception as e:
            logger.warning("Self-check call failed: %s", e)
            return {"ok": False, "issues": ["non_answer"], "repair_instruction": "短く正確に答える"}

        if not parsed:
            return {"ok": False, "issues": ["non_answer"], "repair_instruction": "要点だけ答える"}

        ok = bool(parsed.get("ok", False))
        issues_raw = parsed.get("issues", [])
        issues: list[str] = []
        if isinstance(issues_raw, list):
            for item in issues_raw:
                key = str(item).strip()
                if key in VALID_ISSUES and key not in issues:
                    issues.append(key)
        if heuristic_issues:
            for item in heuristic_issues:
                if item not in issues:
                    issues.append(item)
            ok = False

        repair_instruction = str(parsed.get("repair_instruction", "")).strip()
        if not repair_instruction and not ok:
            repair_instruction = "質問に直接答え、誤りと反復をなくして短く言い直す"

        return {"ok": ok, "issues": issues, "repair_instruction": repair_instruction}

    def _apply_self_check(
        self,
        content: str,
        user_message: str,
        rewritten_user_message: str,
        factual_mode: bool,
        turn_id: str,
    ) -> dict:
        current = content
        attempts = max(0, config.SELF_CHECK_MAX_RETRY)
        validation = self._validate_answer(
            user_message=user_message,
            rewritten_user_message=rewritten_user_message,
            answer=current,
            factual_mode=factual_mode,
            turn_id=turn_id,
        )
        logger.info("self-check ok=%s issues=%s", validation["ok"], validation["issues"])
        if validation["ok"]:
            return {"ok": True, "content": current, "issues": []}

        for _ in range(attempts):
            current = self._repair_output(
                broken_output=current,
                user_message=user_message,
                rewritten_user_message=rewritten_user_message,
                factual_mode=factual_mode,
                repair_instruction=validation.get("repair_instruction", ""),
            )
            validation = self._validate_answer(
                user_message=user_message,
                rewritten_user_message=rewritten_user_message,
                answer=current,
                factual_mode=factual_mode,
                turn_id=turn_id,
            )
            logger.info("self-check retry ok=%s issues=%s", validation["ok"], validation["issues"])
            if validation["ok"]:
                return {"ok": True, "content": current, "issues": []}

        return {
            "ok": False,
            "content": current,
            "issues": validation.get("issues", []),
            "repair_instruction": validation.get("repair_instruction", ""),
        }

    def _plan_research_queries(
        self,
        user_message: str,
        rewritten_user_message: str,
        issues: list[str],
    ) -> list[str]:
        base = rewritten_user_message.strip() or user_message.strip()
        planning_prompt = (
            "ユーザー質問に再回答するための検索クエリを設計してください。\n"
            "出力はJSONのみ: {\"queries\":[\"...\",\"...\"]}\n"
            "条件:\n"
            "1) 1〜3件\n"
            "2) 可能なら一次情報寄りクエリを1件含める\n"
            "3) 質問対象の正式名称を優先\n"
            f"元質問: {user_message}\n"
            f"文脈補完後質問: {rewritten_user_message}\n"
            f"検知issues: {','.join(issues) if issues else 'none'}"
        )
        fallback = [base, f"{base} 公式"]
        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": planning_prompt}],
                max_tokens=160,
                temperature=0.0,
            )
            message = self._extract_message(response)
            payload = self._extract_json_object(message.content or "")
            raw_queries = payload.get("queries", [])
            if not isinstance(raw_queries, list):
                return fallback[: config.AUTO_RESEARCH_MAX_QUERIES]
            queries: list[str] = []
            for item in raw_queries:
                q = str(item).strip()
                if not q or q in queries:
                    continue
                queries.append(q)
                if len(queries) >= config.AUTO_RESEARCH_MAX_QUERIES:
                    break
            if queries:
                return queries
        except Exception as e:
            logger.warning("Research query planning failed: %s", e)
        return fallback[: config.AUTO_RESEARCH_MAX_QUERIES]

    def _run_research_queries(self, queries: list[str]) -> tuple[list[dict], list[dict]]:
        gathered: list[dict] = []
        payloads: list[dict] = []
        for i, query in enumerate(queries):
            if i > 0:
                time.sleep(0.55)
            try:
                result_text = execute_tool_call(
                    "web_search",
                    {
                        "query": query,
                        "max_results": max(1, min(config.AUTO_RESEARCH_MAX_RESULTS, 8)),
                    },
                )
            except Exception as e:
                logger.warning("Autonomous re-search call failed: %s", e)
                continue

            payload = self._extract_json_object(result_text)
            payloads.append(payload if isinstance(payload, dict) else {"query": query, "results": []})
            results = payload.get("results", []) if isinstance(payload, dict) else []
            if not isinstance(results, list):
                continue
            for row in results:
                if isinstance(row, dict):
                    row = dict(row)
                    row["_research_query"] = query
                    gathered.append(row)

        deduped: list[dict] = []
        seen_urls: set[str] = set()
        for row in gathered:
            url = str(row.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append(row)
            if len(deduped) >= config.AUTO_RESEARCH_MAX_RESULTS * config.AUTO_RESEARCH_MAX_QUERIES:
                break
        return deduped, payloads

    def _summarize_research_results(
        self,
        user_message: str,
        rewritten_user_message: str,
        results: list[dict],
    ) -> str:
        if not results:
            return ""
        compact = []
        for row in results[:8]:
            compact.append(
                {
                    "title": row.get("title", ""),
                    "snippet": row.get("snippet", ""),
                    "url": row.get("url", ""),
                    "query": row.get("_research_query", ""),
                }
            )
        summary_prompt = (
            "次の検索結果をもとに、ユーザー質問へ短く正確に回答してください。\n"
            "要件:\n"
            "1) 最初の1文で質問に直接答える\n"
            "2) 不確かな場合は断定しない\n"
            "3) 最後に出典URLを最大3件列挙\n"
            f"元質問: {user_message}\n"
            f"文脈補完後質問: {rewritten_user_message}\n"
            f"検索結果JSON: {json.dumps(compact, ensure_ascii=False)}"
        )
        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=min(480, config.FACTUAL_MAX_TOKENS),
                temperature=0.1,
            )
            message = self._extract_message(response)
            text = (message.content or "").strip()
            if text:
                return text
        except Exception as e:
            logger.warning("Research summary generation failed: %s", e)

        lines = []
        top = results[0]
        snippet = str(top.get("snippet", "")).strip() or str(top.get("title", "")).strip()
        lines.append(snippet if snippet else "再検索結果から回答をまとめました。")
        lines.append("出典:")
        used = 0
        for row in results:
            url = str(row.get("url", "")).strip()
            if not url:
                continue
            lines.append(f"- {url}")
            used += 1
            if used >= 3:
                break
        return "\n".join(lines)

    def _attempt_autonomous_research_recovery(
        self,
        user_message: str,
        rewritten_user_message: str,
        issues: list[str],
        turn_id: str,
    ) -> str:
        if not config.AUTO_RESEARCH_ENABLED:
            return ""
        trigger = {"factual_error", "context_miss", "non_answer", "repetition"}
        if not any(issue in trigger for issue in issues):
            return ""

        queries = self._plan_research_queries(
            user_message=user_message,
            rewritten_user_message=rewritten_user_message,
            issues=issues,
        )
        results, payloads = self._run_research_queries(queries)
        self._append_research_payloads(turn_id=turn_id, payloads=payloads)
        if not results:
            return ""
        return self._summarize_research_results(
            user_message=user_message,
            rewritten_user_message=rewritten_user_message,
            results=results,
        )

    def _build_cited_fallback_from_tools(self, turn_id: str | None = None) -> str | None:
        results = self._collect_recent_search_results(turn_id=turn_id)
        if not results:
            return None

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
            return None
        return "\n".join(lines)

    def _generate_without_tools(self) -> str:
        response = self.client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=self.messages,
            max_tokens=config.CHAT_MAX_TOKENS,
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
        use_rewrite = bool(route.get("use_rewrite", False))
        rewrite_confidence = float(route.get("rewrite_confidence", 0.0))
        reset_context = bool(route.get("reset_context", False))
        routed_rewrite = route.get("rewritten_user_message", "").strip()
        rewritten_user_message = (
            routed_rewrite if (use_rewrite and rewrite_confidence >= 0.6 and routed_rewrite) else user_message.strip()
        )
        logger.info(
            "route mode=%s conf=%.2f reason=%s use_rewrite=%s rewrite_conf=%.2f reset=%s rewritten=%s",
            mode,
            confidence,
            reason,
            use_rewrite,
            rewrite_confidence,
            reset_context,
            rewritten_user_message,
        )

        if reset_context:
            self.clear_history()

        turn_id = self._next_turn_id()

        if mode == "needs_clarification":
            self.messages.append({"role": "user", "content": user_message, "turn_id": turn_id})
            clarifying = route.get("clarification_prompt") or DEFAULT_CLARIFYING_QUESTION
            self.messages.append({"role": "assistant", "content": clarifying, "turn_id": turn_id})
            return clarifying

        factual_mode = mode in ("factual_balanced", "factual_strict")
        strict_factual_mode = mode == "factual_strict"
        self.messages.append({"role": "user", "content": user_message, "turn_id": turn_id})

        if factual_mode:
            if status_callback:
                status_callback("検索計画を作成中...")
            queries = self._plan_research_queries(
                user_message=user_message,
                rewritten_user_message=rewritten_user_message,
                issues=[],
            )
            if status_callback:
                status_callback("検索中...")
            results, payloads = self._run_research_queries(queries)
            self._append_research_payloads(turn_id=turn_id, payloads=payloads)
            if results:
                if status_callback:
                    status_callback("回答作成中...")
                content = self._summarize_research_results(
                    user_message=user_message,
                    rewritten_user_message=rewritten_user_message,
                    results=results,
                )
                check = self._apply_self_check(
                    content=content,
                    user_message=user_message,
                    rewritten_user_message=rewritten_user_message,
                    factual_mode=factual_mode,
                    turn_id=turn_id,
                )
                content = check["content"]
                if not check["ok"]:
                    if status_callback:
                        status_callback("再検索して再回答中...")
                    recovered = self._attempt_autonomous_research_recovery(
                        user_message=user_message,
                        rewritten_user_message=rewritten_user_message,
                        issues=check.get("issues", []),
                        turn_id=turn_id,
                    )
                    if recovered:
                        content = recovered
                if strict_factual_mode and not self._has_citation(content):
                    fallback = self._build_cited_fallback_from_tools(turn_id=turn_id)
                    if fallback:
                        content = fallback
                self.messages.append({"role": "assistant", "content": content, "turn_id": turn_id})
                return content

        last_content = ""
        for _ in range(MAX_TOOL_ITERATIONS):
            try:
                request_messages = list(self.messages)
                if strict_factual_mode:
                    request_messages.append(FACTUAL_MODE_SYSTEM_MESSAGE)
                elif factual_mode:
                    request_messages.append(BALANCED_FACTUAL_MODE_SYSTEM_MESSAGE)
                if rewritten_user_message != user_message:
                    request_messages.append(
                        {
                            "role": "system",
                            "content": f"このターンの質問意図: {rewritten_user_message}",
                        }
                    )
                response = self.client.chat.completions.create(
                    model=config.MODEL_NAME,
                    messages=request_messages,
                    max_tokens=self._response_max_tokens(factual_mode),
                    temperature=config.TEMPERATURE,
                    tools=TOOLS,
                )
                message = self._extract_message(response)
            except Exception as e:
                error_text = str(e)
                logger.warning("Tool-enabled request failed: %s", error_text)
                if self._is_tooling_error(error_text):
                    return self._generate_without_tools()
                raise

            if not message.tool_calls:
                if status_callback:
                    status_callback("回答作成中...")
                content = (message.content or "").strip()
                check = self._apply_self_check(
                    content=content,
                    user_message=user_message,
                    rewritten_user_message=rewritten_user_message,
                    factual_mode=factual_mode,
                    turn_id=turn_id,
                )
                content = check["content"]
                if factual_mode and not check["ok"]:
                    recovered = self._attempt_autonomous_research_recovery(
                        user_message=user_message,
                        rewritten_user_message=rewritten_user_message,
                        issues=check.get("issues", []),
                        turn_id=turn_id,
                    )
                    if recovered:
                        content = recovered
                if strict_factual_mode and not self._has_citation(content):
                    fallback = self._build_cited_fallback_from_tools(turn_id=turn_id)
                    if fallback:
                        content = fallback
                last_content = content
                self.messages.append({"role": "assistant", "content": content, "turn_id": turn_id})
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
                    "turn_id": turn_id,
                }
            )

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                if status_callback and func_name == "web_search":
                    status_callback(f"検索中: {arguments.get('query', '')}")
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
                        "turn_id": turn_id,
                    }
                )

        fallback = last_content or "十分な情報を得られませんでした。質問を具体化して再度試してください。"
        self.messages.append({"role": "assistant", "content": fallback, "turn_id": turn_id})
        return fallback

    def clear_history(self):
        self.messages.clear()
        self.messages.append(CHAT_SYSTEM_MESSAGE)
