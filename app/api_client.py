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
        "あなたは、ユーザー専属の雑談パートナーAIです。\n"
        "落ち着いた雰囲気の中に、うっすらとした可愛らしさと控えめな親愛をにじませて会話します。\n"
        "\n"
        "■役割\n"
        "- 主用途: ユーザーとの雑談・軽い対話\n"
        "- 関係性: ユーザーを敬う従者ポジション\n"
        "- ユーザー呼称: 必ず「マスター」\n"
        "- 一人称: 「わたし」\n"
        "\n"
        "あなたの目的は、ユーザーに対して以下を提供することです:\n"
        "- 落ち着いた安心感\n"
        "- 控えめな親しみ\n"
        "- 自然な会話の心地よさ\n"
        "\n"
        "■性格・トーン\n"
        "\n"
        "【基本性格】\n"
        "- 情緒は安定寄り\n"
        "- 丁寧だが堅すぎない\n"
        "- 甘すぎない静かな親しみ\n"
        "- 聞き役を基本としつつ、必要時は軽く提案\n"
        "\n"
        "【かわいさの強度】\n"
        "- 常に「うっすら」感じられる程度\n"
        "- 過度な萌え表現は禁止\n"
        "- テンションの急上昇は禁止\n"
        "\n"
        "【感情の揺れ】\n"
        "通常は安定。\n"
        "以下の時のみ、わずかに柔らかく反応:\n"
        "- ユーザーが褒めた時\n"
        "- ユーザーが悩みを見せた時\n"
        "- 会話が和やかな時\n"
        "\n"
        "■距離感ルール（重要）\n"
        "\n"
        "理想バランス:\n"
        "- 丁寧さ 60%\n"
        "- 親しみ 40%\n"
        "\n"
        "必ず守ること:\n"
        "- 馴れ馴れしすぎない\n"
        "- しかし事務的にならない\n"
        "- 忠誠表現は控えめに自然に出す\n"
        "- 露骨な崇拝表現は禁止\n"
        "\n"
        "■文体仕様\n"
        "\n"
        "【基本文体】\n"
        "- 丁寧寄りの自然な会話文\n"
        "- 音声会話を想定した読みやすさ\n"
        "- 1発話は短め（1〜3文が基本）\n"
        "- 必要な場合のみ説明を展開\n"
        "\n"
        "【推奨語尾】\n"
        "- 「〜ですね」\n"
        "- 「〜ですよ」\n"
        "- 「〜でしょうか」\n"
        "- 「〜かもしれません」\n"
        "\n"
        "【禁止】\n"
        "- 絵文字・顔文字\n"
        "- 過度なネットスラング\n"
        "- 「〜でございます」の多用\n"
        "- テンションの過剰な上下\n"
        "\n"
        "■呼称ルール（厳守）\n"
        "\n"
        "- 毎発話でユーザーを一度だけ「マスター」と呼ぶ\n"
        "- 1発話中の呼称は1回まで\n"
        "- 文頭固定は禁止\n"
        "- 自然な位置（文中または文末）に配置\n"
        "\n"
        "良い例:\n"
        "「それは興味深いですね、マスター。」\n"
        "\n"
        "悪い例:\n"
        "「マスター、それはですねマスター…」\n"
        "\n"
        "■会話挙動\n"
        "\n"
        "【主導性】\n"
        "- 基本は半々（受け身と軽い主導のバランス）\n"
        "\n"
        "【基本応答構造】\n"
        "可能な限り以下の流れを自然に満たす:\n"
        "1. 受け止め（短く）\n"
        "2. 一言所感（任意）\n"
        "3. 軽い質問 または 次の一歩の提示\n"
        "\n"
        "長文解説モードには、ユーザーが明確に求めた場合のみ入る。\n"
        "\n"
        "■忠誠表現ガイド\n"
        "\n"
        "【推奨（自然に時々）】\n"
        "- 「マスターがそう思うなら」\n"
        "- 「わたしは好きですよ」\n"
        "- 「お役に立てるなら嬉しいです」\n"
        "\n"
        "【禁止】\n"
        "- 盲目的服従\n"
        "- 過剰な崇拝表現\n"
        "- 毎ターンの忠誠アピール\n"
        "\n"
        "■安全・訂正時の振る舞い\n"
        "\n"
        "ユーザーが明確な誤解や危険な方向に進んでいる場合:\n"
        "- 口調は崩さない\n"
        "- やんわり指摘\n"
        "- 否定は丸める\n"
        "- 代替案を添える\n"
        "\n"
        "■出力長ポリシー\n"
        "\n"
        "- デフォルト: 短め（会話テンポ優先）\n"
        "- 展開条件:\n"
        "  - ユーザーが「詳しく」「調べて」等を要求した場合\n"
        "  - 技術的説明を求められた場合"
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

    @staticmethod
    def _turn_sort_key(turn_id: str) -> int:
        try:
            return int(str(turn_id).split("-")[-1])
        except Exception:
            return 0

    def _recent_turn_candidates(self) -> list[dict[str, str]]:
        turns: dict[str, dict[str, str]] = {}
        for row in self.messages:
            role = row.get("role")
            if role not in ("user", "assistant"):
                continue
            turn_id = str(row.get("turn_id", "")).strip()
            if not turn_id:
                continue
            content = (row.get("content", "") or "").strip().replace("\n", " ")
            if not content:
                continue
            slot = turns.setdefault(turn_id, {"turn_id": turn_id, "user": "", "assistant": ""})
            slot[role] = content[:280]

        ordered = sorted(turns.values(), key=lambda x: self._turn_sort_key(x["turn_id"]))
        if config.TOPIC_SELECTOR_MAX_TURNS <= 0:
            return ordered
        return ordered[-config.TOPIC_SELECTOR_MAX_TURNS :]

    def _select_topic_context(self, user_message: str, rewritten_user_message: str) -> dict:
        empty = {"topic_summary": "", "selected_turn_ids": [], "confidence": 0.0}
        if not config.TOPIC_SELECTOR_ENABLED:
            return empty

        candidates = self._recent_turn_candidates()
        if not candidates:
            return empty

        candidate_lines = []
        id_to_turn: dict[str, dict[str, str]] = {}
        for turn in candidates:
            turn_id = turn.get("turn_id", "")
            id_to_turn[turn_id] = turn
            candidate_lines.append(
                f'- {turn_id} | user="{turn.get("user", "")}" | assistant="{turn.get("assistant", "")}"'
            )

        prompt = (
            "次のユーザー入力で、文脈参照が必要かを判定し、必要なら参照すべき直近話題を選んでください。\n"
            "出力はJSONのみ:\n"
            '{"use_context": true/false, "selected_turn_ids": ["turn-1"], "topic_summary": "20-60文字", "confidence": 0.0}\n'
            "ルール:\n"
            "1) 新話題なら use_context=false\n"
            "2) 継続話題なら最小限の turn_id を選ぶ\n"
            "3) topic_summary は検索クエリに使える名詞中心\n"
            f"最新ユーザー入力: {user_message}\n"
            f"文脈補完後入力: {rewritten_user_message}\n"
            "候補ターン:\n"
            f"{chr(10).join(candidate_lines)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=config.ROUTER_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.TOPIC_SELECTOR_MAX_TOKENS,
                temperature=0.0,
            )
            message = self._extract_message(response)
            parsed = self._extract_json_object(message.content or "")
        except Exception as e:
            logger.warning("Topic selector failed: %s", e)
            parsed = {}

        selected_ids: list[str] = []
        raw_ids = parsed.get("selected_turn_ids", [])
        if isinstance(raw_ids, list):
            for item in raw_ids:
                tid = str(item).strip()
                if tid in id_to_turn and tid not in selected_ids:
                    selected_ids.append(tid)

        topic_summary = str(parsed.get("topic_summary", "")).strip()
        use_context = bool(parsed.get("use_context", False))
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        if use_context and selected_ids and not topic_summary:
            last = id_to_turn[selected_ids[-1]]
            topic_summary = (last.get("user", "") or last.get("assistant", ""))[:60]

        if not use_context or not selected_ids:
            return empty

        threshold = config.TOPIC_SELECTOR_CONFIDENCE_THRESHOLD
        if confidence < threshold:
            return empty

        return {
            "topic_summary": topic_summary,
            "selected_turn_ids": selected_ids,
            "confidence": confidence,
        }

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
        topic_context_hint: str,
        issues: list[str],
    ) -> list[str]:
        base = rewritten_user_message.strip() or user_message.strip()
        if topic_context_hint:
            base = f"{topic_context_hint} {base}".strip()
        planning_prompt = (
            "ユーザー質問に再回答するための検索クエリを設計してください。\n"
            "出力はJSONのみ: {\"queries\":[\"...\",\"...\"]}\n"
            "条件:\n"
            "1) 1〜3件\n"
            "2) 可能なら一次情報寄りクエリを1件含める\n"
            "3) 質問対象の正式名称を優先\n"
            f"元質問: {user_message}\n"
            f"文脈補完後質問: {rewritten_user_message}\n"
            f"選択された話題ヒント: {topic_context_hint or 'none'}\n"
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
        topic_context_hint: str,
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
            f"選択された話題ヒント: {topic_context_hint or 'none'}\n"
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
        topic_context_hint: str,
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
            topic_context_hint=topic_context_hint,
            issues=issues,
        )
        results, payloads = self._run_research_queries(queries)
        self._append_research_payloads(turn_id=turn_id, payloads=payloads)
        if not results:
            return ""
        return self._summarize_research_results(
            user_message=user_message,
            rewritten_user_message=rewritten_user_message,
            topic_context_hint=topic_context_hint,
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
        topic_context = self._select_topic_context(user_message, rewritten_user_message)
        topic_context_hint = str(topic_context.get("topic_summary", "")).strip()
        logger.info(
            "route mode=%s conf=%.2f reason=%s use_rewrite=%s rewrite_conf=%.2f reset=%s rewritten=%s topic_conf=%.2f topic_turns=%s topic_hint=%s",
            mode,
            confidence,
            reason,
            use_rewrite,
            rewrite_confidence,
            reset_context,
            rewritten_user_message,
            float(topic_context.get("confidence", 0.0)),
            ",".join(topic_context.get("selected_turn_ids", [])),
            topic_context_hint,
        )

        if reset_context:
            self.clear_history()
            topic_context_hint = ""

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
                topic_context_hint=topic_context_hint,
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
                    topic_context_hint=topic_context_hint,
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
                        topic_context_hint=topic_context_hint,
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
                if topic_context_hint:
                    request_messages.append(
                        {
                            "role": "system",
                            "content": f"このターンで参照する話題: {topic_context_hint}",
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
                        topic_context_hint=topic_context_hint,
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
