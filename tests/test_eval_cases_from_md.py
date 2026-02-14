import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from app import config
from app.api_client import ChatClient

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ddgs").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)

AI_JUDGE_ENABLED = os.getenv("EVAL_AI_JUDGE_ENABLED", "true").lower() in ("1", "true", "yes", "on")
AI_JUDGE_MODEL = os.getenv("EVAL_AI_JUDGE_MODEL", config.MODEL_NAME)


@dataclass
class EvalCase:
    title: str
    users: list[str]
    expect: str


def _parse_eval_cases(md_path: Path) -> list[EvalCase]:
    raw = md_path.read_text(encoding="utf-8")
    cases: list[EvalCase] = []
    for block in re.split(r"^##\s+", raw, flags=re.MULTILINE):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        title = lines[0].strip()
        users = [m.group(1).strip() for m in re.finditer(r"^- User:\s*(.+)$", block, flags=re.MULTILINE)]
        expect_match = re.search(r"^- Expect:\s*(.+)$", block, flags=re.MULTILINE)
        if not users or not expect_match:
            continue
        cases.append(EvalCase(title=title, users=users, expect=expect_match.group(1).strip()))
    return cases


def _run_app_case(case: EvalCase) -> tuple[str, list[str]]:
    client = ChatClient()
    statuses: list[str] = []
    answer = ""
    for msg in case.users:
        answer = client.send_message_with_tools(msg, status_callback=statuses.append)
        logger.info("[APP][%s] user=%s", case.title, msg)
        logger.info("[APP][%s] answer=%s", case.title, answer)
    if statuses:
        logger.info("[APP][%s] statuses=%s", case.title, " -> ".join(statuses))
    return answer, statuses


def _ai_judge(case: EvalCase, answer: str) -> dict:
    judge_client = ChatClient()
    prompt = (
        "あなたは会話テストの判定者です。与えられた期待条件に対して、回答が合格かを判定してください。\n"
        "出力はJSONのみ:\n"
        '{"pass": true/false, "reason": "短い日本語理由"}\n'
        "判定基準:\n"
        "1) 期待条件を満たしているか\n"
        "2) 明らかな非回答や文脈逸脱がないか\n"
        "3) 必要条件(URL付き等)を満たしているか\n"
        "4) 回答文に存在しない内容を補完しない（推測で減点しない）\n"
        "5) 期待条件を満たしていれば pass=true にする\n"
        f"ケース名: {case.title}\n"
        f"会話(ユーザー): {json.dumps(case.users, ensure_ascii=False)}\n"
        f"期待条件: {case.expect}\n"
        f"アプリ回答: {answer}\n"
        "JSON以外を出力しないこと。"
    )
    response = judge_client.client.chat.completions.create(
        model=AI_JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=120,
    )
    message = judge_client._extract_message(response)
    payload = judge_client._extract_json_object(message.content or "")
    if not payload:
        raw = message.content or ""
        pass_match = re.search(r'"pass"\s*:\s*(true|false)', raw, flags=re.IGNORECASE)
        if not pass_match:
            raise AssertionError(f"AI judge returned non-JSON: {message.content}")
        return {
            "pass": pass_match.group(1).lower() == "true",
            "reason": "json_parse_fallback",
        }
    return payload


def _rule_judge(case: EvalCase, answer: str) -> dict:
    ok = bool(answer and answer.strip())
    return {
        "pass": ok,
        "reason": "ai_judge_disabled",
    }


CASES = _parse_eval_cases(Path("tests/eval_cases.md"))


@pytest.mark.parametrize("case", CASES, ids=[c.title for c in CASES])
def test_eval_cases_from_markdown(case: EvalCase):
    answer, _ = _run_app_case(case)
    result = _ai_judge(case, answer) if AI_JUDGE_ENABLED else _rule_judge(case, answer)
    logger.info("[JUDGE][%s] %s", case.title, json.dumps(result, ensure_ascii=False))

    verdict = bool(result.get("pass", False))
    reason = str(result.get("reason", ""))
    assert verdict, f"AI judge failed: title={case.title} reason={reason} answer={answer}"
