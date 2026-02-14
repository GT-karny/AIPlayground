import json
import sys
from types import SimpleNamespace

# ddgs未導入環境でも単体テストを実行できるように、最低限のスタブを差し込む。
if "ddgs" not in sys.modules:
    sys.modules["ddgs"] = SimpleNamespace(DDGS=object)

from app.api_client import ChatClient


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        if not self._responses:
            raise RuntimeError("no fake response queued")
        content = self._responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))]
        )


class _FakeOpenAI:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


def _build_client_with_fake_openai(monkeypatch, responses):
    monkeypatch.setattr("app.api_client.OpenAI", lambda **kwargs: _FakeOpenAI(responses))
    return ChatClient()


def _seed_turns(client: ChatClient) -> None:
    client.messages.extend(
        [
            {"role": "user", "content": "ガンダムって何？", "turn_id": "turn-1"},
            {"role": "assistant", "content": "ロボットアニメ作品だよ。", "turn_id": "turn-1"},
            {"role": "user", "content": "イニシャルDってなに？", "turn_id": "turn-2"},
            {"role": "assistant", "content": "走り屋漫画だよ。", "turn_id": "turn-2"},
        ]
    )


def test_topic_selector_uses_llm_result(monkeypatch):
    payload = {
        "use_context": True,
        "selected_turn_ids": ["turn-2"],
        "topic_summary": "頭文字D",
        "confidence": 0.88,
    }
    client = _build_client_with_fake_openai(monkeypatch, [json.dumps(payload, ensure_ascii=False)])
    _seed_turns(client)

    result = client._select_topic_context("主人公誰？", "主人公誰？")

    assert result["selected_turn_ids"] == ["turn-2"]
    assert result["topic_summary"] == "頭文字D"
    assert result["confidence"] == 0.88


def test_topic_selector_drops_low_confidence(monkeypatch):
    payload = {
        "use_context": True,
        "selected_turn_ids": ["turn-2"],
        "topic_summary": "頭文字D",
        "confidence": 0.10,
    }
    client = _build_client_with_fake_openai(monkeypatch, [json.dumps(payload, ensure_ascii=False)])
    _seed_turns(client)

    result = client._select_topic_context("主人公誰？", "主人公誰？")

    assert result == {"topic_summary": "", "selected_turn_ids": [], "confidence": 0.0}


def test_topic_selector_returns_empty_for_new_topic(monkeypatch):
    payload = {
        "use_context": False,
        "selected_turn_ids": [],
        "topic_summary": "",
        "confidence": 0.92,
    }
    client = _build_client_with_fake_openai(monkeypatch, [json.dumps(payload, ensure_ascii=False)])
    _seed_turns(client)

    result = client._select_topic_context("天気は？", "天気は？")

    assert result == {"topic_summary": "", "selected_turn_ids": [], "confidence": 0.0}
