from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage, ExtractedToolCallInformation
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.pythonic_tool_parser import PythonicToolParser


@ToolParserManager.register_module(["liquid_pythonic"])
class LiquidPythonicToolParser(PythonicToolParser):
    START = "<|tool_call_start|>"
    END = "<|tool_call_end|>"

    @classmethod
    def _strip_wrappers(cls, text: str) -> str:
        if not text:
            return text
        if cls.START in text and cls.END in text:
            start_idx = text.find(cls.START) + len(cls.START)
            end_idx = text.rfind(cls.END)
            if start_idx <= end_idx:
                return text[start_idx:end_idx].strip()
        return text.replace(cls.START, "").replace(cls.END, "").strip()

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        normalized = self._strip_wrappers(model_output)
        return super().extract_tool_calls(normalized, request)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        return super().extract_tool_calls_streaming(
            self._strip_wrappers(previous_text),
            self._strip_wrappers(current_text),
            self._strip_wrappers(delta_text),
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
