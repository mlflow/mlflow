from __future__ import annotations

import logging
import threading

from mlflow.genai.discovery.constants import LLM_MAX_TOKENS, NUM_RETRIES
from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm
from mlflow.metrics.genai.model_utils import convert_mlflow_uri_to_litellm

_logger = logging.getLogger(__name__)


class _TokenCounter:
    """Thread-safe accumulator for LLM token usage across pipeline phases."""

    def __init__(self):
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    def track(self, response) -> None:
        with self._lock:
            if usage := getattr(response, "usage", None):
                self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            if hidden := getattr(response, "_hidden_params", None):
                if cost := hidden.get("response_cost"):
                    self.cost_usd += cost

    def to_dict(self) -> dict[str, object]:
        result = {}
        total = self.input_tokens + self.output_tokens
        if total > 0:
            result["input_tokens"] = self.input_tokens
            result["output_tokens"] = self.output_tokens
            result["total_tokens"] = total
        if self.cost_usd > 0:
            result["cost_usd"] = round(self.cost_usd, 6)
        return result


def _call_llm(
    model: str,
    messages: list[dict[str, str]],
    *,
    json_mode: bool = False,
    token_counter: _TokenCounter | None = None,
) -> object:
    response = _invoke_litellm(
        litellm_model=convert_mlflow_uri_to_litellm(model),
        messages=messages,
        tools=[],
        num_retries=NUM_RETRIES,
        response_format={"type": "json_object"} if json_mode else None,
        include_response_format=json_mode,
        inference_params={"max_tokens": LLM_MAX_TOKENS},
    )
    if token_counter is not None:
        token_counter.track(response)
    return response
