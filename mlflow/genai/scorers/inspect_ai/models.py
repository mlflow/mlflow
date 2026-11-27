"""Inspect AI model provider that routes grader calls through MLflow's judge backends."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from mlflow.genai.scorers.inspect_ai.utils import check_inspect_ai_installed
from mlflow.genai.scorers.llm_backend import ScorerLLMClient

if TYPE_CHECKING:
    from inspect_ai.model import Model

MLFLOW_PROVIDER_NAME = "mlflow"

# GenerateConfig fields forwarded to the MLflow backend when an Inspect scorer sets them.
_FORWARDED_GENERATE_FIELDS = ("temperature", "max_tokens", "top_p", "seed")

_PROVIDER_STATE = {"registered": False}


def _register_mlflow_provider() -> None:
    """Register ``mlflow/<model-uri>`` as an Inspect model provider. Safe to call repeatedly."""
    if _PROVIDER_STATE["registered"]:
        return

    from inspect_ai.model import GenerateConfig, ModelAPI, ModelOutput, modelapi

    class MlflowModelAPI(ModelAPI):
        """Inspect ``ModelAPI`` whose generations go through ``ScorerLLMClient``.

        ``model_name`` is an MLflow judge model URI such as ``openai:/gpt-4o-mini`` or
        ``databricks``, so the Databricks managed judge, MLflow's native providers and the
        LiteLLM fallback are all available to Inspect's model-graded scorers.
        """

        def __init__(
            self,
            model_name: str,
            base_url: str | None = None,
            api_key: str | None = None,
            config: GenerateConfig | None = None,
            model_kwargs: dict[str, Any] | None = None,
            **model_args: Any,
        ):
            super().__init__(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                api_key_vars=[],
                config=config or GenerateConfig(),
            )
            self._backend = ScorerLLMClient(model_name)
            self._model_kwargs = dict(model_kwargs or {})

        async def generate(self, input, tools, tool_choice, config) -> ModelOutput:
            messages = [{"role": message.role, "content": message.text} for message in input]
            kwargs = {
                field: getattr(config, field)
                for field in _FORWARDED_GENERATE_FIELDS
                if getattr(config, field) is not None
            }
            kwargs.update(self._model_kwargs)
            # The MLflow client is synchronous. Run it off the event loop so Inspect's own
            # async machinery keeps working around it.
            text = await asyncio.to_thread(self._backend.complete, messages, **kwargs)
            return ModelOutput.from_content(model=self.model_name, content=text)

    @modelapi(name=MLFLOW_PROVIDER_NAME)
    def _mlflow_model_api():
        return MlflowModelAPI

    _PROVIDER_STATE["registered"] = True


def create_inspect_model(model_uri: str, model_kwargs: dict[str, Any] | None = None) -> Model:
    """Create an Inspect ``Model`` for ``model_uri`` whose generations use MLflow's backends."""
    check_inspect_ai_installed()
    _register_mlflow_provider()
    from inspect_ai.model import get_model

    return get_model(
        f"{MLFLOW_PROVIDER_NAME}/{model_uri}",
        memoize=False,
        model_kwargs=model_kwargs or {},
    )
