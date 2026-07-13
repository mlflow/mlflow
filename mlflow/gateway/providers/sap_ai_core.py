"""SAP AI Core Orchestration v2 provider for MLflow Gateway.

URI scheme: ``sap-ai-core:/<model-name>``

The model name is embedded in the Orchestration v2 request body under
``config.modules.prompt_templating.model.name``.  Auth is not handled here —
requests are routed through an HTTP egress gateway (``http://`` scheme) that
intercepts the call, attaches a bearer token, and forwards it externally.

Environment variable:
    ``MLFLOW_GENAI_JUDGE_BASE_URL`` — full URL of the AI Core Orchestration
    endpoint (required). Must use ``http://`` or ``https://`` scheme.
"""

from __future__ import annotations

from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.gateway.base_models import ConfigModel
from mlflow.gateway.providers.base import ProviderAdapter
from mlflow.gateway.providers.openai_compatible import (
    OpenAICompatibleAdapter,
    OpenAICompatibleProvider,
)
from mlflow.gateway.schemas import chat
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

#: Scalar inference parameter keys forwarded to the model's ``params`` block.
_FORWARDED_PARAMS = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "n",
        "seed",
    }
)


class SapAiCoreConfig(ConfigModel):
    """Configuration for the SAP AI Core Orchestration v2 provider.

    No API key is stored here — authentication is handled by the egress
    gateway that intercepts requests before they leave the cluster.
    The endpoint URL is read at request time from ``MLFLOW_GENAI_JUDGE_BASE_URL``.
    """


class SapAiCoreAdapter(ProviderAdapter):
    """Translates between MLflow's OpenAI-style payload and the AI Core
    Orchestration v2 request/response format.

    Request mapping
    ---------------
    OpenAI ``messages`` list → ``config.modules.prompt_templating.prompt.template``
    OpenAI model name        → ``config.modules.prompt_templating.model.name``
    OpenAI inference params  → ``config.modules.prompt_templating.model.params``
    ``response_format``      → ``config.modules.prompt_templating.model.params.response_format``
    ``tools`` / ``tool_choice`` → ``config.modules.prompt_templating.model.params``

    Response mapping
    ----------------
    The Orchestration v2 response wraps the LLM completion under a
    ``final_result`` key.  The adapter unwraps it and delegates to
    :class:`~mlflow.gateway.providers.openai_compatible.OpenAICompatibleAdapter`.
    """

    @classmethod
    def chat_to_model(cls, payload: dict[str, Any], config) -> dict[str, Any]:
        messages = payload.get("messages", [])
        model_name = config.model.name

        params: dict[str, Any] = {k: payload[k] for k in _FORWARDED_PARAMS if k in payload}

        # Pass response_format and tool-calling fields into model params so
        # structured-output enforcement and trace-loop tool calling are preserved.
        for key in ("response_format", "tools", "tool_choice"):
            if key in payload:
                params[key] = payload[key]

        prompt_templating: dict[str, Any] = {
            "prompt": {
                "template": [
                    {"role": m["role"], "content": m.get("content") or ""}
                    for m in messages
                ]
            },
            "model": {
                "name": model_name,
                **({"params": params} if params else {}),
            },
        }

        return {
            "config": {
                "modules": {
                    "prompt_templating": prompt_templating,
                }
            },
            "placeholder_values": payload.get("placeholder_values", {}),
        }

    @classmethod
    def model_to_chat(cls, resp: dict[str, Any], config) -> chat.ResponsePayload:
        # The Orchestration v2 wrapper: {"request_id": "...", "final_result": {...}, ...}
        # ``final_result`` is a standard OpenAI chat completion object.
        final = resp.get("final_result", resp)
        return OpenAICompatibleAdapter.model_to_chat(final, config)

    # ------------------------------------------------------------------ #
    # Required abstract stubs — not exercised via the judge path           #
    # ------------------------------------------------------------------ #

    @classmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError

    @classmethod
    def model_to_completions(cls, resp, config):
        raise NotImplementedError

    @classmethod
    def completions_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError


class SapAiCoreProvider(OpenAICompatibleProvider):
    """MLflow Gateway provider for SAP AI Core Orchestration v2.

    Reads the endpoint URL from ``MLFLOW_GENAI_JUDGE_BASE_URL`` at request
    time.  No ``Authorization`` header is added — the egress gateway is
    responsible for attaching credentials before the request leaves the
    cluster.  ``extra_headers`` (e.g. ``AI-Resource-Group``) are merged in
    by the
    :class:`~mlflow.genai.judges.adapters.gateway_adapter.GatewayAdapter`
    caller and forwarded as-is.
    """

    DISPLAY_NAME = "SAP AI Core"
    CONFIG_TYPE = SapAiCoreConfig
    DEFAULT_API_BASE = ""

    @property
    def adapter_class(self) -> type[SapAiCoreAdapter]:
        return SapAiCoreAdapter

    def get_endpoint_url(self, route_type: str) -> str:
        from mlflow.environment_variables import MLFLOW_GENAI_JUDGE_BASE_URL

        url = MLFLOW_GENAI_JUDGE_BASE_URL.get()
        if not url:
            raise MlflowException(
                "MLFLOW_GENAI_JUDGE_BASE_URL environment variable must be set "
                "when using the sap-ai-core:/ provider.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if not url.startswith(("http://", "https://")):
            raise MlflowException(
                f"MLFLOW_GENAI_JUDGE_BASE_URL must use http:// or https:// scheme, got: {url!r}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return url.rstrip("/")

    @property
    def headers(self) -> dict[str, str]:
        # Auth is handled by the egress gateway — no Authorization header here.
        return {}
