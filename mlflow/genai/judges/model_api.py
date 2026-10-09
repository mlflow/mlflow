"""API selection for custom judge models."""

from mlflow.exceptions import MlflowException
from mlflow.genai.utils.enum_utils import StrEnum


class ModelAPI(StrEnum):
    """How :func:`mlflow.genai.judges.make_judge` invokes its model."""

    DEFAULT = "default"
    CHAT_COMPLETIONS = "chat_completions"
    DECISIONS = "decisions"


_SUPPORTED_APIS = {
    "openai": frozenset({ModelAPI.CHAT_COMPLETIONS, ModelAPI.DECISIONS}),
    "typesafe": frozenset({ModelAPI.DECISIONS}),
    "gateway": frozenset({ModelAPI.CHAT_COMPLETIONS, ModelAPI.DECISIONS}),
}


def _parse_model_api(model_api: ModelAPI | str | None) -> ModelAPI:
    try:
        return ModelAPI.DEFAULT if model_api is None else ModelAPI(model_api)
    except (TypeError, ValueError):
        raise MlflowException.invalid_parameter_value(
            "model_api must be 'default', 'chat_completions', or 'decisions'."
        ) from None


def _resolve_model_api(model_uri: str, model_api: ModelAPI | str | None) -> ModelAPI:
    selected = _parse_model_api(model_api)
    provider = model_uri.partition(":/")[0]
    supported = _SUPPORTED_APIS.get(provider, frozenset({ModelAPI.CHAT_COMPLETIONS}))
    if selected is ModelAPI.DEFAULT:
        if provider == "typesafe":
            return ModelAPI.DECISIONS
        # A gateway:/ URI does not tell us which API its endpoint serves. The existing
        # chat-first route detects a System One endpoint and switches to Decisions.
        if provider == "gateway":
            return ModelAPI.DEFAULT
        return ModelAPI.CHAT_COMPLETIONS

    if selected not in supported:
        supported_names = ", ".join(f"'{api.value}'" for api in sorted(supported))
        raise MlflowException.invalid_parameter_value(
            f"model_api='{selected.value}' is not supported for {provider} judge models. "
            f"Supported APIs: {supported_names}."
        )
    return selected
