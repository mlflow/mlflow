import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable

from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY
from mlflow.telemetry.schemas import (
    AutologParams,
    GenaiEvaluateParams,
    LogModelParams,
    ModelType,
)

if TYPE_CHECKING:
    from mlflow.genai.scorers.base import Scorer

_logger = logging.getLogger(__name__)


@contextmanager
def temporarily_disable_telemetry():
    original_value = MLFLOW_DISABLE_TELEMETRY.get() if MLFLOW_DISABLE_TELEMETRY.is_set() else None
    try:
        MLFLOW_DISABLE_TELEMETRY.set(True)
        yield
    finally:
        if original_value is None:
            MLFLOW_DISABLE_TELEMETRY.unset()
        else:
            MLFLOW_DISABLE_TELEMETRY.set(original_value)


def is_telemetry_disabled() -> bool:
    return (
        MLFLOW_DISABLE_TELEMETRY.get() or os.environ.get("DO_NOT_TRACK", "false").lower() == "true"
    )


def _get_model_type(model: Any) -> ModelType:
    try:
        from mlflow.pyfunc.model import ChatAgent, ChatModel, PythonModel
    except ImportError:
        pass
    else:
        if isinstance(model, PythonModel):
            return ModelType.PYTHON_MODEL
        elif isinstance(model, ChatModel):
            return ModelType.CHAT_MODEL
        elif isinstance(model, ChatAgent):
            return ModelType.CHAT_AGENT

    try:
        from mlflow.pyfunc.model import ResponsesAgent
    except ImportError:
        pass
    else:
        if isinstance(model, ResponsesAgent):
            return ModelType.RESPONSES_AGENT

    return ModelType.MODEL_OBJECT


def _extract_log_model_params(
    full_func_name: str, arguments: dict[str, Any]
) -> LogModelParams | None:
    splits = full_func_name.rsplit(".", 2)
    if len(splits) != 3:
        _logger.warning(f"Failed to extract log model params for function {full_func_name}")
        return
    flavor = splits[1]

    # model parameter is the first positional argument
    model = next(iter(arguments.values()), None)
    model_type = ModelType.MODEL_PATH if isinstance(model, str) else _get_model_type(model)

    record_params = LogModelParams(flavor=flavor, model=model_type)
    for param in [
        "pip_requirements",
        "extra_pip_requirements",
        "code_paths",
        "params",
        "metadata",
    ]:
        if arguments.get(param) is not None:
            setattr(record_params, param, True)

    return record_params


def _extract_autolog_params(full_func_name: str, arguments: dict[str, Any]) -> AutologParams | None:
    splits = full_func_name.rsplit(".", 2)
    if len(splits) != 3:
        _logger.warning(f"Failed to extract autolog params for function {full_func_name}")
        return
    flavor = splits[1]
    record_params = AutologParams(flavor=flavor)
    for param in ["disable", "log_traces", "log_models"]:
        if (value := arguments.get(param)) is not None:
            setattr(record_params, param, value)
    return record_params


def _sanitize_scorer_name(scorer: "Scorer") -> str:
    """
    Sanitize the scorer name to remove user-customized scorers.
    """
    try:
        from mlflow.genai.scorers.builtin_scorers import BuiltInScorer
    except ImportError:
        pass
    else:
        if isinstance(scorer, BuiltInScorer):
            return scorer.name

    return "CustomScorer"


def _extract_genai_evaluate_params(
    full_func_name: str, arguments: dict[str, Any]
) -> GenaiEvaluateParams:
    record_params = GenaiEvaluateParams()
    scorers = arguments.get("scorers")
    if scorers:
        record_params.scorers = [_sanitize_scorer_name(scorer) for scorer in scorers]
    if arguments.get("predict_fn") is not None:
        record_params.predict_fn = True
    return record_params


# NB: the callables should have the same signature as follows:
# def func(full_func_name: str, arguments: dict[str, Any])
API_RECORD_PARAMS_MAPPING: dict[str, Callable] = {
    "log_model": _extract_log_model_params,
    "autolog": _extract_autolog_params,
    "mlflow.genai.evaluate": _extract_genai_evaluate_params,
}
