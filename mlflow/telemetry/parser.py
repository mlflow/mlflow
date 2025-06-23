import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from mlflow.telemetry.schemas import (
    AutologParams,
    GenaiEvaluateParams,
    LogModelParams,
    ModelType,
)

if TYPE_CHECKING:
    from mlflow.genai.scorers import Scorer

_logger = logging.getLogger(__name__)


class TelemetryParser(ABC):
    @classmethod
    @abstractmethod
    def extract_params(cls, func_name: str, arguments: dict[str, Any]) -> Any:
        """
        Extract the parameters from the function call.

        Args:
            func_name: The full function name.
            arguments: The arguments passed to the function.
        """


class LogModelParser(TelemetryParser):
    @classmethod
    def extract_params(cls, func_name: str, arguments: dict[str, Any]) -> LogModelParams | None:
        splits = func_name.rsplit(".", 2)
        if len(splits) != 3:
            _logger.warning(f"Failed to extract log model params for function {func_name}")
            return
        flavor = splits[1]

        # model parameter is the first positional argument
        model = next(iter(arguments.values()), None)
        model_type = ModelType.MODEL_PATH if isinstance(model, str) else cls.parse_model_type(model)

        record_params = {"flavor": flavor, "model": model_type.value}
        for param in [
            "pip_requirements",
            "extra_pip_requirements",
            "code_paths",
            "params",
            "metadata",
        ]:
            record_params[f"is_{param}_set"] = arguments.get(param) is not None

        return LogModelParams(**record_params)

    @classmethod
    def parse_model_type(cls, model: Any) -> ModelType:
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

        return ModelType.PYTHON_FUNCTION if callable(model) else ModelType.MODEL_OBJECT


class AutologParser(TelemetryParser):
    @classmethod
    def extract_params(cls, func_name: str, arguments: dict[str, Any]) -> AutologParams | None:
        splits = func_name.rsplit(".", 2)
        if len(splits) != 3:
            _logger.warning(f"Failed to extract autolog params for function {func_name}")
            return
        flavor = splits[1]
        record_params = {"flavor": flavor}
        for param in ["disable", "log_traces", "log_models"]:
            record_params[param] = arguments.get(param, False) or False
        return AutologParams(**record_params)


class GenaiEvaluateParser(TelemetryParser):
    @classmethod
    def extract_params(cls, func_name: str, arguments: dict[str, Any]) -> GenaiEvaluateParams:
        scorers = arguments.get("scorers", [])
        scorers = [cls.sanitize_scorer_name(scorer) for scorer in scorers]
        is_predict_fn_set = arguments.get("predict_fn") is not None
        return GenaiEvaluateParams(scorers=scorers, is_predict_fn_set=is_predict_fn_set)

    @classmethod
    def sanitize_scorer_name(cls, scorer: "Scorer") -> str:
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


API_PARSER_MAPPING: dict[str, TelemetryParser] = {
    "log_model": LogModelParser,
    "autolog": AutologParser,
    "mlflow.genai.evaluate": GenaiEvaluateParser,
}
