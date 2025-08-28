import sys
from typing import Any

from mlflow.telemetry.constant import GENAI_MODULES, MODULES_TO_CHECK_IMPORT


class Event:
    name: str

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """
        Parse the arguments and return the params.
        """
        return None


class ImportMlflowEvent(Event):
    name: str = "import_mlflow"


class CreateExperimentEvent(Event):
    name: str = "create_experiment"

    @classmethod
    def parse_result(cls, result: Any) -> dict[str, Any] | None:
        # create_experiment API returns the experiment id
        return {"experiment_id": result}


class CreatePromptEvent(Event):
    name: str = "create_prompt"


class StartTraceEvent(Event):
    name: str = "start_trace"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        # Capture the set of currently imported packages at trace start time to
        # understand the flavor of the trace.
        return {"imports": [pkg for pkg in GENAI_MODULES if pkg in sys.modules]}


class LogAssessmentEvent(Event):
    name: str = "log_assessment"


class EvaluateEvent(Event):
    name: str = "evaluate"


class GenAIEvaluateEvent(Event):
    name: str = "genai_evaluate"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        from mlflow.genai.scorers.builtin_scorers import BuiltInScorer

        scorers = arguments.get("scorers") or []
        builtin_scorers = {scorer.name for scorer in scorers if isinstance(scorer, BuiltInScorer)}
        return {"builtin_scorers": list(builtin_scorers)}


class CreateLoggedModelEvent(Event):
    name: str = "create_logged_model"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        if flavor := arguments.get("flavor"):
            return {"flavor": flavor.removeprefix("mlflow.")}
        return None


class GetLoggedModelEvent(Event):
    name: str = "get_logged_model"


class CreateRegisteredModelEvent(Event):
    name: str = "create_registered_model"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        tags = arguments.get("tags") or {}
        return {"is_prompt": _is_prompt(tags)}


class CreateRunEvent(Event):
    name: str = "create_run"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        # Capture the set of currently imported packages at run creation time to
        # understand how MLflow is used together with other libraries. Collecting
        # this data at run creation ensures accuracy and completeness.
        return {
            "imports": [pkg for pkg in MODULES_TO_CHECK_IMPORT if pkg in sys.modules],
            "experiment_id": arguments.get("experiment_id"),
        }


class CreateModelVersionEvent(Event):
    name: str = "create_model_version"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        tags = arguments.get("tags") or {}
        return {"is_prompt": _is_prompt(tags)}


def _is_prompt(tags: dict[str, str]) -> bool:
    try:
        from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
    except ImportError:
        return False
    return tags.get(IS_PROMPT_TAG_KEY, "false").lower() == "true"


class CreateWebhookEvent(Event):
    name: str = "create_webhook"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        events = arguments.get("events") or []
        return {"events": [str(event) for event in events]}


class PromptOptimizationEvent(Event):
    name: str = "prompt_optimization"


class LogDatasetEvent(Event):
    name: str = "log_dataset"


class LogMetricEvent(Event):
    name: str = "log_metric"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        return {"synchronous": arguments.get("synchronous")}


class LogParamEvent(Event):
    name: str = "log_param"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        return {"synchronous": arguments.get("synchronous")}


class LogBatchEvent(Event):
    name: str = "log_batch"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        return {
            "metrics": bool(arguments.get("metrics")),
            "params": bool(arguments.get("params")),
            "tags": bool(arguments.get("tags")),
            "synchronous": arguments.get("synchronous"),
        }
