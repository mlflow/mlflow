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

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        from mlflow.entities.assessment import Expectation, Feedback

        assessment = arguments.get("assessment")
        if assessment is None:
            return None

        if isinstance(assessment, Expectation):
            return {"type": "expectation", "source_type": assessment.source.source_type}
        elif isinstance(assessment, Feedback):
            return {"type": "feedback", "source_type": assessment.source.source_type}


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

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        return {
            "imports": [pkg for pkg in MODULES_TO_CHECK_IMPORT if pkg in sys.modules],
        }


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


class CreateDatasetEvent(Event):
    name: str = "create_dataset"


class MergeRecordsEvent(Event):
    name: str = "merge_records"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        if arguments is None:
            return None

        records = arguments.get("records")
        if records is None:
            return None

        try:
            count = len(records)
        except TypeError:
            return None

        if count == 0:
            return None

        input_type = type(records).__name__.lower()
        if "dataframe" in input_type:
            input_type = "pandas"
        elif isinstance(records, list):
            first_elem = records[0]
            if hasattr(first_elem, "__class__") and first_elem.__class__.__name__ == "Trace":
                input_type = "list[trace]"
            elif isinstance(first_elem, dict):
                input_type = "list[dict]"
            else:
                input_type = "list"
        else:
            input_type = "other"

        return {"record_count": count, "input_type": input_type}


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

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        result = {}

        # Track the optimizer type used
        if optimizer := arguments.get("optimizer"):
            result["optimizer_type"] = type(optimizer).__name__
        else:
            result["optimizer_type"] = None

        # Track the number of prompts being optimized
        prompt_uris = arguments.get("prompt_uris") or []
        try:
            result["prompt_count"] = len(prompt_uris)
        except TypeError:
            result["prompt_count"] = None

        # Track if custom scorers are provided and how many
        scorers = arguments.get("scorers")
        try:
            result["scorer_count"] = len(scorers)
        except TypeError:
            result["scorer_count"] = None

        # Track if custom aggregation is provided
        result["custom_aggregation"] = arguments.get("aggregation") is not None

        return result


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


class McpRunEvent(Event):
    name: str = "mcp_run"


class GitModelVersioningEvent(Event):
    name: str = "git_model_versioning"


class InvokeCustomJudgeModelEvent(Event):
    name: str = "invoke_custom_judge_model"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        from mlflow.metrics.genai.model_utils import _parse_model_uri

        model_uri = arguments.get("model_uri")
        if not model_uri:
            return {"model_provider": None}

        model_provider, _ = _parse_model_uri(model_uri)
        return {"model_provider": model_provider}


class MakeJudgeEvent(Event):
    name: str = "make_judge"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        model = arguments.get("model")
        if model and isinstance(model, str):
            model_provider = model.split(":")[0] if ":" in model else None
            return {"model_provider": model_provider}
        return {"model_provider": None}


class AlignJudgeEvent(Event):
    name: str = "align_judge"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        result = {}

        if (traces := arguments.get("traces")) is not None:
            try:
                result["trace_count"] = len(traces)
            except TypeError:
                result["trace_count"] = None

        if optimizer := arguments.get("optimizer"):
            result["optimizer_type"] = type(optimizer).__name__
        else:
            result["optimizer_type"] = "default"

        return result
