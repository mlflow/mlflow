import sys
from typing import Any

from mlflow.telemetry.constant import PACKAGES_TO_CHECK_IMPORT


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


class CreatePromptEvent(Event):
    name: str = "create_prompt"


class StartTraceEvent(Event):
    name: str = "start_trace"


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
        return {"imports": [pkg for pkg in PACKAGES_TO_CHECK_IMPORT if pkg in sys.modules]}


class CreateModelVersionEvent(Event):
    name: str = "create_model_version"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        tags = arguments.get("tags") or {}
        return {"is_prompt": _is_prompt(tags)}


class CreateDatasetEvent(Event):
    name: str = "create_dataset"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        try:
            from mlflow.tracking import get_tracking_uri
            from mlflow.utils.databricks_utils import is_databricks_default_tracking_uri

            if is_databricks_default_tracking_uri(get_tracking_uri()):
                return None
            return {}
        except Exception:
            return None


class MergeRecordsEvent(Event):
    name: str = "merge_records"

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> dict[str, Any] | None:
        try:
            from mlflow.tracking import get_tracking_uri
            from mlflow.utils.databricks_utils import is_databricks_default_tracking_uri

            if is_databricks_default_tracking_uri(get_tracking_uri()):
                return None

            records = arguments.get("records")
            if records is None:
                return None

            # Detect input type
            input_type = type(records).__name__.lower()
            if "dataframe" in input_type:
                input_type = "pandas"
            elif isinstance(records, list) and records:
                # Check the first element to determine list type
                first_elem = records[0]
                if hasattr(first_elem, "__class__") and first_elem.__class__.__name__ == "Trace":
                    input_type = "trace"
                elif isinstance(first_elem, dict):
                    input_type = "dict"
                else:
                    input_type = "list"
            elif isinstance(records, list):
                # Empty list
                input_type = "list"
            else:
                input_type = "other"

            # Simple count - just try len() once
            try:
                count = len(records)
                if count > 0:
                    return {"record_count": count, "input_type": input_type}
            except (TypeError, AttributeError):
                pass

            return None
        except Exception:
            return None


def _is_prompt(tags: dict[str, str]) -> bool:
    try:
        from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
    except ImportError:
        return False
    return tags.get(IS_PROMPT_TAG_KEY, "false").lower() == "true"


class PromptOptimizationEvent(Event):
    name: str = "prompt_optimization"
