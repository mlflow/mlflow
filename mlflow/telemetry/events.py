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
