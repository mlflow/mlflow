"""Registry mapping metric names to ADK evaluator classes and metadata."""

from __future__ import annotations

from mlflow.exceptions import MlflowException

ADK_NOT_INSTALLED_ERROR_MESSAGE = (
    "google-adk is required to use ADK scorers. "
    "Install it with: pip install google-adk"
)

# Registry format: metric_name -> (adk_metric_name, is_deterministic)
_METRIC_REGISTRY = {
    "ToolTrajectory": ("tool_trajectory_avg_score", True),
    "FinalResponseMatch": ("final_response_match_v2", False),
    "ResponseMatch": ("response_match_score", True),
    "RubricBasedResponseQuality": ("rubric_based_final_response_quality_v1", False),
    "RubricBasedToolUseQuality": ("rubric_based_tool_use_quality_v1", False),
    "Hallucinations": ("hallucinations_v1", False),
}

# Map ADK metric names to evaluator class paths
_EVALUATOR_CLASSES = {
    "tool_trajectory_avg_score": (
        "google.adk.evaluation.trajectory_evaluator",
        "TrajectoryEvaluator",
    ),
    "final_response_match_v2": (
        "google.adk.evaluation.final_response_match_v2",
        "FinalResponseMatchV2Evaluator",
    ),
    "response_match_score": (
        "google.adk.evaluation.response_evaluator",
        "ResponseEvaluator",
    ),
    "rubric_based_final_response_quality_v1": (
        "google.adk.evaluation.rubric_based_final_response_quality_v1",
        "RubricBasedFinalResponseQualityV1Evaluator",
    ),
    "rubric_based_tool_use_quality_v1": (
        "google.adk.evaluation.rubric_based_tool_use_quality_v1",
        "RubricBasedToolUseV1Evaluator",
    ),
    "hallucinations_v1": (
        "google.adk.evaluation.hallucinations_v1",
        "HallucinationsV1Evaluator",
    ),
}


def get_adk_metric_name(metric_name: str) -> str:
    """Get the ADK metric name for a given scorer metric name."""
    if metric_name in _METRIC_REGISTRY:
        return _METRIC_REGISTRY[metric_name][0]
    raise MlflowException.invalid_parameter_value(
        f"Unknown ADK metric: '{metric_name}'. "
        f"Available metrics: {', '.join(sorted(_METRIC_REGISTRY.keys()))}"
    )


def is_deterministic_metric(metric_name: str) -> bool:
    """Check if a metric is deterministic (no LLM judge needed)."""
    if metric_name in _METRIC_REGISTRY:
        return _METRIC_REGISTRY[metric_name][1]
    return False


def get_evaluator_class(adk_metric_name: str):
    """Dynamically import and return the ADK evaluator class."""
    if adk_metric_name not in _EVALUATOR_CLASSES:
        available = ", ".join(sorted(_EVALUATOR_CLASSES.keys()))
        raise MlflowException.invalid_parameter_value(
            f"No evaluator found for ADK metric '{adk_metric_name}'. "
            f"Available: {available}"
        )

    module_path, class_name = _EVALUATOR_CLASSES[adk_metric_name]
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        raise MlflowException.invalid_parameter_value(
            ADK_NOT_INSTALLED_ERROR_MESSAGE
        ) from e
    except AttributeError:
        raise MlflowException.invalid_parameter_value(
            f"Could not find class '{class_name}' in module '{module_path}'. "
            "Ensure you have a compatible version of google-adk installed."
        )
