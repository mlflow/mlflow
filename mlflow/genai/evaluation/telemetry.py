import json
import threading
import uuid

import mlflow
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.builtin_scorers import BuiltInScorer
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import _REST_API_PATH_PREFIX, call_endpoint
from mlflow.utils.uri import is_databricks_uri
from mlflow.version import VERSION

_SESSION_ID_HEADER = "eval-session-id"
_BATCH_SIZE_HEADER = "eval-session-batch-size"
_CLIENT_VERSION_HEADER = "eval-session-client-version"
_CLIENT_NAME_HEADER = "eval-session-client-name"

_EVAL_TELEMETRY_ENDPOINT = f"{_REST_API_PATH_PREFIX}/agents/evaluation-client-usage-events"


_sessions = threading.local()
_SESSION_KEY = "genai-eval-session"


def emit_custom_metric_event(
    scorers: list[Scorer], eval_count: int | None, aggregated_metrics: dict[str, float]
):
    """Emit events for custom scorers and metrics usage if running on Databricks"""
    if not is_databricks_uri(mlflow.get_tracking_uri()):
        return

    custom_metrics = list(filter(_is_custom_scorer, scorers))

    metric_stats = {m.name: {"average": None, "count": eval_count} for m in custom_metrics}
    for metric_key, metric_value in aggregated_metrics.items():
        name, aggregation = metric_key.split("/", 1)
        if name in metric_stats and aggregation == "mean":
            metric_stats[name]["average"] = metric_value

    metric_stats = [
        {
            "name": name,
            "average": metric_stats[name]["average"],
            "count": metric_stats[name]["count"],
        }
        for name in metric_stats
    ]

    event_payload = {
        "metric_names": [metric.name for metric in custom_metrics],
        "eval_count": eval_count,
        "metrics": metric_stats,
    }
    call_endpoint(
        host_creds=get_databricks_host_creds(),
        endpoint=_EVAL_TELEMETRY_ENDPOINT,
        method="POST",
        headers={
            _CLIENT_VERSION_HEADER: VERSION,
            _SESSION_ID_HEADER: _get_or_create_session_id(),
            _BATCH_SIZE_HEADER: eval_count,
            _CLIENT_NAME_HEADER: "mlflow",
        },
        json_body=json.dumps(event_payload),
    )


def _is_custom_scorer(scorer: Scorer) -> bool:
    if isinstance(scorer, Scorer):
        return not isinstance(scorer, BuiltInScorer)

    # Check for the legacy custom metrics if databricks-agents is installed
    try:
        from databricks.rag_eval.evaluation.custom_metrics import CustomMetric

        return isinstance(scorer, CustomMetric)
    except ImportError:
        return False

    # Treat unknown case as not a custom scorer
    return False


def _get_or_create_session_id() -> str:
    if not hasattr(_sessions, _SESSION_KEY):
        setattr(_sessions, _SESSION_KEY, str(uuid.uuid4()))
    return getattr(_sessions, _SESSION_KEY)
