import hashlib
import threading
import uuid

import mlflow
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.builtin_scorers import BuiltInScorer
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import _REST_API_PATH_PREFIX, http_request
from mlflow.utils.uri import is_databricks_uri
from mlflow.version import VERSION

_SESSION_ID_HEADER = "eval-session-id"
_BATCH_SIZE_HEADER = "eval-session-batch-size"
_CLIENT_VERSION_HEADER = "eval-session-client-version"
_CLIENT_NAME_HEADER = "eval-session-client-name"

_EVAL_TELEMETRY_ENDPOINT = f"{_REST_API_PATH_PREFIX}/agents/evaluation-client-usage-events"


_sessions = threading.local()
_SESSION_KEY = "genai-eval-session"


def emit_metric_usage_event(
    scorers: list[Scorer],
    trace_count: int | None,
    session_count: int | None,
    aggregated_metrics: dict[str, float],
):
    """Emit usage events for custom and built-in scorers when running on Databricks"""
    if not is_databricks_uri(mlflow.get_tracking_uri()):
        return

    custom_metrics = [s for s in scorers if _is_custom_scorer(s)]
    builtin_metrics = [s for s in scorers if not _is_custom_scorer(s)]

    events = []

    if custom_metrics:
        metric_name_to_hash = {m.name: _hash_metric_name(m.name) for m in custom_metrics}

        metric_stats = {
            hashed_name: {
                "average": None,
                "count": session_count
                if getattr(scorer, "is_session_level_scorer", False)
                else trace_count,
            }
            for scorer, hashed_name in zip(custom_metrics, metric_name_to_hash.values())
        }
        for metric_key, metric_value in aggregated_metrics.items():
            name, aggregation = metric_key.split("/", 1)
            hashed_name = metric_name_to_hash.get(name)
            if hashed_name is not None and aggregation == "mean":
                metric_stats[hashed_name]["average"] = metric_value

        metric_stats = [
            {
                "name": hashed_name,
                "average": metric_stats[hashed_name]["average"],
                "count": metric_stats[hashed_name]["count"],
            }
            for hashed_name in metric_stats
        ]

        events.append(
            {
                "custom_metric_usage_event": {
                    "eval_count": trace_count,
                    "metrics": metric_stats,
                }
            }
        )

    if builtin_metrics:
        builtin_stats = [
            {
                "name": type(scorer).__name__,
                "count": session_count if scorer.is_session_level_scorer else trace_count,
            }
            for scorer in builtin_metrics
        ]

        events.append(
            {
                "builtin_scorer_usage_event": {
                    "metrics": builtin_stats,
                }
            }
        )

    if not events:
        return

    payload = {"agent_evaluation_client_usage_events": events}

    extra_headers = {
        _CLIENT_VERSION_HEADER: VERSION,
        _SESSION_ID_HEADER: _get_or_create_session_id(),
        _BATCH_SIZE_HEADER: str(trace_count),
        _CLIENT_NAME_HEADER: "mlflow",
    }

    try:
        from databricks.rag_eval.utils import request_utils

        extra_headers = request_utils.add_traffic_id_header(extra_headers)
    except ImportError:
        pass

    http_request(
        host_creds=get_databricks_host_creds(),
        endpoint=_EVAL_TELEMETRY_ENDPOINT,
        method="POST",
        extra_headers=extra_headers,
        json=payload,
    )


def _get_or_create_session_id() -> str:
    if not hasattr(_sessions, _SESSION_KEY):
        setattr(_sessions, _SESSION_KEY, str(uuid.uuid4()))
    return getattr(_sessions, _SESSION_KEY)


def _is_custom_scorer(scorer) -> bool:
    if isinstance(scorer, Scorer):
        return not isinstance(scorer, BuiltInScorer)

    # Check for the legacy custom metrics if databricks-agents is installed
    try:
        from databricks.rag_eval.evaluation.custom_metrics import CustomMetric

        return isinstance(scorer, CustomMetric)
    except ImportError:
        pass

    # Treat unknown case as not a custom scorer
    return False


def _hash_metric_name(metric_name: str) -> str:
    """Hash metric name in un-recoverable way to avoid leaking sensitive information"""
    return hashlib.sha256(metric_name.encode("utf-8")).hexdigest()
