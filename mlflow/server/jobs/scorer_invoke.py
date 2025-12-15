"""
Huey job function for async scorer invocation.

Design decision: One job per trace (not one job per batch), but structured with
trace_ids as a list to support future batching.

Credential freshness is guaranteed because each job:
1. Is short-lived (single trace evaluation)
2. Fetches credentials at job start
3. Never runs long enough for credentials to become stale (~5min TTL)

Additional benefits:
- Parallelism: Multiple workers can process traces concurrently
- Failure isolation: One trace failure doesn't affect others
- Simpler job logic: Each job does one thing
- Extensibility: List structure supports future batching
"""

from typing import Any

from mlflow.server.jobs import job


@job(max_workers=10)
def invoke_scorer_job(
    experiment_id: str,
    trace_ids: list[str],
    serialized_scorer: str,
    persist_results: bool = True,
) -> dict[str, Any]:
    """
    Huey job function for async scorer invocation.

    Currently processes one trace per job, but the list structure supports
    future batching for better throughput.

    Args:
        experiment_id: The experiment ID containing the trace.
        trace_ids: List of trace IDs to evaluate (currently expects single trace).
        serialized_scorer: JSON string of the serialized scorer.
        persist_results: Whether to persist assessment results.

    Returns:
        Dict with trace_ids, success, assessments (or error info).
    """
    import json
    import os

    # Save the tracking URI BEFORE importing handlers, because _get_tracking_store()
    # calls set_tracking_uri() which overwrites os.environ with the backend store URI.
    # Server-side jobs need the HTTP URL to reach the gateway endpoint.
    gateway_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    from mlflow.genai.scorers.base import Scorer
    from mlflow.server.handlers import _execute_scorer_on_trace, _get_tracking_store

    # Deserialize scorer
    scorer_dict = json.loads(serialized_scorer)
    scorer = Scorer.model_validate(scorer_dict)

    # For now, process single trace (first in list)
    trace_id = trace_ids[0]

    # Fetch single trace - this calls set_tracking_uri() which overwrites os.environ
    tracking_store = _get_tracking_store()

    # Restore the tracking URI AFTER _get_tracking_store() so gateway routing works
    if gateway_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = gateway_tracking_uri
    traces = tracking_store.batch_get_traces(trace_ids=[trace_id])

    if not traces:
        return {
            "trace_ids": trace_ids,
            "success": False,
            "error_message": f"Trace not found: {trace_id}",
            "error_type": "TraceNotFoundError",
            "assessments": [],
        }

    trace = traces[0]

    # Execute scorer on single trace
    result = _execute_scorer_on_trace(
        scorer=scorer,
        trace=trace,
        persist_results=persist_results,
    )

    # Ensure result uses trace_ids (list) instead of trace_id (string)
    result["trace_ids"] = trace_ids
    result.pop("trace_id", None)
    return result
