"""
Utility functions for MLflow server.
"""

import logging
import random
from typing import Any, NamedTuple

from mlflow.entities.trace_state import TraceState

_logger = logging.getLogger(__name__)


class ScorerInfo(NamedTuple):
    """Information needed to identify a scorer."""

    name: str
    version: int


def run_scorers(trace_id: str, experiment_id: str, scorers: list[dict[str, Any]]) -> None:
    """
    Run scorers for a completed trace.

    Args:
        trace_id: The ID of the trace to score.
        experiment_id: The experiment ID containing the trace.
        scorers: List of dicts with 'name' and 'version' keys.
    """
    from mlflow.entities.assessment import Feedback
    from mlflow.entities.trace import Trace
    from mlflow.entities.trace_data import TraceData
    from mlflow.genai.scorers.registry import get_scorer
    from mlflow.server.handlers import _get_trace_artifact_repo, _get_tracking_store

    store = _get_tracking_store()
    trace_info = store.get_trace_info(trace_id)
    trace_data = TraceData.from_dict(_get_trace_artifact_repo(trace_info).download_trace_data())
    trace = Trace(trace_info, trace_data)

    for scorer_dict in scorers:
        scorer_info = ScorerInfo(**scorer_dict)
        scorer = get_scorer(
            experiment_id=experiment_id, name=scorer_info.name, version=scorer_info.version
        )
        try:
            scorer_feedback: Feedback = scorer(trace=trace)
        except Exception:
            # TODO: Log an assessment indicating failure to run a scorers
            pass
        else:
            store.create_assessment(scorer_feedback)


def check_and_submit_scorers_for_trace(trace_id: str, experiment_id: str) -> None:
    """
    Check if a trace is in a terminal state and submit scorer job if so.

    Args:
        trace_id: The ID of the trace to check.
        experiment_id: The experiment ID containing the trace.
    """
    from mlflow.server.handlers import _get_tracking_store
    from mlflow.server.jobs import submit_job

    store = _get_tracking_store()

    # Get the trace info to check its state
    trace_info = store.get_trace_info(trace_id)

    # Check if the trace is in a terminal state
    if trace_info.state in (TraceState.OK, TraceState.ERROR):
        # List all scorers for the experiment
        scorers = store.list_scorers(experiment_id)

        # Filter scorers based on sample rate
        scorers_to_run = []
        for scorer in scorers:
            # Get the sample rate (default to 1.0 if not set)
            sample_rate = getattr(scorer, "sample_rate", 1.0)

            # Use random sampling to determine if this scorer should run
            if random.random() < sample_rate:
                # Create ScorerInfo with just name and version
                scorer_info = ScorerInfo(name=scorer.scorer_name, version=scorer.scorer_version)
                scorers_to_run.append(scorer_info)

        # Only submit job if there are scorers to run
        if scorers_to_run:
            # Submit a job to run scorers asynchronously
            # Convert NamedTuples to dicts for JSON serialization
            submit_job(
                function=run_scorers,
                params={
                    "trace_id": trace_id,
                    "experiment_id": experiment_id,
                    "scorers": [scorer._asdict() for scorer in scorers_to_run],
                },
            )
