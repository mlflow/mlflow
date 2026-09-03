"""Huey job function for the UI-triggered `mlflow.genai.evaluate` flow.

This module backs the `POST /ajax-api/3.0/mlflow/genai/evaluate/invoke` endpoint
used by the "Run evaluation" modal's "Run judges" button.
"""

import logging
import os

import mlflow
from mlflow.client import MlflowClient
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import (
    _MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN,
    MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS,
)
from mlflow.genai.scorers.base import SCORER_BACKEND_TRACKING, Scorer
from mlflow.server.jobs import job
from mlflow.store.tracking import MAX_TRACE_LINKS_PER_REQUEST

_logger = logging.getLogger(__name__)


@job(name="invoke_genai_evaluate", max_workers=MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS.get())
def invoke_genai_evaluate_job(
    trace_ids: list[str],
    serialized_scorers: list[str],
    run_id: str,
    username: str | None = None,
    scorer_versions: list[int | None] | None = None,
    experiment_id: str | None = None,
):
    """
    Run `mlflow.genai.evaluate` against a fixed set of traces and scorers,
    writing all outputs (dataset input, assessments, aggregate metrics) into
    the pre-existing run identified by `run_id`.
    """
    if username is not None:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        if internal_token := _MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.get():
            os.environ["MLFLOW_TRACKING_PASSWORD"] = internal_token

    client = MlflowClient()

    try:
        for i in range(0, len(trace_ids), MAX_TRACE_LINKS_PER_REQUEST):
            client.link_traces_to_run(trace_ids[i : i + MAX_TRACE_LINKS_PER_REQUEST], run_id)
        traces = client._tracing_client.batch_get_traces(trace_ids)
        scorers = [Scorer.model_validate_json(s) for s in serialized_scorers]
        if scorer_versions is not None:
            if len(scorer_versions) != len(scorers):
                raise ValueError("scorer_versions must have the same length as serialized_scorers")
            for scorer, scorer_version in zip(scorers, scorer_versions, strict=True):
                if scorer_version is not None:
                    scorer._set_registration_metadata(
                        backend=SCORER_BACKEND_TRACKING,
                        experiment_id=experiment_id,
                        sampling_config=None,
                        scorer_version=scorer_version,
                    )
    except Exception:
        _logger.exception("genai evaluate job failed during setup for run %s", run_id)
        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FAILED))
        raise

    with mlflow.start_run(run_id=run_id):
        mlflow.genai.evaluate(data=traces, scorers=scorers)

    return {
        "run_id": run_id,
        "total_traces": len(trace_ids),
        "scorer_count": len(scorers),
    }
