"""
Huey job functions for async GenAI evaluation runs.

This is intentionally UI-focused / best-effort and mirrors the existing scorer invocation job
in `mlflow.genai.scorers.job`.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import pandas as pd

import mlflow
from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import Scorer
from mlflow.server.handlers import _get_job_store, _get_tracking_store
from mlflow.server.jobs import job
from mlflow.tracing.utils import parse_trace_id_v4

_logger = logging.getLogger(__name__)


def _resolve_scorers(
    *,
    experiment_id: str,
    judges: list[dict[str, Any]],
    endpoint_name: str | None = None,
) -> list[Scorer]:
    """
    Resolve scorer specs coming from the UI into Scorer objects.

    Supported judge specs:
      - {"type": "registered", "name": "<scorer_name>"}  # from Judges tab
      - {"type": "builtin", "name": "<ClassName>"}       # from mlflow.genai.scorers.<ClassName>
      - {"type": "deepeval", "name": "<MetricName>"}     # from mlflow.genai.scorers.deepeval
      - {"type": "ragas", "name": "<MetricName>"}        # from mlflow.genai.scorers.ragas
      - {"type": "phoenix", "name": "<MetricName>"}      # from mlflow.genai.scorers.phoenix

    Args:
        experiment_id: The experiment ID for loading registered scorers.
        judges: List of judge specifications with type and name.
        endpoint_name: Optional gateway endpoint name for judges to use.
            If provided, scorers will be instantiated with model="gateway:/<endpoint_name>".
    """
    tracking_store = _get_tracking_store()
    scorers: list[Scorer] = []

    for judge in judges:
        jtype = (judge or {}).get("type")
        name = (judge or {}).get("name")
        if not jtype or not name:
            raise MlflowException(f"Invalid judge spec: {judge!r}")

        if jtype == "registered":
            scorer_version = tracking_store.get_scorer(experiment_id, name, None)
            serialized = getattr(scorer_version, "_serialized_scorer", None)
            if not serialized:
                raise MlflowException(f"Failed to load registered judge {name!r}")
            scorers.append(Scorer.model_validate_json(serialized))
        elif jtype == "builtin":
            from mlflow.genai import scorers as builtin_scorers_module

            cls = getattr(builtin_scorers_module, name, None)
            if cls is None:
                raise MlflowException(f"Unknown builtin judge {name!r}")
            # Pass model if endpoint specified, otherwise use default
            if endpoint_name:
                scorers.append(cls(model=f"gateway:/{endpoint_name}"))
            else:
                scorers.append(cls())
        elif jtype == "deepeval":
            try:
                from mlflow.genai.scorers.deepeval import get_scorer as get_deepeval_scorer
            except ImportError as e:
                raise MlflowException(
                    "DeepEval scorers require the 'deepeval' package. "
                    "Install with: pip install deepeval"
                ) from e
            model = f"gateway:/{endpoint_name}" if endpoint_name else None
            scorers.append(get_deepeval_scorer(name, model=model))
        elif jtype == "ragas":
            try:
                from mlflow.genai.scorers.ragas import get_scorer as get_ragas_scorer
            except ImportError as e:
                raise MlflowException(
                    "RAGAS scorers require the 'ragas' package. Install with: pip install ragas"
                ) from e
            model = f"gateway:/{endpoint_name}" if endpoint_name else None
            scorers.append(get_ragas_scorer(name, model=model))
        elif jtype == "phoenix":
            try:
                from mlflow.genai.scorers.phoenix import get_scorer as get_phoenix_scorer
            except ImportError as e:
                raise MlflowException(
                    "Phoenix scorers require the 'arize-phoenix-evals' package. "
                    "Install with: pip install arize-phoenix-evals"
                ) from e
            model = f"gateway:/{endpoint_name}" if endpoint_name else None
            scorers.append(get_phoenix_scorer(name, model=model))
        else:
            raise MlflowException(f"Unknown judge type {jtype!r}")

    return scorers


def _find_own_job_id(job_name: str, client_job_uuid: str) -> str | None:
    """
    Hack: job functions don't receive job_id, so we pass a unique client_job_uuid and
    look ourselves up in the job store.
    """
    try:
        job_store = _get_job_store()
        candidates = list(
            job_store.list_jobs(
                job_name=job_name,
                statuses=[JobStatus.PENDING, JobStatus.RUNNING],
                params={"client_job_uuid": client_job_uuid},
            )
        )
        if not candidates:
            return None
        # Should be unique. If not, pick the newest.
        candidates.sort(key=lambda j: j.creation_time, reverse=True)
        return candidates[0].job_id
    except Exception:
        _logger.debug("Failed to locate job id for progress updates", exc_info=True)
        return None


def _update_job_progress(job_id: str | None, payload: dict[str, Any]) -> None:
    if not job_id:
        return
    try:
        job_store = _get_job_store()
        # NOTE: `_update_job` is intentionally used here (even though it's "private") so we can
        # stream progress while status is RUNNING.
        if hasattr(job_store, "_update_job"):
            job_store._update_job(job_id, JobStatus.RUNNING, result=json.dumps(payload))  # type: ignore[attr-defined]
    except Exception:
        _logger.debug("Failed updating job progress", exc_info=True)


@job(name="evaluate_traces", max_workers=1)
def evaluate_traces_job(
    experiment_id: str,
    trace_ids: list[str],
    judges: list[dict[str, Any]],
    run_name: str | None = None,
    run_id: str | None = None,
    client_job_uuid: str | None = None,
    endpoint_name: str | None = None,
) -> dict[str, Any]:
    """
    Create an MLflow evaluation run by running `mlflow.genai.evaluate` over a list of traces.

    Returns a JSON-serializable dict containing the created run_id.
    """
    if not client_job_uuid:
        client_job_uuid = str(uuid.uuid4())

    job_id = _find_own_job_id("evaluate_traces", client_job_uuid)
    total = len(trace_ids)
    initial_payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "progress": {"completed": 0, "total": total},
    }
    if run_id:
        initial_payload["run_id"] = run_id
    _update_job_progress(job_id, initial_payload)

    if not experiment_id:
        raise MlflowException("Missing required parameter: experiment_id")
    if not trace_ids:
        raise MlflowException("Please select at least one trace to evaluate.")
    if not judges:
        raise MlflowException("Missing required parameter: judges")

    # Ensure tracing APIs (e.g. `mlflow.search_traces`) default to the same experiment we're
    # evaluating, instead of falling back to the "default" experiment (often "0").
    # This matters for some judge tooling / evaluation internals that call
    # `search_traces(run_id=...)` without explicitly specifying `locations`.
    mlflow.set_experiment(experiment_id=experiment_id)

    # UI may pass v4-style trace IDs like: "trace:/<location>/<trace_id>"
    # while the SQLAlchemy tracing store expects the underlying v3 trace id ("tr-...").
    normalized_trace_ids: list[str] = []
    for tid in trace_ids:
        _, parsed_tid = parse_trace_id_v4(tid)
        if not parsed_tid:
            raise MlflowException(f"Invalid trace id: {tid!r}")
        normalized_trace_ids.append(parsed_tid)
    trace_ids = normalized_trace_ids

    tracking_store = _get_tracking_store()
    traces = tracking_store.batch_get_traces(trace_ids)
    trace_map = {t.info.trace_id: t for t in traces}
    if missing := [tid for tid in trace_ids if tid not in trace_map]:
        raise MlflowException(f"Traces not found: {missing}")

    ordered_traces = [trace_map[tid] for tid in trace_ids]
    eval_df = pd.DataFrame({"trace": ordered_traces})

    scorers = _resolve_scorers(
        experiment_id=experiment_id, judges=judges, endpoint_name=endpoint_name
    )

    completed_counter = {"n": 0}
    # If UI pre-created a run, preserve that id from the beginning.
    run_id_holder: dict[str, str | None] = {"run_id": run_id}

    def progress_callback(completed: int, total_items: int) -> None:
        # "completed" counts single-turn eval items in the harness
        completed_counter["n"] = completed
        payload: dict[str, Any] = {
            "experiment_id": experiment_id,
            "progress": {"completed": completed, "total": total_items},
        }
        # IMPORTANT: Keep run_id in every update once available. The UI polls periodically and
        # may miss a one-off update that includes run_id.
        if run_id_holder["run_id"]:
            payload["run_id"] = run_id_holder["run_id"]
        _update_job_progress(
            job_id,
            payload,
        )

    # If a run_id is provided, reuse that run (created by the UI) so all evaluation
    # artifacts/metrics are logged to the intended run. Otherwise, create a new run.
    start_run_kwargs: dict[str, Any]
    if run_id:
        start_run_kwargs = {"run_id": run_id}
    else:
        start_run_kwargs = {"experiment_id": experiment_id, "run_name": run_name}

    with mlflow.start_run(**start_run_kwargs) as run:
        # If the run was newly created, publish it immediately.
        if not run_id_holder["run_id"]:
            run_id_holder["run_id"] = run.info.run_id
            _update_job_progress(
                job_id,
                {
                    "run_id": run.info.run_id,
                    "experiment_id": experiment_id,
                    "progress": {"completed": completed_counter["n"], "total": total},
                },
            )
        mlflow.genai.evaluate(
            data=eval_df,
            scorers=scorers,
            progress_callback=progress_callback,
        )

        # Ensure we end at 100% even if callback wasn't invoked (e.g. edge cases)
        _update_job_progress(
            job_id,
            {
                "run_id": run_id_holder["run_id"] or run.info.run_id,
                "experiment_id": experiment_id,
                "progress": {"completed": total, "total": total},
            },
        )
        return {
            "run_id": run_id_holder["run_id"] or run.info.run_id,
            "experiment_id": experiment_id,
            "progress": {"completed": total, "total": total},
        }
