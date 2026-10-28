import logging
import os
from typing import Any

import mlflow
from mlflow.client import MlflowClient
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import (
    MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS,
    MLFLOW_SERVER_SCHEDULED_ISSUE_DETECTION_MAX_WORKERS,
)
from mlflow.exceptions import MlflowException
from mlflow.genai.discovery.checkpoint import (
    MAX_SCHEDULED_SESSION_SAMPLE_SIZE,
    CheckpointTrace,
    SampleWindow,
    advance_checkpoint,
    read_detection_checkpoint,
    write_detection_checkpoint,
)
from mlflow.genai.discovery.constants import DEFAULT_MODEL
from mlflow.genai.discovery.pipeline import discover_issues
from mlflow.genai.discovery.sampling import sample_traces
from mlflow.server.jobs import job
from mlflow.server.jobs.utils import MLFLOW_SERVER_JOB_ID_ENV_VAR
from mlflow.store.tracking import MAX_TRACE_LINKS_PER_REQUEST
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.mlflow_tags import (
    MLFLOW_ISSUE_DETECTION_JOB_ID,
    MLFLOW_ISSUE_DETECTION_TRIGGER,
    MLFLOW_ISSUE_DETECTION_TRIGGER_SCHEDULE,
    MLFLOW_RUN_TYPE,
    MLFLOW_RUN_TYPE_ISSUE_DETECTION,
)
from mlflow.utils.providers import _CORE_PROVIDER_ENV_VARS

_logger = logging.getLogger(__name__)


def _fetch_provider_credentials(
    store: SqlAlchemyStore, provider: str, secret_id: str
) -> dict[str, str]:
    """
    Retrieve and extract provider credentials from secret.

    Args:
        provider: The provider name (e.g., "openai", "anthropic").
        secret_id: The ID of the secret containing the credentials.
    """
    env_var_config = _CORE_PROVIDER_ENV_VARS.get(provider.lower())
    if env_var_config is None:
        raise MlflowException(
            f"Unknown provider '{provider}'. "
            f"Supported providers: {', '.join(_CORE_PROVIDER_ENV_VARS.keys())}. "
            "To use other providers, create an AI Gateway endpoint instead."
        )

    secret_value = store._get_decrypted_secret(secret_id)
    auth_config = store.get_secret_info(secret_id=secret_id).auth_config or {}
    secret_dict = secret_value | auth_config
    credentials = {}
    if isinstance(env_var_config, dict):
        for secret_key, env_var_name in env_var_config.items():
            if value := secret_dict.get(secret_key):
                credentials[env_var_name] = value
    else:
        if api_key := secret_dict.get("api_key"):
            credentials[env_var_config] = api_key

    return credentials


def _newest_trace_matching(experiment_id: str, filter_string: str) -> CheckpointTrace | None:
    """
    The newest trace matching ``filter_string``, or ``None`` when nothing matches.

    This is what the watermark advances to, so it is read directly rather than taken from the
    sample: the sample is random and drops whole sessions, and a watermark taken from it would
    leave the traces it skipped to be re-covered by the next occurrence.
    """
    traces = mlflow.search_traces(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["timestamp_ms DESC"],
        max_results=1,
        return_type="list",
    )
    if not traces:
        return None
    return CheckpointTrace(trace_id=traces[0].info.trace_id, time_ms=traces[0].info.request_time)


def _detect_issues_into_run(
    client: MlflowClient,
    experiment_id: str,
    trace_ids: list[str],
    categories: list[str],
    run_id: str,
    model: str | None,
    instructions: str | None = None,
) -> dict[str, Any]:
    """
    Link the traces to the run, analyze them, and finish the run either way.

    Shared by the manual and the scheduled path so both record a detection run the same way; they
    differ only in how they arrived at ``trace_ids``.
    """
    try:
        # Cannot link more than MAX_TRACE_LINKS_PER_REQUEST traces to a run in a single request
        for i in range(0, len(trace_ids), MAX_TRACE_LINKS_PER_REQUEST):
            batch = trace_ids[i : i + MAX_TRACE_LINKS_PER_REQUEST]
            client.link_traces_to_run(batch, run_id)
        traces = client._tracing_client.batch_get_traces(trace_ids)
        result = discover_issues(
            experiment_id=experiment_id,
            traces=traces,
            model=model,
            run_id=run_id,
            categories=categories,
            instructions=instructions,
        )
        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED))
        client.set_tag(run_id, "total_cost_usd", result.total_cost_usd)
        return {
            "summary": result.summary,
            "issues": len(result.issues),
            "total_traces_analyzed": result.total_traces_analyzed,
            "total_cost_usd": result.total_cost_usd,
        }
    except Exception:
        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FAILED))
        raise


@job(name="invoke_issue_detection", max_workers=MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS.get())
def invoke_issue_detection_job(
    experiment_id: str,
    trace_ids: list[str],
    categories: list[str],
    run_id: str,
    model: str | None = None,
    instructions: str | None = None,
):
    """
    Job function to run issue detection on traces.

    Note: Provider credentials should be passed via environment variables set by submit_job().
    """
    return _detect_issues_into_run(
        MlflowClient(), experiment_id, trace_ids, categories, run_id, model, instructions
    )


def _create_scheduled_run(
    client: MlflowClient,
    experiment_id: str,
    categories: list[str],
    model: str | None,
    total_traces: int,
) -> str:
    """
    Create the run that records one scheduled occurrence.

    Every occurrence that fires records one, including an occurrence that found nothing to analyze:
    without a run there is no way for an experiment's owner to tell a schedule that fired over a
    quiet window from one that never fired at all.
    """
    tags = {
        MLFLOW_RUN_TYPE: MLFLOW_RUN_TYPE_ISSUE_DETECTION,
        MLFLOW_ISSUE_DETECTION_TRIGGER: MLFLOW_ISSUE_DETECTION_TRIGGER_SCHEDULE,
        "categories": ",".join(categories),
        "model": model or DEFAULT_MODEL,
        # Stringified because create_run stores tag values verbatim, unlike the fluent
        # start_run the manual path uses, which coerces them.
        "total_traces": str(total_traces),
    }
    # The same tag the manual path sets from the submitting handler. Without it the Issues tab
    # cannot poll this run's job: it treats a RUNNING detection run carrying no job id as a
    # submission that never started, so a scheduled run would vanish from the tab after a couple
    # of minutes and let someone start a second one over the top of it. Read from the environment
    # because nothing passes the id into a running job.
    if job_id := os.environ.get(MLFLOW_SERVER_JOB_ID_ENV_VAR):
        tags[MLFLOW_ISSUE_DETECTION_JOB_ID] = job_id
    return client.create_run(experiment_id=experiment_id, tags=tags).info.run_id


def _record_quiet_occurrence(
    client: MlflowClient, experiment_id: str, categories: list[str], model: str | None
) -> str:
    """
    Record an occurrence that had nothing to analyze: a finished run over zero traces.

    A schedule that fired over a quiet window and one that never fired are otherwise
    indistinguishable. Nothing is linked and the model is never called, so this costs nothing
    beyond the run itself, and it carries the same tags as any other scheduled run -- a
    ``total_traces`` of zero is what tells them apart.
    """
    run_id = _create_scheduled_run(client, experiment_id, categories, model, total_traces=0)
    client.set_tag(run_id, "total_cost_usd", 0.0)
    client.set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED))
    return run_id


@job(
    name="scheduled_issue_detection",
    # Sized separately from judge invocation: these occurrences are unattended and spend on a
    # cadence, so an operator may want to bound how many run at once without touching the
    # concurrency of judge calls a user is waiting on.
    max_workers=MLFLOW_SERVER_SCHEDULED_ISSUE_DETECTION_MAX_WORKERS.get(),
    # One scheduled run per experiment at a time, so two occurrences can never publish findings
    # for the same experiment concurrently.
    exclusive=["experiment_id"],
)
def run_scheduled_issue_detection_job(
    experiment_id: str,
    window_lower_bound_ms: int,
    window_upper_bound_ms: int,
    categories: list[str],
    instructions: str | None = None,
    trace_filter: str = "",
    model: str | None = None,
) -> dict[str, Any]:
    """
    Run one scheduled occurrence over the traces in its window.

    Unlike the manual path, which is handed the traces a user selected, this samples its own from
    the window, and records a run either way -- see ``_record_quiet_occurrence`` for why an
    occurrence that found nothing still leaves one.

    The watermark advances only on success, and only as far as ``advance_checkpoint`` allows, so a
    failed run leaves the next occurrence to cover this window again.

    The experiment's saved detection priorities -- its category selection and its free-text
    instructions -- are read by the dispatcher and handed over here, since there is no browser to
    send them, so a scheduled run analyzes an experiment the way its owner configured it. Its
    credentials arrive the same way, resolved by the dispatcher from the model's provider.
    """
    client = MlflowClient()
    # The bounds arrive fixed at dispatch time, but this job may have waited behind another
    # occurrence for the same experiment -- ``exclusive`` serializes them without refreshing what
    # they were told to cover. So the lower bound is raised to the watermark as it stands now:
    # whatever ran first has already covered up to there, and re-analyzing it would spend on the
    # same traces twice and re-derive findings cross-run matching then has to discard.
    #
    # The upper bound is left alone. It is this occurrence's fire time, which is what the
    # dispatcher decided it covers; moving it would let a delayed job swallow the next
    # occurrence's window.
    covered = read_detection_checkpoint(client.get_experiment(experiment_id))
    lower_bound_ms = max(
        window_lower_bound_ms,
        covered.last_scanned_trace_time_ms if covered is not None else window_lower_bound_ms,
    )
    if lower_bound_ms > window_lower_bound_ms:
        _logger.info(
            "Scheduled issue detection for experiment %s raised its window lower bound from %d "
            "to the watermark at %d: another occurrence covered that stretch while this one was "
            "queued",
            experiment_id,
            window_lower_bound_ms,
            lower_bound_ms,
        )
    window = SampleWindow(lower_bound_ms=lower_bound_ms, now_ms=window_upper_bound_ms)
    window_clause = window.to_filter_clause()

    # Deliberately the window alone, without the schedule's clauses: they scope what this
    # occurrence analyzes, not how far it covered. Applying them here too would stall the
    # watermark on every window that holds traces but none matching, leaving each later
    # occurrence to re-scan the same stretch until the stale cap cut it off.
    newest = _newest_trace_matching(experiment_id, window_clause)
    if newest is None:
        run_id = _record_quiet_occurrence(client, experiment_id, categories, model)
        _logger.info(
            "Scheduled issue detection found no traces for experiment %s in %s; recorded run %s "
            "over zero traces and left its watermark where it is",
            experiment_id,
            window_clause,
            run_id,
        )
        return {"traces_in_window": 0, "run_id": run_id}

    filter_string = " AND ".join(filter(None, [window_clause, trace_filter]))
    # Whole sessions, so a multi-turn conversation is analyzed together rather than cut in half.
    # The session count bounds the traces too -- see MAX_SCHEDULED_SESSION_SAMPLE_SIZE.
    traces = sample_traces(
        MAX_SCHEDULED_SESSION_SAMPLE_SIZE,
        search_kwargs={"experiment_ids": [experiment_id], "filter_string": filter_string},
    )
    if not traces:
        # Nothing matched, so there is nothing to analyze -- but the occurrence did cover its
        # window, so the watermark still moves past it.
        run_id = _record_quiet_occurrence(client, experiment_id, categories, model)
        _logger.info(
            "Scheduled issue detection sampled no traces for experiment %s in %s; recorded run %s "
            "over zero traces and advanced its watermark across the window regardless",
            experiment_id,
            filter_string,
            run_id,
        )
        if checkpoint := advance_checkpoint(newest, run_id=run_id):
            write_detection_checkpoint(client, experiment_id, checkpoint)
        return {"traces_in_window": 0, "run_id": run_id}

    run_id = _create_scheduled_run(client, experiment_id, categories, model, len(traces))
    result = _detect_issues_into_run(
        client,
        experiment_id,
        [trace.info.trace_id for trace in traces],
        categories,
        run_id,
        model,
        instructions,
    )
    # Only now that the run has finished does the window count as covered.
    if checkpoint := advance_checkpoint(newest, run_id=run_id):
        write_detection_checkpoint(client, experiment_id, checkpoint)
    return result | {"traces_in_window": len(traces), "run_id": run_id}
