import logging
from typing import Any

import mlflow
from mlflow.client import MlflowClient
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS
from mlflow.exceptions import MlflowException
from mlflow.genai.discovery.checkpoint import (
    MAX_SCHEDULED_SAMPLE_SIZE,
    CheckpointTrace,
    SampleWindow,
    advance_checkpoint,
    write_detection_checkpoint,
)
from mlflow.genai.discovery.constants import DEFAULT_MODEL
from mlflow.genai.discovery.pipeline import discover_issues
from mlflow.genai.discovery.sampling import sample_traces
from mlflow.server.jobs import job
from mlflow.store.tracking import MAX_TRACE_LINKS_PER_REQUEST
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.mlflow_tags import (
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


@job(
    name="scheduled_issue_detection",
    max_workers=MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS.get(),
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

    Unlike the manual path, which is handed the traces a user selected and a run created up front,
    this samples its own traces and creates the run only once it knows there is something to
    analyze -- an occurrence over a quiet window should leave no empty run behind.

    The watermark advances only after the run finishes, and to the newest trace in the *window*
    rather than the newest trace sampled: traces this occurrence was entitled to see and did not
    sample are not carried forward for a second look. A failed run leaves the watermark alone, so
    the next occurrence covers this window again.

    The experiment's saved detection priorities -- its category selection and its free-text
    instructions -- are read by the dispatcher and handed over here, since there is no browser to
    send them. A scheduled run therefore analyzes an experiment the way its owner configured it,
    exactly as a manual run does.

    Note: a schedule fires with no user session, so nobody is there to supply credentials. The
    dispatcher resolves them from the provider of the model the schedule names, falling back to
    the credentials in the server's own environment when nothing is stored for it.
    """
    window = SampleWindow(lower_bound_ms=window_lower_bound_ms, now_ms=window_upper_bound_ms)
    window_clause = window.to_filter_clause()

    # Deliberately the window alone, without the schedule's clauses: they scope what this
    # occurrence analyzes, not how far it covered. Applying them here too would stall the
    # watermark on every window that holds traces but none matching, leaving each later
    # occurrence to re-scan the same stretch until the stale cap cut it off.
    newest = _newest_trace_matching(experiment_id, window_clause)
    if newest is None:
        _logger.info(
            "Scheduled issue detection found no traces for experiment %s in %s; "
            "leaving its watermark where it is",
            experiment_id,
            window_clause,
        )
        return {"traces_in_window": 0}

    filter_string = " AND ".join(filter(None, [window_clause, trace_filter]))
    traces = sample_traces(
        MAX_SCHEDULED_SAMPLE_SIZE,
        search_kwargs={"experiment_ids": [experiment_id], "filter_string": filter_string},
    )
    client = MlflowClient()
    if not traces:
        # Nothing matched, so there is nothing to analyze and no run worth leaving behind -- but
        # the occurrence did cover its window, so the watermark still moves past it.
        _logger.info(
            "Scheduled issue detection sampled no traces for experiment %s in %s; "
            "advancing its watermark across the window regardless",
            experiment_id,
            filter_string,
        )
        if checkpoint := advance_checkpoint(newest):
            write_detection_checkpoint(client, experiment_id, checkpoint)
        return {"traces_in_window": 0}

    run = client.create_run(
        experiment_id=experiment_id,
        tags={
            MLFLOW_RUN_TYPE: MLFLOW_RUN_TYPE_ISSUE_DETECTION,
            MLFLOW_ISSUE_DETECTION_TRIGGER: MLFLOW_ISSUE_DETECTION_TRIGGER_SCHEDULE,
            "categories": ",".join(categories),
            "model": model or DEFAULT_MODEL,
            # Stringified because create_run stores tag values verbatim, unlike the fluent
            # start_run the manual path uses, which coerces them.
            "total_traces": str(len(traces)),
        },
    )
    result = _detect_issues_into_run(
        client,
        experiment_id,
        [trace.info.trace_id for trace in traces],
        categories,
        run.info.run_id,
        model,
        instructions,
    )
    # Only now that the run has finished does the window count as covered.
    if checkpoint := advance_checkpoint(newest, run_id=run.info.run_id):
        write_detection_checkpoint(client, experiment_id, checkpoint)
    return result | {"traces_in_window": len(traces), "run_id": run.info.run_id}
