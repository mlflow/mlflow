from typing import Annotated

from pydantic import Field

from mlflow.entities import LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.mcp.request_context import get_mcp_request_username, is_mcp_http_request
from mlflow.mcp.tools._args import OrderBy, PageToken, View, as_list, as_tag_dict, as_view_type
from mlflow.mcp.tools._types import (
    CreatedRun,
    LinkedTraces,
    RunDetails,
    RunPage,
    RunRef,
    RunSummary,
)
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NOTE, MLFLOW_USER

_TERMINAL_STATUSES = ("FINISHED", "FAILED", "KILLED")


def list_runs(
    experiment_id: Annotated[str, Field(description="ID of the experiment to list runs of.")],
    view: View = "active_only",
    max_results: Annotated[
        int, Field(description="Maximum number of runs to return.", ge=1)
    ] = SEARCH_MAX_RESULTS_DEFAULT,
    page_token: PageToken = None,
    filter_string: Annotated[
        str | None,
        Field(description="Search filter, e.g. \"metrics.rmse < 1 and params.model = 'tree'\"."),
    ] = None,
    order_by: OrderBy = None,
) -> RunPage:
    """List the runs of an experiment, newest first unless ``order_by`` is given."""
    page = MlflowClient().search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string or "",
        run_view_type=as_view_type(view),
        max_results=max_results,
        order_by=as_list(order_by),
        page_token=page_token,
    )
    return RunPage(
        runs=[RunSummary.from_entity(run) for run in page],
        next_page_token=page.token or None,
    )


def delete_run(
    run_id: Annotated[str, Field(description="ID of the run to delete.")],
) -> RunRef:
    """
    Mark a run for deletion. A deleted run can be restored with ``restore_run`` until it is
    permanently deleted.
    """
    MlflowClient().delete_run(run_id)
    return RunRef(run_id=run_id)


def restore_run(
    run_id: Annotated[str, Field(description="ID of the run to restore.")],
) -> RunRef:
    """Restore a deleted run."""
    MlflowClient().restore_run(run_id)
    return RunRef(run_id=run_id)


def describe_run(
    run_id: Annotated[str, Field(description="ID of the run to describe.")],
) -> RunDetails:
    """Get a run's details: info, latest metrics, params, tags, inputs and outputs."""
    return RunDetails.from_entity(MlflowClient().get_run(run_id))


def _run_tags(user_tags: dict[str, str]) -> dict[str, str]:
    if not is_mcp_http_request():
        # The stdio server runs on the caller's machine: resolve the same context tags (user,
        # source, git commit) as ``mlflow.start_run``.
        return resolve_tags(user_tags)
    # Served over HTTP the process belongs to the tracking server, so its local context says
    # nothing about the caller. The run is attributed to the authenticated user instead.
    tags = dict(user_tags)
    if (username := get_mcp_request_username()) is not None:
        tags[MLFLOW_USER] = username
    return tags


def create_run(
    experiment_id: Annotated[
        str | None,
        Field(
            description="ID of the experiment to create the run in. Give this or experiment_name."
        ),
    ] = None,
    experiment_name: Annotated[
        str | None,
        Field(
            description="Name of the experiment to create the run in; created when missing. "
            "Give this or experiment_id."
        ),
    ] = None,
    run_name: Annotated[str | None, Field(description="Human-readable name for the run.")] = None,
    description: Annotated[
        str | None, Field(description="Longer description of what the run represents.")
    ] = None,
    tags: Annotated[
        dict[str, str] | list[str] | None,
        Field(description="Run tags as an object, or a list of 'key=value' strings."),
    ] = None,
    status: Annotated[
        str,
        Field(
            description="Final status of the run: FINISHED (default), FAILED or KILLED. "
            "Case-insensitive."
        ),
    ] = "FINISHED",
    parent_run_id: Annotated[
        str | None, Field(description="ID of a parent run to nest the new run under.")
    ] = None,
) -> CreatedRun:
    """
    Create a run and immediately end it with the given status, e.g. to record a completed
    experiment.
    """
    if (experiment_id is None) == (experiment_name is None):
        raise MlflowException.invalid_parameter_value(
            "Must specify exactly one of experiment_id or experiment_name."
        )
    final_status = status.upper()
    if final_status not in _TERMINAL_STATUSES:
        raise MlflowException.invalid_parameter_value(
            f"Invalid status '{status}'. Must be one of {', '.join(_TERMINAL_STATUSES)}."
        )
    user_tags = as_tag_dict(tags)
    if description:
        if MLFLOW_RUN_NOTE in user_tags:
            raise MlflowException.invalid_parameter_value(
                f"Description is already set via the tag {MLFLOW_RUN_NOTE} in tags."
            )
        user_tags[MLFLOW_RUN_NOTE] = description

    client = MlflowClient()
    if parent_run_id is not None:
        parent_run = client.get_run(parent_run_id)
        if parent_run.info.lifecycle_stage == LifecycleStage.DELETED:
            raise MlflowException.invalid_parameter_value(
                f"Cannot create a run under parent run with ID {parent_run_id} because it is in "
                "the deleted state."
            )
        user_tags[MLFLOW_PARENT_RUN_ID] = parent_run_id

    created_experiment = False
    if experiment_name is not None:
        if experiment := client.get_experiment_by_name(experiment_name):
            experiment_id = experiment.experiment_id
        else:
            experiment_id = client.create_experiment(experiment_name)
            created_experiment = True

    run = client.create_run(experiment_id, tags=_run_tags(user_tags), run_name=run_name)
    client.set_terminated(run.info.run_id, status=final_status)
    result = CreatedRun(
        run_id=run.info.run_id,
        experiment_id=run.info.experiment_id,
        run_name=run.info.run_name,
        status=final_status,
    )
    result._created_experiment = created_experiment
    return result


def link_traces_to_run(
    run_id: Annotated[str, Field(description="ID of the run to link the traces to.")],
    trace_ids: Annotated[
        list[str],
        Field(description="IDs of the traces to link (at most 100).", min_length=1),
    ],
) -> LinkedTraces:
    """Link traces to an existing run."""
    MlflowClient().link_traces_to_run(trace_ids, run_id)
    return LinkedTraces(run_id=run_id, trace_ids=trace_ids)
