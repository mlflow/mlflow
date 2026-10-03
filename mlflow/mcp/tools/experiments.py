from typing import Annotated

from pydantic import Field

from mlflow.exceptions import MlflowException
from mlflow.mcp.tools._args import (
    DeprecatedOutput,
    OrderBy,
    PageToken,
    View,
    as_list,
    as_view_type,
    check_non_negative,
)
from mlflow.mcp.tools._types import ExperimentInfo, ExperimentPage, ExperimentRef, ExperimentUpdate
from mlflow.protos import databricks_pb2
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.utils.trace_archival import (
    _encode_trace_archival_retention_tag,
    _encode_trace_archive_now_tag,
)
from mlflow.tracing.constant import TraceExperimentTagKey
from mlflow.tracking import MlflowClient
from mlflow.utils.validation import _validate_trace_archival_retention_string


def _validate_duration(value: str | None) -> str | None:
    return None if value is None else _validate_trace_archival_retention_string(value)


def create_experiment(
    experiment_name: Annotated[str, Field(description="Name of the experiment.")],
    artifact_location: Annotated[
        str | None,
        Field(
            description="Base artifact location for the experiment's runs. Server default "
            "when omitted."
        ),
    ] = None,
    trace_archival_retention: Annotated[
        str | None,
        Field(description="Experiment-level trace archival retention, e.g. '30d' or '12h'."),
    ] = None,
) -> ExperimentRef:
    """Create an experiment."""
    tags = None
    if (retention := _validate_duration(trace_archival_retention)) is not None:
        tags = {
            TraceExperimentTagKey.ARCHIVAL_RETENTION: _encode_trace_archival_retention_tag(
                retention
            )
        }
    experiment_id = MlflowClient().create_experiment(experiment_name, artifact_location, tags=tags)
    return ExperimentRef(experiment_id=experiment_id, name=experiment_name)


def update_experiment(
    experiment_id: Annotated[str, Field(description="ID of the experiment to update.")],
    trace_archival_retention: Annotated[
        str | None,
        Field(description="Set the trace archival retention override, e.g. '30d' or '12h'."),
    ] = None,
    clear_trace_archival_retention: Annotated[
        bool, Field(description="Clear the trace archival retention override.")
    ] = False,
    trace_archive_now: Annotated[
        bool, Field(description="Request archive-now processing on the next scheduler pass.")
    ] = False,
    trace_archive_now_older_than: Annotated[
        str | None,
        Field(description="Request archive-now for traces older than this duration."),
    ] = None,
    clear_trace_archive_now: Annotated[
        bool, Field(description="Clear a pending archive-now request.")
    ] = False,
) -> ExperimentUpdate:
    """
    Update experiment trace archival policy controls. These configure or request server-owned
    archival; they do not run archival.
    """
    retention = _validate_duration(trace_archival_retention)
    older_than = _validate_duration(trace_archive_now_older_than)
    if retention is not None and clear_trace_archival_retention:
        raise MlflowException.invalid_parameter_value(
            "Cannot specify both trace_archival_retention and clear_trace_archival_retention."
        )
    if trace_archive_now and older_than is not None:
        raise MlflowException.invalid_parameter_value(
            "Cannot specify both trace_archive_now and trace_archive_now_older_than."
        )
    if clear_trace_archive_now and (trace_archive_now or older_than is not None):
        raise MlflowException.invalid_parameter_value(
            "Cannot specify clear_trace_archive_now together with archive-now requests."
        )
    if not any([
        retention is not None,
        clear_trace_archival_retention,
        trace_archive_now,
        older_than is not None,
        clear_trace_archive_now,
    ]):
        raise MlflowException.invalid_parameter_value("Must specify at least one update option.")

    client = MlflowClient()
    existing_tags = client.get_experiment(experiment_id).tags
    changes = []

    if retention is not None:
        client.set_experiment_tag(
            experiment_id,
            TraceExperimentTagKey.ARCHIVAL_RETENTION,
            _encode_trace_archival_retention_tag(retention),
        )
        changes.append(f"set trace archival retention to {retention}")
    elif clear_trace_archival_retention:
        if TraceExperimentTagKey.ARCHIVAL_RETENTION in existing_tags:
            client.delete_experiment_tag(experiment_id, TraceExperimentTagKey.ARCHIVAL_RETENTION)
            changes.append("cleared trace archival retention override")
        else:
            changes.append("trace archival retention override was already unset")

    if trace_archive_now or older_than is not None:
        client.set_experiment_tag(
            experiment_id,
            TraceExperimentTagKey.ARCHIVE_NOW,
            _encode_trace_archive_now_tag(older_than),
        )
        if older_than is None:
            changes.append("requested archive-now on the next scheduler pass")
        else:
            changes.append(
                f"requested archive-now for traces older than {older_than} on the next "
                "scheduler pass"
            )
    elif clear_trace_archive_now:
        if TraceExperimentTagKey.ARCHIVE_NOW in existing_tags:
            client.delete_experiment_tag(experiment_id, TraceExperimentTagKey.ARCHIVE_NOW)
            changes.append("cleared pending archive-now request")
        else:
            changes.append("archive-now request was already unset")

    return ExperimentUpdate(experiment_id=experiment_id, changes=changes)


def search_experiments(
    view: View = "active_only",
    max_results: Annotated[
        int | None,
        Field(
            description="Maximum number of experiments to return. Every experiment when omitted."
        ),
    ] = None,
    page_token: PageToken = None,
    filter_string: Annotated[
        str | None,
        Field(description="Search filter, e.g. \"name LIKE 'prod-%'\" or \"tags.team = 'a'\"."),
    ] = None,
    order_by: OrderBy = None,
) -> ExperimentPage:
    """Search for experiments in the configured tracking server."""
    check_non_negative(max_results, "max_results")
    # The stores reject a page size of 0; an empty page that resumes where it started matches
    # the filtered search non-admin callers get over the tracking server.
    if max_results == 0:
        return ExperimentPage(experiments=[], next_page_token=page_token or None)
    client = MlflowClient()
    view_type = as_view_type(view)
    order_by_list = as_list(order_by)

    if max_results is not None:
        page = client.search_experiments(
            view_type=view_type,
            max_results=max_results,
            filter_string=filter_string,
            order_by=order_by_list,
            page_token=page_token,
        )
        return ExperimentPage(
            experiments=[ExperimentInfo.from_entity(e) for e in page],
            next_page_token=page.token or None,
        )

    experiments = []
    while True:
        page = client.search_experiments(
            view_type=view_type,
            max_results=SEARCH_MAX_RESULTS_DEFAULT,
            filter_string=filter_string,
            order_by=order_by_list,
            page_token=page_token,
        )
        experiments.extend(ExperimentInfo.from_entity(e) for e in page)
        page_token = page.token
        if not page_token:
            break
    return ExperimentPage(experiments=experiments, next_page_token=None)


def get_experiment(
    experiment_id: Annotated[
        str | None, Field(description="ID of the experiment. Give this or experiment_name.")
    ] = None,
    experiment_name: Annotated[
        str | None, Field(description="Name of the experiment. Give this or experiment_id.")
    ] = None,
    output: DeprecatedOutput = None,
) -> ExperimentInfo:
    """
    Get an experiment by ID or name: name, artifact location, lifecycle stage, tags, creation
    and last update time.
    """
    if (experiment_id is None) == (experiment_name is None):
        raise MlflowException.invalid_parameter_value(
            "Must specify exactly one of experiment_id or experiment_name."
        )
    client = MlflowClient()
    if experiment_id is not None:
        experiment = client.get_experiment(experiment_id)
    else:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise MlflowException(
                f"Experiment with name '{experiment_name}' does not exist.",
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
    return ExperimentInfo.from_entity(experiment)


def delete_experiment(
    experiment_id: Annotated[str, Field(description="ID of the experiment to delete.")],
) -> ExperimentRef:
    """
    Mark an active experiment, its runs and their data for deletion. Deleted experiments can be
    restored with ``restore_experiment`` until they are permanently deleted.
    """
    MlflowClient().delete_experiment(experiment_id)
    return ExperimentRef(experiment_id=experiment_id)


def restore_experiment(
    experiment_id: Annotated[str, Field(description="ID of the experiment to restore.")],
) -> ExperimentRef:
    """Restore a deleted experiment, its runs and their data."""
    MlflowClient().restore_experiment(experiment_id)
    return ExperimentRef(experiment_id=experiment_id)


def rename_experiment(
    experiment_id: Annotated[str, Field(description="ID of the experiment to rename.")],
    new_name: Annotated[str, Field(description="New name for the experiment.")],
) -> ExperimentRef:
    """Rename an active experiment."""
    MlflowClient().rename_experiment(experiment_id, new_name)
    return ExperimentRef(experiment_id=experiment_id, name=new_name)
