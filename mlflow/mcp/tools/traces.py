from typing import Annotated, Any, Literal

from pydantic import Field

from mlflow.entities import AssessmentSource, Expectation, Feedback
from mlflow.exceptions import MlflowException
from mlflow.mcp.tools._args import (
    DeprecatedOutput,
    PageToken,
    as_string_map,
    as_json_value,
    as_list,
)
from mlflow.mcp.tools._types import (
    AssessmentInfo,
    AssessmentRef,
    DeletedTraces,
    TracePage,
    TraceResult,
    TraceTag,
)
from mlflow.store.tracking import SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.tracing.assessment import log_expectation, log_feedback
from mlflow.tracing.client import TracingClient
from mlflow.utils.jsonpath_utils import filter_json_by_fields, validate_field_paths

TraceId = Annotated[str, Field(description="ID of the trace, e.g. 'tr-1234567890abcdef'.")]
AssessmentId = Annotated[str, Field(description="ID of the assessment.")]
ExtractFields = Annotated[
    list[str] | str | None,
    Field(
        description="Fields to keep, in dot notation: e.g. ['info.trace_id', "
        "'info.assessments.*', 'data.spans.*.name']. Use backticks for keys containing dots: "
        "'info.tags.`mlflow.traceName`'. A comma-separated string is also accepted. Every field "
        "is returned when omitted."
    ),
]
Verbose = Annotated[
    bool, Field(description="List every available field when a requested field is invalid.")
]
SourceType = Annotated[
    Literal["HUMAN", "LLM_JUDGE", "CODE"] | None,
    Field(description="Source type of the assessment. Used together with source_id."),
]
SourceId = Annotated[
    str | None,
    Field(description="Source identifier, e.g. an email for HUMAN or a model name for LLM_JUDGE."),
]
Metadata = Annotated[
    dict[str, Any] | str | None,
    Field(
        description="Additional metadata as an object, or a JSON string holding one. Values are "
        "stored as strings."
    ),
]
SpanId = Annotated[str | None, Field(description="Associate the assessment with this span.")]


def _project(
    trace_dicts: list[dict[str, Any]], fields: list[str] | None, verbose: bool
) -> list[dict[str, Any]]:
    if not fields or not trace_dicts:
        return trace_dicts
    try:
        validate_field_paths(fields, trace_dicts[0], verbose=verbose)
    except ValueError as e:
        raise MlflowException.invalid_parameter_value(str(e)) from None
    return [filter_json_by_fields(trace_dict, fields) for trace_dict in trace_dicts]


def _source(source_type: str | None, source_id: str | None) -> AssessmentSource | None:
    # As in the CLI, a source is only recorded when both parts are given.
    if source_type and source_id:
        return AssessmentSource(source_type=source_type, source_id=source_id)
    return None


def search_traces(
    experiment_id: Annotated[str, Field(description="ID of the experiment to search.")],
    filter_string: Annotated[
        str | None,
        Field(
            description="Search filter, e.g. \"status = 'OK' AND timestamp_ms > 1700000000000\". "
            "Fields: run_id, status, timestamp_ms, execution_time_ms, name, metadata.<key>, "
            "tags.<key> (backticks for keys containing dots)."
        ),
    ] = None,
    max_results: Annotated[
        int | None, Field(description="Maximum number of traces to return (default 100).", ge=1)
    ] = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
    order_by: Annotated[
        list[str] | str | None,
        Field(
            description="Order-by clauses, e.g. ['timestamp_ms DESC']. A comma-separated string "
            "is also accepted."
        ),
    ] = None,
    page_token: PageToken = None,
    run_id: Annotated[
        str | None, Field(description="Only return traces associated with this run.")
    ] = None,
    include_spans: Annotated[
        bool, Field(description="Include span data. Disable for metadata-only queries.")
    ] = True,
    model_id: Annotated[
        str | None, Field(description="Only return traces associated with this model.")
    ] = None,
    extract_fields: ExtractFields = None,
    verbose: Verbose = False,
    output: DeprecatedOutput = None,
) -> TracePage:
    """
    Search for traces in an experiment. Each trace is returned in the shape of
    ``Trace.to_dict()``: ``info`` (trace_id, request_time, execution_duration, state,
    request_preview, response_preview, trace_metadata, tags, assessments) and ``data.spans``.
    """
    traces = TracingClient().search_traces(
        locations=[experiment_id],
        filter_string=filter_string,
        max_results=SEARCH_TRACES_DEFAULT_MAX_RESULTS if max_results is None else max_results,
        order_by=as_list(order_by),
        page_token=page_token,
        run_id=run_id,
        include_spans=include_spans,
        model_id=model_id,
    )
    return TracePage(
        traces=_project([trace.to_dict() for trace in traces], as_list(extract_fields), verbose),
        next_page_token=traces.token or None,
    )


def get_trace(
    trace_id: TraceId,
    extract_fields: ExtractFields = None,
    verbose: Verbose = False,
) -> TraceResult:
    """Get a trace, in the shape of ``Trace.to_dict()``: ``info`` and ``data.spans``."""
    trace_dict = TracingClient().get_trace(trace_id).to_dict()
    (projected,) = _project([trace_dict], as_list(extract_fields), verbose)
    return TraceResult(trace=projected)


def delete_traces(
    experiment_id: Annotated[str, Field(description="ID of the experiment to delete from.")],
    trace_ids: Annotated[
        list[str] | str | None,
        Field(
            description="IDs of the traces to delete. A comma-separated string is also "
            "accepted. Give this or max_timestamp_millis."
        ),
    ] = None,
    max_timestamp_millis: Annotated[
        int | None,
        Field(description="Delete traces older than this timestamp (milliseconds since epoch)."),
    ] = None,
    max_traces: Annotated[
        int | None, Field(description="Maximum number of traces to delete.")
    ] = None,
) -> DeletedTraces:
    """Delete traces from an experiment, by ID or by age."""
    count = TracingClient().delete_traces(
        experiment_id=experiment_id,
        trace_ids=as_list(trace_ids),
        max_timestamp_millis=max_timestamp_millis,
        max_traces=max_traces,
    )
    return DeletedTraces(experiment_id=experiment_id, deleted_count=count)


def set_trace_tag(
    trace_id: TraceId,
    key: Annotated[str, Field(description="Tag key.")],
    value: Annotated[str, Field(description="Tag value.")],
) -> TraceTag:
    """Set a tag on a trace."""
    TracingClient().set_trace_tag(trace_id, key, value)
    return TraceTag(trace_id=trace_id, key=key, value=value)


def delete_trace_tag(
    trace_id: TraceId,
    key: Annotated[str, Field(description="Key of the tag to delete.")],
) -> TraceTag:
    """Delete a tag from a trace."""
    TracingClient().delete_trace_tag(trace_id, key)
    return TraceTag(trace_id=trace_id, key=key)


def log_trace_feedback(
    trace_id: TraceId,
    name: Annotated[str, Field(description="Feedback name, e.g. 'relevance'.")],
    value: Annotated[
        Any,
        Field(
            description="Feedback value: a number, string, boolean, list or object. Strings "
            "holding JSON are parsed."
        ),
    ] = None,
    source_type: SourceType = None,
    source_id: SourceId = None,
    rationale: Annotated[
        str | None, Field(description="Explanation or justification for the feedback.")
    ] = None,
    metadata: Metadata = None,
    span_id: SpanId = None,
) -> AssessmentInfo:
    """Log feedback (an evaluation score) to a trace."""
    assessment = log_feedback(
        trace_id=trace_id,
        name=name,
        value=as_json_value(value),
        source=_source(source_type, source_id),
        rationale=rationale,
        metadata=as_string_map(metadata, "metadata"),
        span_id=span_id,
    )
    return AssessmentInfo.from_entity(assessment)


def log_trace_expectation(
    trace_id: TraceId,
    name: Annotated[str, Field(description="Expectation name, e.g. 'expected_answer'.")],
    value: Annotated[
        Any,
        Field(
            description="Expected value: a string, number, boolean, list or object. Strings "
            "holding JSON are parsed."
        ),
    ],
    source_type: SourceType = None,
    source_id: SourceId = None,
    metadata: Metadata = None,
    span_id: SpanId = None,
) -> AssessmentInfo:
    """Log an expectation (a ground truth label) to a trace."""
    assessment = log_expectation(
        trace_id=trace_id,
        name=name,
        value=as_json_value(value),
        source=_source(source_type, source_id),
        metadata=as_string_map(metadata, "metadata"),
        span_id=span_id,
    )
    return AssessmentInfo.from_entity(assessment)


def get_trace_assessment(trace_id: TraceId, assessment_id: AssessmentId) -> AssessmentInfo:
    """Get an assessment (feedback or expectation) of a trace."""
    return AssessmentInfo.from_entity(TracingClient().get_assessment(trace_id, assessment_id))


def update_trace_assessment(
    trace_id: TraceId,
    assessment_id: AssessmentId,
    value: Annotated[
        Any,
        Field(
            description="New value. Strings holding JSON are parsed. Unchanged when omitted or "
            "null."
        ),
    ] = None,
    rationale: Annotated[
        str | None, Field(description="New rationale (feedback only). Unchanged when omitted.")
    ] = None,
    metadata: Annotated[
        dict[str, Any] | str | None,
        Field(
            description="New metadata as an object or JSON string; values are stored as strings. "
            "Unchanged when omitted."
        ),
    ] = None,
) -> AssessmentInfo:
    """
    Update the value, rationale or metadata of an assessment. Its name cannot be changed.
    """
    client = TracingClient()
    existing = client.get_assessment(trace_id, assessment_id)
    # An omitted value, a JSON null and the legacy string "null" all leave the value as it is: an
    # assessment cannot be left without a value, so null is never a replacement.
    new_value = as_json_value(value)
    keep_value = new_value is None
    if keep_value:
        new_value = existing.value
    new_metadata = existing.metadata if metadata is None else as_string_map(metadata, "metadata")
    if isinstance(existing, Feedback):
        # The store replaces the feedback value and its error together, so a stored error (a
        # judge or scorer failure) is kept unless the caller supplies a replacement value.
        updated = Feedback(
            name=existing.name,
            value=new_value,
            error=existing.error if keep_value else None,
            rationale=existing.rationale if rationale is None else rationale,
            metadata=new_metadata,
        )
    else:
        updated = Expectation(name=existing.name, value=new_value, metadata=new_metadata)
    client.update_assessment(trace_id, assessment_id, updated)
    return AssessmentInfo.from_entity(client.get_assessment(trace_id, assessment_id))


def delete_trace_assessment(trace_id: TraceId, assessment_id: AssessmentId) -> AssessmentRef:
    """Delete an assessment from a trace."""
    TracingClient().delete_assessment(trace_id, assessment_id)
    return AssessmentRef(trace_id=trace_id, assessment_id=assessment_id)
