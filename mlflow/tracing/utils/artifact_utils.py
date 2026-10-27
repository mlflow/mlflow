from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation, TraceLocationType
from mlflow.exceptions import MlflowTraceDataCorrupted
from mlflow.tracing.constant import TRACE_ID_V4_PREFIX, TraceTagKey
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION

TRACE_DATA_FILE_NAME = "traces.json"

# Unity Catalog table traces keep spans in tables and never receive an artifact location.
_TRACE_LOCATIONS_WITHOUT_ARTIFACT_STORE = frozenset({
    TraceLocationType.UC_SCHEMA,
    TraceLocationType.UC_TABLE_PREFIX,
})

UC_TABLE_ATTACHMENTS_UNSUPPORTED_MESSAGE = (
    "Trace attachments are not supported for Unity Catalog table traces because "
    "they have no artifact location. Multimodal content is kept inline in the span."
)


def _get_trace_request_id(trace_info: TraceInfo) -> str | None:
    return getattr(trace_info, "trace_id", getattr(trace_info, "request_id", None))


def trace_location_supports_artifact_uploads(trace_location: TraceLocation | None) -> bool:
    """
    Whether this trace location can store attachment bytes in an artifact store.

    Unity Catalog table traces keep spans in tables and do not carry an artifact
    location. An unknown location is treated as supporting artifact uploads so
    experiment traces keep the existing extraction behavior.
    """
    if trace_location is None:
        return True
    return trace_location.type not in _TRACE_LOCATIONS_WITHOUT_ARTIFACT_STORE


def trace_attachments_unsupported_message(trace_info: TraceInfo) -> str | None:
    """
    Explain why attachment bytes cannot be uploaded, or return None to continue.

    A missing artifact location on an experiment trace is still reported as
    corrupted trace data by the caller. Unity Catalog table traces get a
    location specific message instead.
    """
    tags = getattr(trace_info, "tags", None) or {}
    if MLFLOW_ARTIFACT_LOCATION in tags:
        return None

    location = getattr(trace_info, "trace_location", None)
    trace_id = _get_trace_request_id(trace_info)
    # V4 trace IDs are minted only for Unity Catalog table destinations. Use that
    # when the returned trace info has no location object.
    location_has_no_artifact_store = (
        location is not None and not trace_location_supports_artifact_uploads(location)
    )
    v4_trace_without_location = (
        location is None and isinstance(trace_id, str) and (trace_id.startswith(TRACE_ID_V4_PREFIX))
    )
    if not location_has_no_artifact_store and not v4_trace_without_location:
        return None
    return f"{UC_TABLE_ATTACHMENTS_UNSUPPORTED_MESSAGE} (request_id={trace_id})"


def get_artifact_uri_for_trace(trace_info: TraceInfo):
    """
    Get the artifact uri for accessing the trace data.

    The artifact root is specified in the trace tags, which is
    set when logging the trace in the backend.

    Args:
        trace_info: Either a TraceInfo or TraceInfoV3 object containing trace metadata.

    Returns:
        The artifact URI string for the trace data.
    """
    # Both TraceInfo and TraceInfoV3 access tags the same way
    if MLFLOW_ARTIFACT_LOCATION not in trace_info.tags:
        raise MlflowTraceDataCorrupted(request_id=_get_trace_request_id(trace_info))
    return trace_info.tags[MLFLOW_ARTIFACT_LOCATION]


def get_archive_uri_for_trace(trace_info: TraceInfo):
    """
    Get the archive uri for accessing archived trace span payloads.

    Args:
        trace_info: Either a TraceInfo or TraceInfoV3 object containing trace metadata.

    Returns:
        The archival URI string for the archived trace payload.
    """
    if TraceTagKey.ARCHIVE_LOCATION not in trace_info.tags:
        raise MlflowTraceDataCorrupted(request_id=_get_trace_request_id(trace_info))
    return trace_info.tags[TraceTagKey.ARCHIVE_LOCATION]
