from typing import Union

from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_info_v2 import TraceInfoV2
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION

TRACE_DATA_FILE_NAME = "traces.json"


def get_artifact_uri_for_trace(trace_info: Union[TraceInfoV2, TraceInfo]):
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
        raise MlflowException(
            "Unable to determine trace artifact location.",
            error_code=INTERNAL_ERROR,
        )
    return trace_info.tags[MLFLOW_ARTIFACT_LOCATION]
