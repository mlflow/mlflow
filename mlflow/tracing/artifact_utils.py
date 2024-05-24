from mlflow.entities.trace_info import TraceInfo
from mlflow.exceptions import MlflowException
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION

TRACE_DATA_FILE_NAME = "traces.json"


def get_artifact_uri_for_trace(trace_info: TraceInfo) -> str:
    """
    Get the artifact uri for accessing the trace data.

    The artifact root is specified in the trace tags, which is
    set when logging the trace in the backend.
    """
    if MLFLOW_ARTIFACT_LOCATION not in trace_info.tags:
        raise MlflowException(
            "Trace artifact location not specified, please specify it with "
            f"tag '{MLFLOW_ARTIFACT_LOCATION}' in the trace."
        )
    return trace_info.tags[MLFLOW_ARTIFACT_LOCATION]
