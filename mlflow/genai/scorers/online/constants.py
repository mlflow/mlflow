"""Constants for online scoring."""

from mlflow.tracing.constant import TraceMetadataKey

# Maximum lookback period to prevent getting stuck on old failing traces (1 hour)
MAX_LOOKBACK_MS = 60 * 60 * 1000

# Buffer time to wait before considering a session complete (5 minutes)
# Sessions with no new traces for this duration are considered complete
SESSION_COMPLETION_BUFFER_MS = 1 * 15 * 1000

# Maximum traces to include in a single scoring job
MAX_TRACES_PER_JOB = 500

# Maximum sessions to include in a single scoring job
MAX_SESSIONS_PER_JOB = 100

# Filter to exclude eval run traces (traces generated from MLflow runs)
EXCLUDE_EVAL_RUN_TRACES_FILTER = f"metadata.{TraceMetadataKey.SOURCE_RUN} IS NULL"
