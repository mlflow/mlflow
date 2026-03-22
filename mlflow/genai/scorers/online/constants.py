"""Constants for online scoring."""

from mlflow.tracing.constant import TraceMetadataKey

# Maximum lookback period to prevent getting stuck on old failing traces (1 hour)
MAX_LOOKBACK_MS = 60 * 60 * 1000

# Extra lookback added to the start of every scoring window to catch long-running
# traces that *started* before the current checkpoint but *completed* (status
# changed to OK/ERROR) after it.  Without this buffer, any trace whose execution
# time exceeds the scheduler polling interval will be missed because:
#   1. The scheduler scans window [T1, T2] using trace.timestamp_ms (start time).
#   2. The trace starts at T ∈ [T1, T2] but is still IN_PROGRESS at scan time.
#   3. The checkpoint advances to T2.
#   4. The next window starts at T2 > T, so the trace is permanently excluded.
# The buffer extends the window start backwards so the trace can be picked up
# in the next scan once its status transitions to a terminal state.
# Default: 2 minutes — covers schedulers running every 60 s with a generous margin.
TRACE_COMPLETION_OVERLAP_MS = 2 * 60 * 1000

# Maximum traces to include in a single scoring job
MAX_TRACES_PER_JOB = 500

# Maximum sessions to include in a single scoring job
MAX_SESSIONS_PER_JOB = 100

# Filter to exclude eval run traces (traces generated from MLflow runs)
EXCLUDE_EVAL_RUN_TRACES_FILTER = f"metadata.{TraceMetadataKey.SOURCE_RUN} IS NULL"
