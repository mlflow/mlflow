"""Constants for online scoring."""

from mlflow.tracing.constant import TraceMetadataKey

# Maximum lookback period to prevent getting stuck on old failing traces (1 hour)
MAX_LOOKBACK_MS = 60 * 60 * 1000

# Maximum traces to include in a single scoring batch
MAX_TRACES_PER_JOB = 500

# Maximum sessions to include in a single scoring batch
MAX_SESSIONS_PER_JOB = 100

# A job keeps running batches while the previous batch's discovery page was full (there is a
# backlog) and the checkpoint is advancing, until it has run for this long. Without this a job
# would score at most one batch per scheduler tick and a backlog larger than
# MAX_LOOKBACK_MS worth of ticks would fall out of the window before it was reached.
MAX_JOB_RUNTIME_MS = 20 * 60 * 1000

# How long (measured from the trace's start timestamp) online scoring keeps waiting for a
# trace that is still in progress, or whose spans cannot be loaded yet, before giving up on
# it. While a trace is waiting the checkpoint is held before it so it is retried on the next
# run; once it is older than this it is skipped so one stuck trace cannot stall scoring.
MAX_TRACE_COMPLETION_WAIT_MS = 10 * 60 * 1000

# How long (measured from the trace's start timestamp, or the session's last trace timestamp)
# a scorer failure is retried before it is recorded as an error assessment. Judge failures are
# usually transient (rate limits, timeouts, gateway errors), so while the entity is younger
# than this the checkpoint is held before it and nothing is persisted; once it is older the
# error is persisted so it is visible and the pipeline moves on. Must stay well below
# MAX_LOOKBACK_MS so the backlog behind a held entity is still reachable when the hold ends.
MAX_SCORER_FAILURE_RETRY_MS = 30 * 60 * 1000

# Filter to exclude eval run traces (traces generated from MLflow runs)
EXCLUDE_EVAL_RUN_TRACES_FILTER = f"metadata.{TraceMetadataKey.SOURCE_RUN} IS NULL"
