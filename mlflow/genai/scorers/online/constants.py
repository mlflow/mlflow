"""Constants for online scoring."""

# Checkpoint tags for tracking last processed trace/session
TRACE_CHECKPOINT_TAG = "mlflow.latestOnlineScoring.trace.checkpoint"
SESSION_CHECKPOINT_TAG = "mlflow.latestOnlineScoring.session.checkpoint"

# Maximum lookback period to prevent getting stuck on old failing traces/sessions (1 hour)
MAX_LOOKBACK_MS = 60 * 60 * 1000

# Session inactivity buffer: 10 minutes without new traces = session complete
SESSION_COMPLETION_BUFFER_MS = 10 * 60 * 1000

# Maximum sessions to process in a single scoring job
MAX_SESSIONS_PER_JOB = 50

# Maximum traces to include in a single scoring job
MAX_TRACES_PER_JOB = 500

# Filter to exclude eval run traces (traces generated from MLflow runs)
EXCLUDE_EVAL_RUN_TRACES_FILTER = "metadata.mlflow.sourceRun IS NULL"
