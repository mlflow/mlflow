"""Constants for online scoring."""

# Checkpoint tag for tracking last processed trace
TRACE_CHECKPOINT_TAG = "mlflow.latestOnlineScoring.trace.checkpoint"

# Maximum lookback period to prevent getting stuck on old failing traces (1 hour)
MAX_LOOKBACK_MS = 60 * 60 * 1000

# Maximum traces to include in a single scoring job
MAX_TRACES_PER_JOB = 500

# Filter to exclude eval run traces (traces generated from MLflow runs)
EXCLUDE_EVAL_RUN_TRACES_FILTER = "metadata.mlflow.sourceRun IS NULL"
