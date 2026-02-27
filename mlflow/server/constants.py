"""
Constants used for internal server-to-worker communication.

These are internal environment variables (prefixed with _MLFLOW_SERVER_) used for
communication between the MLflow CLI and forked server processes (gunicorn/uvicorn workers).
They are set by the server and read by workers, and should not be set by end users.
"""

# Backend store configuration
# URI for the backend store (e.g., sqlite:///mlflow.db, postgresql://..., mysql://...)
BACKEND_STORE_URI_ENV_VAR = "_MLFLOW_SERVER_FILE_STORE"

# URI for the model registry store (defaults to same as backend store if not specified)
REGISTRY_STORE_URI_ENV_VAR = "_MLFLOW_SERVER_REGISTRY_STORE"

# Default root directory for storing run artifacts when not explicitly specified
ARTIFACT_ROOT_ENV_VAR = "_MLFLOW_SERVER_ARTIFACT_ROOT"

# Destination for proxied artifact storage operations (used with --serve-artifacts)
ARTIFACTS_DESTINATION_ENV_VAR = "_MLFLOW_SERVER_ARTIFACT_DESTINATION"

# Server features
# Whether the server should act as an artifact proxy (enabled via --serve-artifacts)
SERVE_ARTIFACTS_ENV_VAR = "_MLFLOW_SERVER_SERVE_ARTIFACTS"

# Whether to run in artifacts-only mode (no tracking server, only artifact proxy)
ARTIFACTS_ONLY_ENV_VAR = "_MLFLOW_SERVER_ARTIFACTS_ONLY"

# Flask session secret key for signing cookies and sessions
# (user-configurable via MLFLOW_FLASK_SERVER_SECRET_KEY)
FLASK_SERVER_SECRET_KEY_ENV_VAR = "MLFLOW_FLASK_SERVER_SECRET_KEY"

# Monitoring
# Directory for Prometheus multiprocess metrics collection (enabled via --expose-prometheus)
PROMETHEUS_EXPORTER_ENV_VAR = "prometheus_multiproc_dir"

# Job execution
# Directory path for Huey SQLite task queue storage (used by job execution backend)
HUEY_STORAGE_PATH_ENV_VAR = "_MLFLOW_HUEY_STORAGE_PATH"

# Unique key identifying which Huey instance to use (typically the job function fullname)
MLFLOW_HUEY_INSTANCE_KEY = "_MLFLOW_HUEY_INSTANCE_KEY"

# Secrets management - KEK (Key Encryption Key) environment variables
# NOTE: These are duplicated in mlflow/utils/crypto.py for skinny client compatibility.
# The canonical definitions are in mlflow/utils/crypto.py to avoid Flask import dependency.
# These are kept here for documentation and backwards compatibility with server-side code.
#
# SECURITY: Server-admin-only credential. NEVER pass via CLI (visible in ps/logs).
# Set via environment variable or .env file. Users do NOT need this - only server admins.
# Must be high-entropy (32+ characters) from a secrets manager.
#
# KEK Rotation Workflow (for changing the passphrase):
#   1. Shut down the MLflow server
#   2. Set MLFLOW_CRYPTO_KEK_PASSPHRASE to the OLD passphrase
#   3. Run: mlflow crypto rotate-kek --new-passphrase "NEW_PASSPHRASE"
#   4. Update MLFLOW_CRYPTO_KEK_PASSPHRASE to NEW passphrase in deployment config
#   5. Restart the MLflow server
#
# The rotation is atomic and idempotent - safe to re-run if it fails.
CRYPTO_KEK_PASSPHRASE_ENV_VAR = "MLFLOW_CRYPTO_KEK_PASSPHRASE"

# KEK version for tracking which KEK encrypted each secret (default 1).
# Automatically tracked during rotation. See `mlflow crypto rotate-kek` for rotation workflow.
CRYPTO_KEK_VERSION_ENV_VAR = "MLFLOW_CRYPTO_KEK_VERSION"

# Secrets cache configuration
# Time-to-live for server-side secrets cache in seconds (10-300s range, default 60s)
SECRETS_CACHE_TTL_ENV_VAR = "MLFLOW_SERVER_SECRETS_CACHE_TTL"

# Maximum number of entries in server-side secrets cache (default 1000 entries)
SECRETS_CACHE_MAX_SIZE_ENV_VAR = "MLFLOW_SERVER_SECRETS_CACHE_MAX_SIZE"
