# Abridged retryable error codes for deployments clients.
# These are modified from the standard MLflow Tracking server retry codes for the MLflowClient to
# remove timeouts from the list of the retryable conditions. A long-running timeout with
# retries for the proxied providers generally indicates an issue with the underlying query or
# the model being served having issues responding to the query due to parameter configuration.
MLFLOW_DEPLOYMENT_CLIENT_REQUEST_RETRY_CODES = frozenset(
    [
        429,  # Too many requests
        500,  # Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
    ]
)
