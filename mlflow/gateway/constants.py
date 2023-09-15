MLFLOW_GATEWAY_HEALTH_ENDPOINT = "/health"
MLFLOW_GATEWAY_CRUD_ROUTE_BASE = "/api/2.0/gateway/routes/"
MLFLOW_GATEWAY_LIMITS_BASE = "/api/2.0/gateway/limits/"
MLFLOW_GATEWAY_ROUTE_BASE = "/gateway/"
MLFLOW_QUERY_SUFFIX = "/invocations"
MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE = 3000

# Specifies the timeout for the Gateway server to declare a request submitted to a provider has
# timed out.
MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS = 300

# Specifies the timeout for the MLflowGatewayClient APIs to declare a request has timed out
MLFLOW_GATEWAY_CLIENT_QUERY_TIMEOUT_SECONDS = 300

# Abridged retryable error codes for the interface to the Gateway Server.
# These are modified from the standard MLflow Tracking server retry codes for the MLflowClient to
# remove timeouts from the list of the retryable conditions. A long-running timeout with
# retries for the proxied providers generally indicates an issue with the underlying query or
# the model being served having issues responding to the query due to parameter configuration.
MLFLOW_GATEWAY_CLIENT_QUERY_RETRY_CODES = frozenset(
    [
        429,  # Too many requests
        500,  # Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
    ]
)

# Provider constants
MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS = 1_000_000
MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS = 200_000

# MLflow model serving constants
MLFLOW_SERVING_RESPONSE_KEY = "predictions"

# MosaicML supported chat model names
# These validated names are used for the MosaicML provider due to the need to perform prompt
# translations prior to sending a request payload to their chat endpoints.
# to reduce the need to case-match, supported model prefixes are lowercase.
MLFLOW_AI_GATEWAY_MOSAICML_CHAT_SUPPORTED_MODEL_PREFIXES = ["llama2"]
