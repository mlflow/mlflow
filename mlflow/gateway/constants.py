CONF_PATH_ENV_VAR = "MLFLOW_GATEWAY_CONFIG"
BASE_WAIT_TIME_SECONDS = 0.1
MAX_WAIT_TIME_SECONDS = 5
GATEWAY_SERVER_STATE_FILE = "~/.mlflow/gateway/state.yaml"
PROVIDERS = {"openai", "anthropic", "databricks_serving_endpoint", "mlflow_served_model"}
