/**
 * Provider id for the in-server MLflow AI Gateway assistant backend.
 *
 * Must match the provider `name` defined server-side in
 * `mlflow/assistant/providers/__init__.py`, since the `/config` payload keys
 * `providers` by that name.
 */
export const GATEWAY_PROVIDER_ID = 'mlflow_gateway';
