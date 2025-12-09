/**
 * API endpoint for telemetry. We don't use an absolute URL because it breaks
 * reverse-proxy setups (e.g. someone deploys mlflow at www.example.com/mlflow).
 * Instead, we use a relative URL that goes up one level, since the worker JS file
 * is served from the static files directory under the root.
 */
// eslint-disable-next-line mlflow/no-absolute-ajax-urls
export const TELEMETRY_ENDPOINT = '../ajax-api/2.0/mlflow/telemetry';
