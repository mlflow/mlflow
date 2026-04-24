import { init } from '@mlflow/core';

let initialized = false;

/**
 * Check if MLflow Claude Code tracing is enabled via environment variable.
 */
export function isTracingEnabled(): boolean {
  const value = (process.env.MLFLOW_CLAUDE_TRACING_ENABLED ?? '').toLowerCase();
  return value === 'true' || value === '1' || value === 'yes';
}

/**
 * Initialize the MLflow SDK with tracking URI and experiment settings.
 * No-ops if already initialized or if required env vars are missing.
 */
export function ensureInitialized(): boolean {
  if (initialized) {
    return true;
  }

  const trackingUri = process.env.MLFLOW_TRACKING_URI;
  if (!trackingUri) {
    console.error('[mlflow] MLFLOW_TRACKING_URI is not set');
    return false;
  }

  const experimentId = process.env.MLFLOW_EXPERIMENT_ID;
  if (!experimentId) {
    console.error('[mlflow] MLFLOW_EXPERIMENT_ID is not set');
    return false;
  }

  try {
    init({
      trackingUri,
      experimentId,
    });
    initialized = true;
    return true;
  } catch (err) {
    console.error('[mlflow] Failed to initialize:', err);
    return false;
  }
}
