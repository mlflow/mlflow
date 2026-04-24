import { init } from '@mlflow/core';

let initialized = false;

/**
 * Initialize the MLflow SDK with tracking URI and experiment settings.
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

  init({
    trackingUri,
    experimentId: process.env.MLFLOW_EXPERIMENT_ID,
  });

  initialized = true;
  return true;
}
