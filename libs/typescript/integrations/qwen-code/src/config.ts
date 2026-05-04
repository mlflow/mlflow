import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { resolve } from 'node:path';

import { init } from '@mlflow/core';

let initialized = false;

const TRACING_CONFIG_FILE = 'mlflow-tracing.json';

interface TracingConfig {
  trackingUri?: string;
  experimentId?: string;
}

export interface ResolveConfigOptions {
  /** Override the user home directory. Defaults to `os.homedir()`. */
  home?: string;
  /** Override the current working directory. Defaults to `process.cwd()`. */
  cwd?: string;
}

function readTracingConfigFile(path: string): TracingConfig {
  if (!existsSync(path)) {
    return {};
  }
  try {
    return JSON.parse(readFileSync(path, 'utf-8')) as TracingConfig;
  } catch {
    return {};
  }
}

/**
 * Resolve the effective MLflow tracing config. Precedence (highest first):
 *   1. `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT_ID` environment variables
 *   2. `./.qwen/mlflow-tracing.json` (project-local)
 *   3. `~/.qwen/mlflow-tracing.json` (user-level)
 */
export function resolveTracingConfig(options: ResolveConfigOptions = {}): TracingConfig {
  const projectPath = resolve(options.cwd ?? process.cwd(), '.qwen', TRACING_CONFIG_FILE);
  const userPath = resolve(options.home ?? homedir(), '.qwen', TRACING_CONFIG_FILE);
  const projectConfig = readTracingConfigFile(projectPath);
  const userConfig = readTracingConfigFile(userPath);
  return {
    trackingUri:
      process.env.MLFLOW_TRACKING_URI ?? projectConfig.trackingUri ?? userConfig.trackingUri,
    experimentId:
      process.env.MLFLOW_EXPERIMENT_ID ?? projectConfig.experimentId ?? userConfig.experimentId,
  };
}

/**
 * Initialize the MLflow SDK with tracking URI and experiment settings.
 */
export function ensureInitialized(): boolean {
  if (initialized) {
    return true;
  }

  const { trackingUri, experimentId } = resolveTracingConfig();
  if (!trackingUri) {
    console.error(
      '[mlflow] MLflow tracking URI is not configured — checked $MLFLOW_TRACKING_URI,',
      './.qwen/mlflow-tracing.json, and ~/.qwen/mlflow-tracing.json.',
    );
    console.error('[mlflow] Run `mlflow-qwen-code setup` to configure.');
    return false;
  }

  init({ trackingUri, experimentId });
  initialized = true;
  return true;
}
