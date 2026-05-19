import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, resolve } from 'node:path';

import { createAuthProvider, init, MlflowClient } from '@mlflow/core';

export const MLFLOW_CLAUDE_TRACING_ENABLED = 'MLFLOW_CLAUDE_TRACING_ENABLED';
export const MLFLOW_TRACKING_URI = 'MLFLOW_TRACKING_URI';
export const MLFLOW_EXPERIMENT_ID = 'MLFLOW_EXPERIMENT_ID';
export const MLFLOW_EXPERIMENT_NAME = 'MLFLOW_EXPERIMENT_NAME';

type ConfigSource = 'environment' | 'project' | 'user' | 'none';

export interface ClaudeSettings {
  env?: Record<string, string>;
  [key: string]: unknown;
}

export interface TracingConfig {
  enabled: boolean;
  trackingUri?: string;
  experimentId?: string;
  experimentName?: string;
  source: ConfigSource;
  settingsPath?: string;
}

export interface ConfigPathOptions {
  home?: string;
  cwd?: string;
}

let initializedKey: string | null = null;

function isTruthy(value: string | undefined): boolean {
  const normalized = (value ?? '').trim().toLowerCase();
  return normalized === 'true' || normalized === '1' || normalized === 'yes';
}

function hasConfigValue(value: string | undefined): value is string {
  return Boolean(value && value.trim().length > 0);
}

function hasAnyTracingKey(env: Record<string, string | undefined>): boolean {
  return [
    MLFLOW_CLAUDE_TRACING_ENABLED,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
  ].some((key) => env[key] !== undefined);
}

function hasTracingConfig(config: TracingConfig): boolean {
  return Boolean(
    config.trackingUri || config.experimentId || config.experimentName || config.enabled,
  );
}

function parseTracingConfig(
  env: Record<string, string | undefined>,
  source: ConfigSource,
  settingsPath?: string,
): TracingConfig {
  return {
    enabled: isTruthy(env[MLFLOW_CLAUDE_TRACING_ENABLED]),
    trackingUri: env[MLFLOW_TRACKING_URI],
    experimentId: env[MLFLOW_EXPERIMENT_ID],
    experimentName: env[MLFLOW_EXPERIMENT_NAME],
    source,
    settingsPath,
  };
}

export function resolveSettingsPath(
  projectLocal: boolean,
  options: ConfigPathOptions = {},
): string {
  return projectLocal
    ? resolve(options.cwd ?? process.cwd(), '.claude', 'settings.json')
    : resolve(options.home ?? homedir(), '.claude', 'settings.json');
}

export function loadSettings(path: string): ClaudeSettings {
  let raw: string;
  try {
    raw = readFileSync(path, 'utf-8');
  } catch {
    return {};
  }
  return JSON.parse(raw) as ClaudeSettings;
}

export function saveSettings(path: string, settings: ClaudeSettings): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(settings, null, 2) + '\n', 'utf-8');
}

export function getScopeTracingConfig(
  projectLocal: boolean,
  options: ConfigPathOptions = {},
): TracingConfig {
  const settingsPath = resolveSettingsPath(projectLocal, options);
  const settings = loadSettings(settingsPath);
  return parseTracingConfig(
    Object.fromEntries(
      Object.entries(settings.env ?? {}).map(([key, value]) => [key, String(value)]),
    ),
    projectLocal ? 'project' : 'user',
    settingsPath,
  );
}

export function getEffectiveTracingConfig(options: ConfigPathOptions = {}): TracingConfig {
  const userConfig = getScopeTracingConfig(false, options);
  const projectConfig = getScopeTracingConfig(true, options);

  const merged = {
    enabled: userConfig.enabled,
    trackingUri: userConfig.trackingUri,
    experimentId: userConfig.experimentId,
    experimentName: userConfig.experimentName,
    ...(hasTracingConfig(projectConfig)
      ? {
          enabled: projectConfig.enabled,
          trackingUri: projectConfig.trackingUri,
          experimentId: projectConfig.experimentId,
          experimentName: projectConfig.experimentName,
        }
      : {}),
  };

  const envConfig = parseTracingConfig(process.env, 'environment');
  const effective: TracingConfig = {
    enabled:
      process.env[MLFLOW_CLAUDE_TRACING_ENABLED] !== undefined ? envConfig.enabled : merged.enabled,
    trackingUri: process.env[MLFLOW_TRACKING_URI] ?? merged.trackingUri,
    experimentId: process.env[MLFLOW_EXPERIMENT_ID] ?? merged.experimentId,
    experimentName: process.env[MLFLOW_EXPERIMENT_NAME] ?? merged.experimentName,
    source: 'none',
  };

  if (hasAnyTracingKey(process.env)) {
    effective.source = 'environment';
  } else if (hasTracingConfig(projectConfig)) {
    effective.source = 'project';
    effective.settingsPath = projectConfig.settingsPath;
  } else if (hasTracingConfig(userConfig)) {
    effective.source = 'user';
    effective.settingsPath = userConfig.settingsPath;
  }

  return effective;
}

export function isTracingEnabled(): boolean {
  return getEffectiveTracingConfig().enabled;
}

export function isValidTrackingUri(raw: string): boolean {
  if (raw === 'databricks' || raw.startsWith('databricks://')) {
    return true;
  }

  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    return false;
  }
  return parsed.protocol === 'http:' || parsed.protocol === 'https:';
}

export async function resolveExperiment(
  trackingUri: string,
  experimentId?: string,
  experimentName?: string,
): Promise<{ experimentId: string; experimentName?: string; created: boolean }> {
  if (hasConfigValue(experimentId)) {
    return { experimentId, experimentName, created: false };
  }

  if (!hasConfigValue(experimentName)) {
    throw new Error(
      'Either MLFLOW_EXPERIMENT_ID or MLFLOW_EXPERIMENT_NAME must be configured for Claude tracing.',
    );
  }

  const authProvider = createAuthProvider({ trackingUri });
  const client = new MlflowClient({ trackingUri, authProvider });
  const existing = await client.getExperimentByName(experimentName);
  if (existing) {
    return {
      experimentId: existing.experimentId,
      experimentName: existing.name,
      created: false,
    };
  }

  return {
    experimentId: await client.createExperiment(experimentName),
    experimentName,
    created: true,
  };
}

export function writeTracingSettings(
  settingsPath: string,
  config: { trackingUri: string; experimentId: string; experimentName?: string; enabled?: boolean },
): void {
  const settings = loadSettings(settingsPath);
  const env = { ...(settings.env ?? {}) };

  env[MLFLOW_CLAUDE_TRACING_ENABLED] = config.enabled === false ? 'false' : 'true';
  env[MLFLOW_TRACKING_URI] = config.trackingUri;

  if (hasConfigValue(config.experimentId)) {
    env[MLFLOW_EXPERIMENT_ID] = config.experimentId;
  } else {
    delete env[MLFLOW_EXPERIMENT_ID];
  }

  if (hasConfigValue(config.experimentName)) {
    env[MLFLOW_EXPERIMENT_NAME] = config.experimentName;
  } else {
    delete env[MLFLOW_EXPERIMENT_NAME];
  }

  settings.env = env;
  saveSettings(settingsPath, settings);
}

export async function ensureInitialized(): Promise<boolean> {
  const config = getEffectiveTracingConfig();
  if (!config.enabled) {
    return false;
  }

  if (!hasConfigValue(config.trackingUri)) {
    console.error('[mlflow] MLFLOW_TRACKING_URI is not set');
    return false;
  }

  if (!hasConfigValue(config.experimentId) && !hasConfigValue(config.experimentName)) {
    console.error('[mlflow] MLFLOW_EXPERIMENT_ID or MLFLOW_EXPERIMENT_NAME is not set');
    return false;
  }

  // Fast path: skip network call when experimentId is already known.
  if (hasConfigValue(config.experimentId)) {
    const quickKey = JSON.stringify({
      trackingUri: config.trackingUri,
      experimentId: config.experimentId,
    });
    if (initializedKey === quickKey) {
      return true;
    }
  }

  try {
    const resolvedExperiment = await resolveExperiment(
      config.trackingUri,
      config.experimentId,
      config.experimentName,
    );
    const initKey = JSON.stringify({
      trackingUri: config.trackingUri,
      experimentId: resolvedExperiment.experimentId,
    });
    if (initializedKey === initKey) {
      return true;
    }

    init({
      trackingUri: config.trackingUri,
      experimentId: resolvedExperiment.experimentId,
    });
    initializedKey = initKey;
    return true;
  } catch (err) {
    console.error('[mlflow] Failed to initialize:', err);
    return false;
  }
}
