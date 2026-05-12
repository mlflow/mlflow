import { existsSync } from 'node:fs';

import {
  getEffectiveTracingConfig,
  isValidTrackingUri,
  resolveExperiment,
  resolveSettingsPath,
  writeTracingSettings,
  type ConfigPathOptions,
} from '../config.js';

export interface SetupOptions extends ConfigPathOptions {
  trackingUri?: string;
  experimentId?: string;
  experimentName?: string;
}

export interface ParsedSetupArgs extends SetupOptions {
  projectLocal: boolean | undefined;
}

export function parseSetupArgs(args: string[]): ParsedSetupArgs {
  const parsed: ParsedSetupArgs = { projectLocal: undefined };
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--project' || arg === '-p') {
      parsed.projectLocal = true;
    } else if (arg === '--user' || arg === '-u') {
      parsed.projectLocal = false;
    } else if (arg === '--tracking-uri') {
      parsed.trackingUri = args[++i];
    } else if (arg === '--experiment-id') {
      parsed.experimentId = args[++i];
    } else if (arg === '--experiment-name') {
      parsed.experimentName = args[++i];
    }
  }
  return parsed;
}

function printSummary(
  settingsPath: string,
  config: { trackingUri: string; experimentId: string; experimentName?: string },
): void {
  console.error('\nCurrent configuration');
  console.error('  Tracing enabled: true');
  console.error(`  Settings file: ${settingsPath}`);
  console.error(`  Tracking URI: ${config.trackingUri}`);
  console.error(`  Experiment ID: ${config.experimentId}`);
  if (config.experimentName) {
    console.error(`  Experiment name: ${config.experimentName}`);
  }
}

export async function runSetup(args: string[], options: SetupOptions = {}): Promise<void> {
  const parsed = parseSetupArgs(args);
  const merged: ParsedSetupArgs = { ...parsed, ...options, projectLocal: parsed.projectLocal };

  if (merged.projectLocal === undefined) {
    console.error('Error: must pass --project or --user.');
    process.exitCode = 1;
    return;
  }

  if (!merged.trackingUri) {
    console.error('Error: --tracking-uri is required.');
    process.exitCode = 1;
    return;
  }

  if (!isValidTrackingUri(merged.trackingUri)) {
    console.error(
      `Error: invalid --tracking-uri: ${merged.trackingUri}. ` +
        "Must be 'databricks', 'databricks://<profile>', or an absolute http(s) URL.",
    );
    process.exitCode = 1;
    return;
  }

  if (merged.experimentId && merged.experimentName) {
    console.error('Error: pass only one of --experiment-id or --experiment-name.');
    process.exitCode = 1;
    return;
  }

  if (!merged.experimentId && !merged.experimentName) {
    console.error('Error: must pass --experiment-id or --experiment-name.');
    process.exitCode = 1;
    return;
  }

  const settingsPath = resolveSettingsPath(merged.projectLocal, merged);

  try {
    const settingsFileExisted = existsSync(settingsPath);
    const resolved = await resolveExperiment(
      merged.trackingUri,
      merged.experimentId,
      merged.experimentName,
    );
    writeTracingSettings(settingsPath, {
      trackingUri: merged.trackingUri,
      experimentId: resolved.experimentId,
      experimentName: resolved.experimentName,
    });

    console.error(`\n${settingsFileExisted ? 'Updated' : 'Created'} ${settingsPath}`);
    if (resolved.created && resolved.experimentName) {
      console.error(
        `Created MLflow experiment ${resolved.experimentName} (${resolved.experimentId})`,
      );
    } else {
      console.error(`Resolved MLflow experiment ID ${resolved.experimentId}`);
    }

    printSummary(settingsPath, {
      trackingUri: merged.trackingUri,
      experimentId: resolved.experimentId,
      experimentName: resolved.experimentName,
    });
  } catch (error) {
    console.error(`Error: ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  }
}

export function runStatus(options: ConfigPathOptions = {}): void {
  const config = getEffectiveTracingConfig(options);

  console.error('\nMLflow Tracing Status\n');
  console.error(`  Enabled: ${config.enabled ? 'true' : 'false'}`);
  console.error(`  Source: ${config.source}`);
  if (config.settingsPath) {
    console.error(`  Settings file: ${config.settingsPath}`);
  }
  console.error(`  Tracking URI: ${config.trackingUri ?? 'not set'}`);
  console.error(`  Experiment ID: ${config.experimentId ?? 'not set'}`);
  console.error(`  Experiment name: ${config.experimentName ?? 'not set'}`);

  if (!config.enabled) {
    console.error('\nTracing is disabled. Run `mlflow-claude-code setup` to configure it.');
  }
}
