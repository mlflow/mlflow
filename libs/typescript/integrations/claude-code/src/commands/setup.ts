import { existsSync } from 'node:fs';
import { createInterface } from 'node:readline';

import {
  getEffectiveTracingConfig,
  isValidTrackingUri,
  resolveExperiment,
  resolveSettingsPath,
  writeTracingSettings,
  type ConfigPathOptions,
} from '../config.js';
import { selectPrompt } from '../ui-select.js';
import { FAIL, OK, WARN, bold, cyan, dim } from '../ui.js';

const DEFAULT_TRACKING_URI = 'http://localhost:5000';
const DEFAULT_EXPERIMENT_NAME = 'claude-code-traces';

export interface SetupOptions extends ConfigPathOptions {
  trackingUri?: string;
  experimentId?: string;
  experimentName?: string;
  nonInteractive?: boolean;
}

export interface ParsedSetupArgs extends SetupOptions {
  projectLocal: boolean | undefined;
}

function askQuestion(
  label: string,
  defaultValue: string,
  validate?: (value: string) => string | null,
): Promise<string> {
  return new Promise((resolvePromise) => {
    const rl = createInterface({ input: process.stdin, output: process.stdout });
    const ask = (): void => {
      rl.question(`  ${label} ${dim(`[${defaultValue}]`)} `, (answer) => {
        const value = answer.trim() || defaultValue;
        const error = validate?.(value);
        if (error) {
          console.error(`  ${FAIL} ${error}`);
          ask();
          return;
        }
        rl.close();
        resolvePromise(value);
      });
    };
    ask();
  });
}

function validateTrackingUri(value: string): string | null {
  return isValidTrackingUri(value)
    ? null
    : "Must be 'databricks', 'databricks://<profile>', or an absolute http(s) URL.";
}

export function parseSetupArgs(args: string[]): ParsedSetupArgs {
  const parsed: ParsedSetupArgs = { projectLocal: undefined };
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--project' || arg === '-p') {
      parsed.projectLocal = true;
    } else if (arg === '--user' || arg === '-u') {
      parsed.projectLocal = false;
    } else if (arg === '--non-interactive' || arg === '-y') {
      parsed.nonInteractive = true;
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

async function promptScope(): Promise<boolean> {
  return selectPrompt<boolean>({
    question: 'Where should MLflow tracing be configured?',
    options: [
      { value: true, label: 'Project', hint: './.claude/settings.json' },
      { value: false, label: 'User', hint: '~/.claude/settings.json' },
    ],
    defaultIndex: 0,
  });
}

async function promptExperimentMode(): Promise<'id' | 'name'> {
  return selectPrompt<'id' | 'name'>({
    question: 'How should MLflow choose the experiment?',
    options: [
      { value: 'name', label: 'Experiment name', hint: 'Create if it does not exist' },
      { value: 'id', label: 'Experiment ID', hint: 'Use an existing experiment directly' },
    ],
    defaultIndex: 0,
  });
}

function printSummary(settingsPath: string, config: { trackingUri: string; experimentId: string; experimentName?: string }): void {
  console.error(`\n${bold('Current configuration')}`);
  console.error(`  Tracing enabled: ${cyan('true')}`);
  console.error(`  Settings file: ${cyan(settingsPath)}`);
  console.error(`  Tracking URI: ${cyan(config.trackingUri)}`);
  console.error(`  Experiment ID: ${cyan(config.experimentId)}`);
  if (config.experimentName) {
    console.error(`  Experiment name: ${cyan(config.experimentName)}`);
  }

  const port =
    config.trackingUri.startsWith('http://') || config.trackingUri.startsWith('https://')
      ? new URL(config.trackingUri).port || '5000'
      : '5000';
  console.error(`\n${bold('Next steps')}`);
  if (config.trackingUri.startsWith('http://') || config.trackingUri.startsWith('https://')) {
    console.error(`  1. Start MLflow if it is not already running: ${cyan(`mlflow server --port ${port}`)}`);
  } else {
    console.error('  1. Make sure your Databricks credentials are available in this shell.');
  }
  console.error('  2. Use Claude Code normally in this repo.');
  console.error('  3. Open MLflow to inspect traces after a Claude conversation ends.');
}

export async function runSetup(args: string[], options: SetupOptions = {}): Promise<void> {
  const parsed = parseSetupArgs(args);
  const merged: ParsedSetupArgs = { ...parsed, ...options, projectLocal: parsed.projectLocal };

  if (merged.experimentId && merged.experimentName) {
    console.error(`${FAIL} Choose either --experiment-id or --experiment-name, not both.`);
    process.exitCode = 1;
    return;
  }

  const interactive = !merged.nonInteractive;
  if (interactive) {
    console.error(`\n${bold('MLflow Tracing Setup')}\n`);
  }

  const projectLocal =
    merged.projectLocal !== undefined
      ? merged.projectLocal
      : interactive
        ? await promptScope()
        : true;
  const settingsPath = resolveSettingsPath(projectLocal, merged);

  try {
    const trackingUri =
      merged.trackingUri ??
      (interactive
        ? await askQuestion('MLflow tracking URI', DEFAULT_TRACKING_URI, validateTrackingUri)
        : DEFAULT_TRACKING_URI);

    if (!isValidTrackingUri(trackingUri)) {
      console.error(`${FAIL} Invalid tracking URI: ${trackingUri}`);
      process.exitCode = 1;
      return;
    }

    let experimentId = merged.experimentId;
    let experimentName = merged.experimentName;
    if (!experimentId && !experimentName) {
      if (interactive) {
        const mode = await promptExperimentMode();
        if (mode === 'id') {
          experimentId = await askQuestion('MLflow experiment ID', '0');
        } else {
          experimentName = await askQuestion(
            'MLflow experiment name',
            DEFAULT_EXPERIMENT_NAME,
          );
        }
      } else {
        experimentName = DEFAULT_EXPERIMENT_NAME;
      }
    }

    const settingsFileExisted = existsSync(settingsPath);
    const resolvedExperiment = await resolveExperiment(trackingUri, experimentId, experimentName);
    writeTracingSettings(settingsPath, {
      trackingUri,
      experimentId: resolvedExperiment.experimentId,
      experimentName: resolvedExperiment.experimentName,
    });

    console.error(`\n${OK} ${settingsFileExisted ? 'Updated' : 'Created'} ${cyan(settingsPath)}`);
    if (resolvedExperiment.created && resolvedExperiment.experimentName) {
      console.error(
        `${OK} Created MLflow experiment ${cyan(resolvedExperiment.experimentName)} (${cyan(resolvedExperiment.experimentId)})`,
      );
    } else {
      console.error(`${OK} Resolved MLflow experiment ID ${cyan(resolvedExperiment.experimentId)}`);
    }

    printSummary(settingsPath, {
      trackingUri,
      experimentId: resolvedExperiment.experimentId,
      experimentName: resolvedExperiment.experimentName,
    });
  } catch (error) {
    console.error(`${FAIL} ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  }
}

export async function runStatus(options: ConfigPathOptions = {}): Promise<void> {
  const config = getEffectiveTracingConfig(options);

  console.error(`\n${bold('MLflow Tracing Status')}\n`);
  console.error(`  Enabled: ${config.enabled ? cyan('true') : dim('false')}`);
  console.error(`  Source: ${cyan(config.source)}`);

  if (config.settingsPath) {
    console.error(`  Settings file: ${cyan(config.settingsPath)}`);
  }
  console.error(`  Tracking URI: ${config.trackingUri ? cyan(config.trackingUri) : dim('not set')}`);
  console.error(`  Experiment ID: ${config.experimentId ? cyan(config.experimentId) : dim('not set')}`);
  console.error(
    `  Experiment name: ${config.experimentName ? cyan(config.experimentName) : dim('not set')}`,
  );

  if (!config.enabled) {
    console.error(`\n${WARN} Tracing is disabled. Run ${bold('mlflow-claude-code setup')} to configure it.`);
  }
}
