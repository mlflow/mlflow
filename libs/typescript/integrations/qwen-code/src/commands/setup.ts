/**
 * `mlflow-qwen-code setup` — interactively register the Stop hook in the
 * user's Qwen Code settings and write an MLflow tracing config alongside it.
 *
 * Writes:
 *   - `settings.json` — registers the `mlflow-qwen-code stop-hook` entry
 *     under `hooks.Stop`; leaves unrelated fields untouched.
 *   - `mlflow-tracing.json` — persists the tracking URI and experiment ID
 *     so the hook can run without shell exports.
 *
 * Scope selection:
 *   - Interactive runs prompt for project-local (`./.qwen/`) vs user-level
 *     (`~/.qwen/`), defaulting to project-local.
 *   - `--project` / `-p` forces project-local and skips the prompt.
 *   - `--non-interactive` defaults to project-local, matching the interactive
 *     default.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { createInterface } from 'node:readline';

import { FAIL, OK, WARN, bold, cyan, dim } from '../ui.js';
import { selectPrompt } from '../ui-select.js';

const HOOK_COMMAND = 'mlflow-qwen-code stop-hook';
const DEFAULT_TRACKING_URI = 'http://localhost:5000';
const DEFAULT_EXPERIMENT_ID = '0';

/**
 * Returns true only when `raw` parses as a URL with an http(s) scheme.
 * Rejects scheme-less inputs like `localhost:5000` (which `new URL` otherwise
 * interprets as a custom scheme with an empty port).
 */
export function isValidTrackingUri(raw: string): boolean {
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    return false;
  }
  return parsed.protocol === 'http:' || parsed.protocol === 'https:';
}

interface QwenHookEntry {
  type: string;
  command: string;
}

interface QwenHookGroup {
  hooks: QwenHookEntry[];
}

interface QwenSettings {
  hooks?: {
    Stop?: QwenHookGroup[];
    [key: string]: QwenHookGroup[] | undefined;
  };
  [key: string]: unknown;
}

export interface SetupOptions {
  /** Override the user home directory. Defaults to `os.homedir()`. */
  home?: string;
  /** Override the current working directory. Defaults to `process.cwd()`. */
  cwd?: string;
  /** Pre-supplied tracking URI. Skips the interactive prompt. */
  trackingUri?: string;
  /** Pre-supplied experiment ID. Skips the interactive prompt. */
  experimentId?: string;
  /** Suppress prompts entirely. Uses defaults for any unset values. */
  nonInteractive?: boolean;
}

export function resolveSettingsPath(projectLocal: boolean, options: SetupOptions = {}): string {
  return projectLocal
    ? resolve(options.cwd ?? process.cwd(), '.qwen', 'settings.json')
    : resolve(options.home ?? homedir(), '.qwen', 'settings.json');
}

function resolveTracingConfigPath(projectLocal: boolean, options: SetupOptions = {}): string {
  return projectLocal
    ? resolve(options.cwd ?? process.cwd(), '.qwen', 'mlflow-tracing.json')
    : resolve(options.home ?? homedir(), '.qwen', 'mlflow-tracing.json');
}

function readSettings(path: string): QwenSettings | null {
  if (!existsSync(path)) {
    return {};
  }
  const content = readFileSync(path, 'utf-8').trim();
  if (!content) {
    return {};
  }
  try {
    return JSON.parse(content) as QwenSettings;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`${FAIL} Failed to parse ${cyan(path)}: ${msg}`);
    console.error(`  Fix the file manually and rerun ${cyan('mlflow-qwen-code setup')}.`);
    return null;
  }
}

function writeSettings(path: string, settings: QwenSettings): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(settings, null, 2) + '\n', 'utf-8');
}

function writeTracingConfig(
  path: string,
  config: { trackingUri: string; experimentId: string },
): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(config, null, 2) + '\n', 'utf-8');
}

function hasMlflowHook(groups: QwenHookGroup[]): boolean {
  return groups.some((group) => group.hooks?.some((hook) => hook.command?.trim() === HOOK_COMMAND));
}

/**
 * Parse raw CLI args for the setup command. Supports:
 *   --project / -p
 *   --non-interactive / -y
 *   --tracking-uri <url>
 *   --experiment-id <id>
 *
 * `projectLocal` is left undefined when `--project` is not passed so the
 * caller can apply its own default (interactive prompts; non-interactive
 * defaults to project-local).
 */
export function parseSetupArgs(
  args: string[],
): SetupOptions & { projectLocal: boolean | undefined } {
  const out: SetupOptions & { projectLocal: boolean | undefined } = { projectLocal: undefined };
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--project' || arg === '-p') {
      out.projectLocal = true;
    } else if (arg === '--non-interactive' || arg === '-y') {
      out.nonInteractive = true;
    } else if (arg === '--tracking-uri') {
      out.trackingUri = args[++i];
    } else if (arg === '--experiment-id') {
      out.experimentId = args[++i];
    }
  }
  return out;
}

type Readline = ReturnType<typeof createInterface>;

function askOn(
  rl: Readline,
  label: string,
  defaultValue: string,
  validate?: (value: string) => string | null,
): Promise<string> {
  return new Promise((resolvePromise) => {
    const ask = (): void => {
      rl.question(`  ${label} ${dim(`[${defaultValue}]`)} `, (answer) => {
        const value = answer.trim() || defaultValue;
        const err = validate?.(value);
        if (err) {
          console.error(`  ${FAIL} ${err}`);
          ask();
          return;
        }
        resolvePromise(value);
      });
    };
    ask();
  });
}

const validateTrackingUri = (value: string): string | null =>
  isValidTrackingUri(value) ? null : 'Must be an absolute http:// or https:// URL.';

function promptScope(): Promise<boolean> {
  return selectPrompt<boolean>({
    question: 'Where should MLflow tracing be installed?',
    options: [
      { value: true, label: 'Project', hint: './.qwen/' },
      { value: false, label: 'User', hint: '~/.qwen/' },
    ],
    defaultIndex: 0,
  });
}

export async function runSetup(args: string[], options: SetupOptions = {}): Promise<void> {
  const parsed = parseSetupArgs(args);
  const merged: SetupOptions & { projectLocal: boolean | undefined } = {
    ...parsed,
    ...options,
    projectLocal: parsed.projectLocal,
  };

  const interactive = !merged.nonInteractive;
  const needsBanner =
    interactive &&
    (merged.projectLocal === undefined || !merged.trackingUri || !merged.experimentId);
  if (needsBanner) {
    console.error(`\n${bold('Configure MLflow tracing for Qwen Code')}`);
  }

  const projectLocal =
    merged.projectLocal !== undefined
      ? merged.projectLocal
      : interactive
        ? await promptScope()
        : true;

  const settingsPath = resolveSettingsPath(projectLocal, merged);
  const tracingConfigPath = resolveTracingConfigPath(projectLocal, merged);

  const settings = readSettings(settingsPath);
  if (settings == null) {
    process.exitCode = 1;
    return;
  }
  settings.hooks ??= {};
  settings.hooks.Stop ??= [];

  if (hasMlflowHook(settings.hooks.Stop)) {
    console.error(`${WARN} Stop hook already registered in ${cyan(settingsPath)}`);
  } else {
    settings.hooks.Stop.push({
      hooks: [{ type: 'command', command: HOOK_COMMAND }],
    });
    writeSettings(settingsPath, settings);
    console.error(`${OK} Registered Stop hook in ${cyan(settingsPath)}`);
  }
  console.error('');

  const needsTextPrompt = interactive && (!merged.trackingUri || !merged.experimentId);
  const rl = needsTextPrompt
    ? createInterface({ input: process.stdin, output: process.stdout })
    : null;
  try {
    const trackingUri =
      merged.trackingUri ??
      (rl
        ? await askOn(rl, 'MLflow tracking URI', DEFAULT_TRACKING_URI, validateTrackingUri)
        : DEFAULT_TRACKING_URI);
    const experimentId =
      merged.experimentId ??
      (rl ? await askOn(rl, 'MLflow experiment ID', DEFAULT_EXPERIMENT_ID) : DEFAULT_EXPERIMENT_ID);

    if (!isValidTrackingUri(trackingUri)) {
      console.error(
        `${FAIL} Invalid tracking URI: ${bold(trackingUri)} — must be an absolute http:// or https:// URL.`,
      );
      process.exitCode = 1;
      return;
    }
    writeTracingConfig(tracingConfigPath, { trackingUri, experimentId });
    console.error(`\n${OK} Wrote tracing config to ${cyan(tracingConfigPath)}`);

    const port = new URL(trackingUri).port || '5000';
    console.error(`\n${bold('Next steps')}`);
    console.error('  1. Start the MLflow tracking server in a separate terminal:');
    console.error(`       ${cyan(`mlflow server --port ${port}`)}`);
    console.error(
      `  2. Launch ${cyan('qwen')} — traces appear at ${bold(trackingUri)} after each turn.`,
    );
    console.error(
      `\n${dim('Override per-shell with $MLFLOW_TRACKING_URI / $MLFLOW_EXPERIMENT_ID.')}`,
    );
  } finally {
    rl?.close();
  }
}
