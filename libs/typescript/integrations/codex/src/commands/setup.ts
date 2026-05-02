/**
 * `mlflow-codex setup` — interactively register the notify hook in the user's
 * Codex config and write an MLflow tracing config alongside it.
 *
 * Writes:
 *   - `config.toml` — prepends `notify = ["mlflow-codex", "notify-hook"]`
 *     ahead of any `[section]` headers. Refuses to modify a pre-existing
 *     `notify = ...` entry to avoid mangling the user's config.
 *   - `mlflow-tracing.json` — persists the tracking URI and experiment ID
 *     so the hook can run without shell exports.
 *
 * Scope selection:
 *   - Interactive runs prompt for project-local (`./.codex/`) vs user-level
 *     (`~/.codex/`), defaulting to project-local.
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

const HOOK_LINE = 'notify = ["mlflow-codex", "notify-hook"]';
const NOTIFY_LINE_RE = /^\s*notify\s*=.*$/m;
const NOTIFY_HAS_MLFLOW_RE = /^\s*notify\s*=.*["']mlflow-codex["']/m;
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

export function resolveConfigPath(projectLocal: boolean, options: SetupOptions = {}): string {
  return projectLocal
    ? resolve(options.cwd ?? process.cwd(), '.codex', 'config.toml')
    : resolve(options.home ?? homedir(), '.codex', 'config.toml');
}

function resolveTracingConfigPath(projectLocal: boolean, options: SetupOptions = {}): string {
  return projectLocal
    ? resolve(options.cwd ?? process.cwd(), '.codex', 'mlflow-tracing.json')
    : resolve(options.home ?? homedir(), '.codex', 'mlflow-tracing.json');
}

function writeConfigWithHook(path: string, original: string | null): void {
  mkdirSync(dirname(path), { recursive: true });
  const prefix = `# Added by \`mlflow-codex setup\` — forwards each Codex turn to MLflow Tracing.\n${HOOK_LINE}\n`;
  const content = original ? prefix + '\n' + original : prefix;
  writeFileSync(path, content, 'utf-8');
}

function writeTracingConfig(
  path: string,
  config: { trackingUri: string; experimentId: string },
): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(config, null, 2) + '\n', 'utf-8');
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
      { value: true, label: 'Project', hint: './.codex/' },
      { value: false, label: 'User', hint: '~/.codex/' },
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
    console.error(`\n${bold('Configure MLflow tracing for Codex CLI')}`);
  }

  const projectLocal =
    merged.projectLocal !== undefined
      ? merged.projectLocal
      : interactive
        ? await promptScope()
        : true;

  const configPath = resolveConfigPath(projectLocal, merged);
  const tracingConfigPath = resolveTracingConfigPath(projectLocal, merged);

  let hookRegistered = false;
  if (!existsSync(configPath)) {
    writeConfigWithHook(configPath, null);
    console.error(`${OK} Created ${cyan(configPath)} with notify hook`);
    hookRegistered = true;
  } else {
    const content = readFileSync(configPath, 'utf-8');
    if (NOTIFY_LINE_RE.test(content)) {
      if (NOTIFY_HAS_MLFLOW_RE.test(content)) {
        console.error(`${WARN} Notify hook already registered in ${cyan(configPath)}`);
        hookRegistered = true;
      } else {
        console.error(`${FAIL} ${cyan(configPath)} already has a \`notify = ...\` entry.`);
        console.error(`  Update it manually to: ${bold(HOOK_LINE)}`);
        process.exitCode = 1;
        return;
      }
    } else {
      writeConfigWithHook(configPath, content);
      console.error(`${OK} Added notify hook to ${cyan(configPath)}`);
      hookRegistered = true;
    }
  }

  if (!hookRegistered) {
    return;
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

    writeTracingConfigIfValid(tracingConfigPath, trackingUri, experimentId);
  } finally {
    rl?.close();
  }
}

function writeTracingConfigIfValid(
  tracingConfigPath: string,
  trackingUri: string,
  experimentId: string,
): void {
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
    `  2. Launch ${cyan('codex')} — traces appear at ${bold(trackingUri)} after each turn.`,
  );
  console.error(
    `\n${dim('Override per-shell with $MLFLOW_TRACKING_URI / $MLFLOW_EXPERIMENT_ID.')}`,
  );
}
