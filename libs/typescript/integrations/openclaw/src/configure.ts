/**
 * CLI registration for `openclaw mlflow configure` and `openclaw mlflow status`.
 */

import type { OpenClawConfig } from 'openclaw/plugin-sdk/plugin-entry';

const PLUGIN_ID = 'mlflow-openclaw';

type ConfigDeps = {
  loadConfig: () => OpenClawConfig;
  writeConfigFile: (cfg: OpenClawConfig) => Promise<void>;
};

// Minimal structural type for the Commander.js `Command` object OpenClaw
// hands to `api.registerCli`. Only covers the subset we actually use.
export type CommanderLike = {
  command: (name: string) => CommandLike;
};

type CommandLike = {
  description: (d: string) => CommandLike;
  command: (name: string) => CommandLike;
  action: (fn: () => void) => CommandLike;
};

type RegisterCliParams = {
  program: CommanderLike;
} & ConfigDeps;

type ClackPrompts = {
  intro: (msg: string) => void;
  outro: (msg: string) => void;
  cancel: (msg: string) => void;
  isCancel: (value: unknown) => boolean;
  text: (options: {
    message: string;
    placeholder?: string;
    initialValue?: string;
    validate?: (value: string) => string | void;
  }) => Promise<string | symbol>;
};

function asObject(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {};
  }
  return value as Record<string, unknown>;
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function getPluginEntry(cfg: OpenClawConfig): {
  enabled?: boolean;
  config: Record<string, unknown>;
} {
  const root = asObject(cfg);
  const plugins = asObject(root.plugins);
  const entries = asObject(plugins.entries);
  const entry = asObject(entries[PLUGIN_ID]);
  const config = asObject(entry.config);
  return {
    enabled: typeof entry.enabled === 'boolean' ? entry.enabled : undefined,
    config,
  };
}

function setPluginEntry(
  cfg: OpenClawConfig,
  config: Record<string, unknown>,
  enabled = true,
): OpenClawConfig {
  const root = asObject(cfg);
  const plugins = asObject(root.plugins);
  const entries = asObject(plugins.entries);
  entries[PLUGIN_ID] = { ...asObject(entries[PLUGIN_ID]), enabled, config };
  plugins.entries = entries;
  root.plugins = plugins;
  return root as OpenClawConfig;
}

function showStatus(deps: ConfigDeps): void {
  const cfg = deps.loadConfig();
  const entry = getPluginEntry(cfg);
  const trackingUri =
    asString(entry.config.trackingUri) || process.env.MLFLOW_TRACKING_URI || '(not set)';
  const experimentId =
    asString(entry.config.experimentId) || process.env.MLFLOW_EXPERIMENT_ID || '(not set)';

  const lines: string[] = [
    `  Enabled:        ${entry.enabled ?? 'not set'}`,
    `  Tracking URI:   ${trackingUri}`,
    `  Experiment ID:  ${experimentId}`,
  ];

  process.stdout.write(`MLflow status:\n\n${lines.join('\n')}\n`);
}

async function runConfigure(deps: ConfigDeps): Promise<void> {
  // Dynamic import to avoid requiring @clack/prompts at module load time
  const p = (await import('@clack/prompts')) as unknown as ClackPrompts;

  p.intro('MLflow Tracing configuration');

  const cfg = deps.loadConfig();
  const entry = getPluginEntry(cfg);

  const trackingUri = await p.text({
    message: 'MLflow Tracking URI',
    placeholder: 'http://localhost:5000',
    initialValue: asString(entry.config.trackingUri) || process.env.MLFLOW_TRACKING_URI || '',
    validate: (value: string) => {
      if (!value) {
        return 'Tracking URI is required';
      }
      try {
        new URL(value);
      } catch {
        return 'Invalid URL';
      }
    },
  });
  if (p.isCancel(trackingUri)) {
    p.cancel('Configuration cancelled');
    return;
  }

  const experimentId = await p.text({
    message: 'Experiment ID',
    placeholder: '0',
    initialValue: asString(entry.config.experimentId) || process.env.MLFLOW_EXPERIMENT_ID || '',
    validate: (value: string) => {
      if (!value) {
        return 'Experiment ID is required';
      }
    },
  });
  if (p.isCancel(experimentId)) {
    p.cancel('Configuration cancelled');
    return;
  }

  const updated = setPluginEntry(cfg, {
    ...entry.config,
    trackingUri: trackingUri as string,
    experimentId: experimentId as string,
  });
  await deps.writeConfigFile(updated);

  p.outro('MLflow configuration saved. Restart the gateway to apply.');
}

export function registerMlflowCli(params: RegisterCliParams): void {
  const { program, loadConfig, writeConfigFile } = params;
  const deps: ConfigDeps = { loadConfig, writeConfigFile };

  const root = program.command('mlflow').description('MLflow trace export integration');

  root
    .command('configure')
    .description('Interactive setup for MLflow trace export')
    .action(() => {
      void runConfigure(deps);
    });

  root
    .command('status')
    .description('Show current MLflow configuration')
    .action(() => {
      showStatus(deps);
    });
}
