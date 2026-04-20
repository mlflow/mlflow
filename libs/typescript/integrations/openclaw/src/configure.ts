/**
 * CLI registration for `openclaw mlflow configure` and `openclaw mlflow status`.
 */

import type { OpenClawConfig } from 'openclaw/plugin-sdk/plugin-entry';

const PLUGIN_ID = 'mlflow-openclaw';

type ConfigDeps = {
  loadConfig: () => OpenClawConfig;
  writeConfigFile: (cfg: OpenClawConfig) => Promise<void>;
};

type RegisterCliParams = {
  program: unknown;
} & ConfigDeps;

function asObject(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  return value as Record<string, unknown>;
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
  const lines: string[] = [];

  lines.push(`  Enabled:        ${entry.enabled ?? 'not set'}`);
  lines.push(
    `  Tracking URI:   ${entry.config.trackingUri || process.env.MLFLOW_TRACKING_URI || '(not set)'}`,
  );
  lines.push(
    `  Experiment ID:  ${entry.config.experimentId || process.env.MLFLOW_EXPERIMENT_ID || '(not set)'}`,
  );

  console.log('MLflow status:\n');
  console.log(lines.join('\n'));
}

async function runConfigure(deps: ConfigDeps): Promise<void> {
  // Dynamic import to avoid requiring @clack/prompts at module load time
  const p = await import('@clack/prompts');

  p.intro('MLflow Tracing configuration');

  const cfg = deps.loadConfig();
  const entry = getPluginEntry(cfg);

  const trackingUri = await p.text({
    message: 'MLflow Tracking URI',
    placeholder: 'http://localhost:5000',
    initialValue: (entry.config.trackingUri as string) || process.env.MLFLOW_TRACKING_URI || '',
    validate: (value) => {
      if (!value) return 'Tracking URI is required';
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
    initialValue: (entry.config.experimentId as string) || process.env.MLFLOW_EXPERIMENT_ID || '',
    validate: (value) => {
      if (!value) return 'Experiment ID is required';
    },
  });
  if (p.isCancel(experimentId)) {
    p.cancel('Configuration cancelled');
    return;
  }

  const updated = setPluginEntry(cfg, {
    ...entry.config,
    trackingUri,
    experimentId,
  });
  await deps.writeConfigFile(updated);

  p.outro('MLflow configuration saved. Restart the gateway to apply.');
}

export function registerMlflowCli(params: RegisterCliParams): void {
  const { program, loadConfig, writeConfigFile } = params;
  const deps: ConfigDeps = { loadConfig, writeConfigFile };

  const cmd = program as {
    command: (name: string) => {
      description: (d: string) => {
        command: (name: string) => {
          description: (d: string) => { action: (fn: () => void) => unknown };
        };
      };
    };
  };
  const root = cmd.command('mlflow').description('MLflow trace export integration');

  root
    .command('configure')
    .description('Interactive setup for MLflow trace export')
    .action(async () => {
      await runConfigure(deps);
    });

  root
    .command('status')
    .description('Show current MLflow configuration')
    .action(() => {
      showStatus(deps);
    });
}
