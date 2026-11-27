/**
 * CLI registration for `openclaw mlflow configure` and `openclaw mlflow status`.
 */

import type { OpenClawConfig } from 'openclaw/plugin-sdk/plugin-entry';

const PLUGIN_ID = 'mlflow-openclaw';

type ConfigDeps = {
  loadConfig: () => OpenClawConfig;
  updateConfig: (mutate: (draft: OpenClawConfig) => void) => Promise<void>;
};

// Minimal structural type for the Commander.js `Command` object OpenClaw
// hands to `api.registerCli`. Only covers the subset we actually use.
export type CommanderLike = {
  command: (name: string) => CommandLike;
};

type CommandLike = {
  description: (d: string) => CommandLike;
  command: (name: string) => CommandLike;
  action: (fn: () => void | Promise<void>) => CommandLike;
};

type RegisterCliParams = {
  program: CommanderLike;
} & ConfigDeps;

type ClackPrompts = {
  intro: (msg: string) => void;
  outro: (msg: string) => void;
  cancel: (msg: string) => void;
  isCancel: (value: unknown) => boolean;
  confirm: (options: { message: string; initialValue: boolean }) => Promise<boolean | symbol>;
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
  conversationAccess: boolean;
  config: Record<string, unknown>;
} {
  const root = asObject(cfg);
  const plugins = asObject(root.plugins);
  const entries = asObject(plugins.entries);
  const entry = asObject(entries[PLUGIN_ID]);
  const config = asObject(entry.config);
  return {
    enabled: typeof entry.enabled === 'boolean' ? entry.enabled : undefined,
    conversationAccess: asObject(entry.hooks).allowConversationAccess === true,
    config,
  };
}

// Mutates `draft` in place: the host persists the draft it handed out.
function mergePluginEntry(
  draft: OpenClawConfig,
  config: Record<string, unknown>,
  grantConversationAccess: boolean,
): void {
  const plugins = asObject(draft.plugins);
  const entries = asObject(plugins.entries);
  const entry = asObject(entries[PLUGIN_ID]);
  entries[PLUGIN_ID] = {
    ...entry,
    enabled: true,
    ...(grantConversationAccess && {
      hooks: { ...asObject(entry.hooks), allowConversationAccess: true },
    }),
    config: { ...asObject(entry.config), ...config },
  };
  plugins.entries = entries;
  draft.plugins = plugins;
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

  let grantConversationAccess = false;
  if (!entry.conversationAccess) {
    const answer = await p.confirm({
      message: 'Allow MLflow to access conversation content and export it in traces?',
      initialValue: false,
    });
    if (p.isCancel(answer)) {
      p.cancel('Configuration cancelled');
      return;
    }
    grantConversationAccess = answer === true;
  }

  let conversationAccess = false;
  await deps.updateConfig((draft) => {
    mergePluginEntry(
      draft,
      { trackingUri: trackingUri as string, experimentId: experimentId as string },
      grantConversationAccess,
    );
    conversationAccess = getPluginEntry(draft).conversationAccess;
  });

  // A running gateway hot-reloads the plugin in the same process, but the
  // second tracing init there cannot replace the global OpenTelemetry
  // provider, so traces stop until the gateway restarts.
  const accessMessage = conversationAccess
    ? 'Conversation access is enabled.'
    : 'Conversation access is not enabled. OpenClaw will block the hooks required for tracing.';
  p.outro(`MLflow configuration saved. ${accessMessage} Restart the gateway to apply.`);
}

export function registerMlflowCli(params: RegisterCliParams): void {
  const { program, loadConfig, updateConfig } = params;
  const deps: ConfigDeps = { loadConfig, updateConfig };

  const root = program.command('mlflow').description('MLflow trace export integration');

  root
    .command('configure')
    .description('Interactive setup for MLflow trace export')
    // Return the promise: the host tears the plugin down once the action settles.
    .action(() => runConfigure(deps));

  root
    .command('status')
    .description('Show current MLflow configuration')
    .action(() => {
      showStatus(deps);
    });
}
