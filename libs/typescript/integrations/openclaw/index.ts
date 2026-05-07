import { definePluginEntry, type OpenClawPluginApi } from 'openclaw/plugin-sdk/plugin-entry';
import { createMLflowService } from './src/service.js';
import { registerMlflowCli, type CommanderLike } from './src/configure.js';

function parsePluginConfig(raw: unknown): Record<string, unknown> {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return {};
  }
  return raw as Record<string, unknown>;
}

export default definePluginEntry({
  id: 'mlflow-openclaw',
  name: 'MLflow Tracing',
  description: 'Export OpenClaw LLM traces to MLflow',
  register(api: OpenClawPluginApi) {
    const pluginConfig = parsePluginConfig(api.pluginConfig);
    const service = createMLflowService(api, pluginConfig);
    service.registerHooks();
    api.registerService(service);
    api.registerCli(
      ({ program }) => {
        registerMlflowCli({
          program: program as CommanderLike,
          loadConfig: api.runtime.config.loadConfig,
          writeConfigFile: api.runtime.config.writeConfigFile,
        });
      },
      { commands: ['mlflow'] },
    );
  },
});
