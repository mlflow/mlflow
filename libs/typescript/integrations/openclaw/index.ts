import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';
import { emptyPluginConfigSchema } from 'openclaw/plugin-sdk';
import { createMLflowService } from './src/service.js';
import { registerMlflowCli } from './src/configure.js';

const plugin = {
  id: 'mlflow-openclaw',
  name: 'MLflow Tracing',
  description: 'Export OpenClaw LLM traces to MLflow',
  configSchema: emptyPluginConfigSchema(),
  register(api: OpenClawPluginApi) {
    api.registerService(createMLflowService(api));
    api.registerCli(
      ({ program }) => {
        registerMlflowCli({
          program,
          loadConfig: api.runtime.config.loadConfig,
          writeConfigFile: api.runtime.config.writeConfigFile,
        });
      },
      { commands: ['mlflow'] },
    );
  },
};

export default plugin;
