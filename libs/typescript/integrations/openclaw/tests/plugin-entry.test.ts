import type { OpenClawPluginApi } from 'openclaw/plugin-sdk/plugin-entry';

import { registerMlflowCli } from '../src/configure';
import plugin from '../src/index';

jest.mock('openclaw/plugin-sdk/plugin-entry', () => ({
  definePluginEntry: (entry: unknown) => entry,
}));
jest.mock('../src/configure', () => ({ registerMlflowCli: jest.fn() }));

it('uses the host runtime config APIs for the MLflow CLI', async () => {
  const config = Object.freeze({ gateway: { port: 18789 } });
  const current = jest.fn(() => config);
  const mutateConfigFile = jest.fn(() => Promise.resolve());
  const registerCli = jest.fn<void, Parameters<OpenClawPluginApi['registerCli']>>();
  const api: OpenClawPluginApi = {
    runtime: { config: { current, mutateConfigFile } },
    registerService: jest.fn(),
    registerCli,
    on: jest.fn(),
  };
  const entry = plugin as { register: (api: OpenClawPluginApi) => void };
  entry.register(api);

  const [[register, options]] = registerCli.mock.calls;
  expect(options).toEqual({ commands: ['mlflow'] });
  const program = { command: jest.fn() };
  register({ program });

  const [[cli]] = jest.mocked(registerMlflowCli).mock.calls;
  expect(cli.program).toBe(program);
  expect(cli.loadConfig()).toBe(config);
  expect(current).toHaveBeenCalledTimes(1);

  const mutate = jest.fn();
  await cli.updateConfig(mutate);
  expect(mutateConfigFile).toHaveBeenCalledWith({ afterWrite: { mode: 'auto' }, mutate });
});
