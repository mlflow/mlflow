import { registerMlflowCli, type CommanderLike } from '../src/configure';

const answers: string[] = [];

jest.mock('@clack/prompts', () => ({
  intro: jest.fn(),
  outro: jest.fn(),
  cancel: jest.fn(),
  isCancel: () => false,
  text: jest.fn(() => Promise.resolve(answers.shift())),
}));

type Config = Record<string, unknown>;

function setup(config: Config) {
  const actions = new Map<string, () => void | Promise<void>>();
  const command = (name: string) => {
    const cmd = {
      description: () => cmd,
      command,
      action: (fn: () => void | Promise<void>) => {
        actions.set(name, fn);
        return cmd;
      },
    };
    return cmd;
  };
  const updateConfig = jest.fn((mutate: (draft: Config) => void) => {
    mutate(config);
    return Promise.resolve();
  });
  registerMlflowCli({
    program: { command } as unknown as CommanderLike,
    // The host returns a frozen snapshot from `current()`.
    loadConfig: () => Object.freeze(structuredClone(config)),
    updateConfig,
  });
  return { actions, updateConfig };
}

describe('openclaw mlflow configure', () => {
  it('awaits the host mutation and keeps other config', async () => {
    const config: Config = {
      gateway: { port: 18789 },
      plugins: {
        entries: {
          other: { enabled: true },
          'mlflow-openclaw': {
            hooks: { allowConversationAccess: true },
            config: { trackingUri: 'http://old:5000', experimentId: '1' },
          },
        },
      },
    };
    const { actions, updateConfig } = setup(config);
    answers.push('http://localhost:5000', '7');

    await actions.get('configure')!();

    expect(updateConfig).toHaveBeenCalledTimes(1);
    expect(config).toEqual({
      gateway: { port: 18789 },
      plugins: {
        entries: {
          other: { enabled: true },
          'mlflow-openclaw': {
            enabled: true,
            hooks: { allowConversationAccess: true },
            config: { trackingUri: 'http://localhost:5000', experimentId: '7' },
          },
        },
      },
    });
  });

  it('creates the plugin entry in an empty config', async () => {
    const config: Config = {};
    const { actions } = setup(config);
    answers.push('http://localhost:5000', '0');

    await actions.get('configure')!();

    expect(config).toEqual({
      plugins: {
        entries: {
          'mlflow-openclaw': {
            enabled: true,
            config: { trackingUri: 'http://localhost:5000', experimentId: '0' },
          },
        },
      },
    });
  });
});

describe('openclaw mlflow status', () => {
  it('prints the stored configuration', () => {
    const { actions } = setup({
      plugins: {
        entries: {
          'mlflow-openclaw': {
            enabled: true,
            config: { trackingUri: 'http://localhost:5000', experimentId: '7' },
          },
        },
      },
    });
    const write = jest.spyOn(process.stdout, 'write').mockImplementation(() => true);

    actions.get('status')!();

    const output = write.mock.calls.map(([chunk]) => String(chunk)).join('');
    write.mockRestore();
    expect(output).toContain('Enabled:        true');
    expect(output).toContain('Tracking URI:   http://localhost:5000');
    expect(output).toContain('Experiment ID:  7');
  });
});
