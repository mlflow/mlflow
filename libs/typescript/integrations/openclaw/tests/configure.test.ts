import { cancel, confirm, outro } from '@clack/prompts';

import { registerMlflowCli, type CommanderLike } from '../src/configure';

const answers: string[] = [];
const cancelled = Symbol('cancelled');
const confirmationAnswers: Array<boolean | symbol> = [];

jest.mock('@clack/prompts', () => ({
  intro: jest.fn(),
  outro: jest.fn(),
  cancel: jest.fn(),
  isCancel: (value: unknown) => value === cancelled,
  confirm: jest.fn(({ initialValue }: { initialValue: boolean }) =>
    Promise.resolve(confirmationAnswers.shift() ?? initialValue),
  ),
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
  beforeEach(() => {
    answers.length = 0;
    confirmationAnswers.length = 0;
    jest.clearAllMocks();
  });

  it('grants conversation access and keeps other config', async () => {
    const config: Config = {
      gateway: { port: 18789 },
      plugins: {
        entries: {
          other: { enabled: true },
          'mlflow-openclaw': {
            hooks: { allowPromptInjection: false },
            config: { trackingUri: 'http://old:5000', experimentId: '1' },
          },
        },
      },
    };
    const { actions, updateConfig } = setup(config);
    answers.push('http://localhost:5000', '7');
    confirmationAnswers.push(true);

    await actions.get('configure')!();

    expect(updateConfig).toHaveBeenCalledTimes(1);
    expect(outro).toHaveBeenCalledWith(
      'MLflow configuration saved. Conversation access is enabled. Restart the gateway to apply.',
    );
    expect(config).toEqual({
      gateway: { port: 18789 },
      plugins: {
        entries: {
          other: { enabled: true },
          'mlflow-openclaw': {
            enabled: true,
            hooks: { allowPromptInjection: false, allowConversationAccess: true },
            config: { trackingUri: 'http://localhost:5000', experimentId: '7' },
          },
        },
      },
    });
  });

  it('defaults to yes and grants conversation access', async () => {
    const config: Config = {};
    const { actions } = setup(config);
    answers.push('http://localhost:5000', '0');

    await actions.get('configure')!();

    expect(confirm).toHaveBeenCalledWith({
      message: 'Allow MLflow to access conversation content and export it in traces?',
      initialValue: true,
    });
    expect(outro).toHaveBeenCalledWith(
      'MLflow configuration saved. Conversation access is enabled. Restart the gateway to apply.',
    );
    expect(config).toEqual({
      plugins: {
        entries: {
          'mlflow-openclaw': {
            enabled: true,
            hooks: { allowConversationAccess: true },
            config: { trackingUri: 'http://localhost:5000', experimentId: '0' },
          },
        },
      },
    });
  });

  it('creates the plugin entry without a conversation grant when declined', async () => {
    const config: Config = {};
    const { actions } = setup(config);
    answers.push('http://localhost:5000', '0');
    confirmationAnswers.push(false);

    await actions.get('configure')!();

    expect(outro).toHaveBeenCalledWith(
      'MLflow configuration saved. Conversation access is not enabled. OpenClaw will block the hooks required for tracing. Restart the gateway to apply.',
    );
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

  it.each([
    { existing: false, answer: false, expected: false, prompts: 1 },
    { existing: false, answer: true, expected: true, prompts: 1 },
    { existing: true, answer: false, expected: true, prompts: 0 },
  ])(
    'preserves $existing unless the user opts in (answer: $answer)',
    async ({ existing, answer, expected, prompts }) => {
      const hooks = { allowConversationAccess: existing, allowPromptInjection: false };
      const entry = { hooks };
      const config = { plugins: { entries: { 'mlflow-openclaw': entry } } };
      const { actions } = setup(config);
      answers.push('http://localhost:5000', '0');
      confirmationAnswers.push(answer);

      await actions.get('configure')!();

      expect(confirm).toHaveBeenCalledTimes(prompts);
      expect(config.plugins.entries['mlflow-openclaw'].hooks).toEqual({
        allowConversationAccess: expected,
        allowPromptInjection: false,
      });
    },
  );

  it('does not write config when the conversation access prompt is cancelled', async () => {
    const { actions, updateConfig } = setup({});
    answers.push('http://localhost:5000', '0');
    confirmationAnswers.push(cancelled);

    await actions.get('configure')!();

    expect(cancel).toHaveBeenCalledWith('Configuration cancelled');
    expect(updateConfig).not.toHaveBeenCalled();
    expect(outro).not.toHaveBeenCalled();
  });

  it('waits for the host write before announcing success', async () => {
    const { actions, updateConfig } = setup({});
    let startWrite!: () => void;
    let finishWrite!: () => void;
    const writeStarted = new Promise<void>((resolve) => {
      startWrite = resolve;
    });
    const writeFinished = new Promise<void>((resolve) => {
      finishWrite = resolve;
    });
    updateConfig.mockImplementationOnce(() => {
      startWrite();
      return writeFinished;
    });
    answers.push('http://localhost:5000', '0');

    const action = actions.get('configure')!();
    try {
      await writeStarted;
      expect(outro).not.toHaveBeenCalled();
    } finally {
      finishWrite();
      await action;
    }
    expect(outro).toHaveBeenCalledTimes(1);
  });
});

describe('openclaw mlflow status', () => {
  it('prints the stored configuration', async () => {
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

    await actions.get('status')!();

    const output = write.mock.calls.map(([chunk]) => String(chunk)).join('');
    write.mockRestore();
    expect(output).toContain('Enabled:        true');
    expect(output).toContain('Tracking URI:   http://localhost:5000');
    expect(output).toContain('Experiment ID:  7');
  });
});
