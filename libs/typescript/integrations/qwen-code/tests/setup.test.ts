import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createInterface } from 'node:readline';

import { runSetup } from '../src/commands/setup';
import { selectPrompt } from '../src/ui-select';

jest.mock('node:readline', () => ({ createInterface: jest.fn() }));
jest.mock('../src/ui-select', () => ({ selectPrompt: jest.fn() }));
const createInterfaceMock = jest.mocked(createInterface);
const selectPromptMock = jest.mocked(selectPrompt);

function mockTextPrompts(answers: string[]): void {
  const queue = [...answers];
  createInterfaceMock.mockImplementation(
    () =>
      ({
        question: (_prompt: string, cb: (answer: string) => void) => {
          const next = queue.shift();
          if (next === undefined) {
            throw new Error('mockTextPrompts: ran out of answers');
          }
          cb(next);
        },
        close: () => {},
      }) as unknown as ReturnType<typeof createInterface>,
  );
}

beforeEach(() => {
  selectPromptMock.mockReset();
  createInterfaceMock.mockReset();
});

const HOOK_COMMAND = 'mlflow-qwen-code stop-hook';
const NON_INTERACTIVE = ['--non-interactive'];

describe('runSetup', () => {
  let tmpHome: string;

  beforeEach(() => {
    tmpHome = mkdtempSync(join(tmpdir(), 'qwen-setup-test-'));
  });

  afterEach(() => {
    rmSync(tmpHome, { recursive: true, force: true });
  });

  function read(path: string): Record<string, unknown> {
    return JSON.parse(readFileSync(path, 'utf-8')) as Record<string, unknown>;
  }

  it('creates settings.json and mlflow-tracing.json with defaults when absent', async () => {
    await runSetup(NON_INTERACTIVE, { home: tmpHome, cwd: tmpHome });

    const settingsPath = join(tmpHome, '.qwen', 'settings.json');
    const tracingPath = join(tmpHome, '.qwen', 'mlflow-tracing.json');
    expect(existsSync(settingsPath)).toBe(true);
    expect(existsSync(tracingPath)).toBe(true);
    expect(read(settingsPath)).toEqual({
      hooks: {
        Stop: [{ hooks: [{ type: 'command', command: HOOK_COMMAND }] }],
      },
    });
    expect(read(tracingPath)).toEqual({
      trackingUri: 'http://localhost:5000',
      experimentId: '0',
    });
  });

  it('writes the supplied --tracking-uri and --experiment-id to mlflow-tracing.json', async () => {
    await runSetup(
      ['--non-interactive', '--tracking-uri', 'http://mlflow.example/', '--experiment-id', '42'],
      { home: tmpHome, cwd: tmpHome },
    );

    expect(read(join(tmpHome, '.qwen', 'mlflow-tracing.json'))).toEqual({
      trackingUri: 'http://mlflow.example/',
      experimentId: '42',
    });
  });

  it('preserves unrelated fields when merging into an existing settings.json', async () => {
    const settingsPath = join(tmpHome, '.qwen', 'settings.json');
    mkdirSync(join(tmpHome, '.qwen'), { recursive: true });
    writeFileSync(
      settingsPath,
      JSON.stringify({ theme: 'dark', security: { auth: { selectedType: 'openai' } } }),
      'utf-8',
    );

    await runSetup(NON_INTERACTIVE, { home: tmpHome, cwd: tmpHome });

    const settings = read(settingsPath);
    expect(settings.theme).toBe('dark');
    expect(settings.security).toEqual({ auth: { selectedType: 'openai' } });
    expect((settings.hooks as { Stop: unknown }).Stop).toEqual([
      { hooks: [{ type: 'command', command: HOOK_COMMAND }] },
    ]);
  });

  it('is idempotent when the hook is already registered', async () => {
    await runSetup(NON_INTERACTIVE, { home: tmpHome, cwd: tmpHome });
    const settingsPath = join(tmpHome, '.qwen', 'settings.json');
    const firstSettings = readFileSync(settingsPath, 'utf-8');

    await runSetup(NON_INTERACTIVE, { home: tmpHome, cwd: tmpHome });
    const secondSettings = readFileSync(settingsPath, 'utf-8');

    expect(secondSettings).toBe(firstSettings);
  });

  it('appends alongside other Stop hooks without overwriting them', async () => {
    const settingsPath = join(tmpHome, '.qwen', 'settings.json');
    mkdirSync(join(tmpHome, '.qwen'), { recursive: true });
    writeFileSync(
      settingsPath,
      JSON.stringify({
        hooks: { Stop: [{ hooks: [{ type: 'command', command: 'some-other-hook' }] }] },
      }),
      'utf-8',
    );

    await runSetup(NON_INTERACTIVE, { home: tmpHome, cwd: tmpHome });

    const settings = read(settingsPath) as { hooks: { Stop: unknown[] } };
    expect(settings.hooks.Stop).toHaveLength(2);
    expect(settings.hooks.Stop).toEqual([
      { hooks: [{ type: 'command', command: 'some-other-hook' }] },
      { hooks: [{ type: 'command', command: HOOK_COMMAND }] },
    ]);
  });

  it('rejects a tracking URI that is missing an http(s) scheme and exits 1', async () => {
    const originalExitCode = process.exitCode;
    process.exitCode = undefined;

    try {
      await runSetup(
        ['--non-interactive', '--tracking-uri', 'localhost:5678', '--experiment-id', '48'],
        { home: tmpHome, cwd: tmpHome },
      );
      expect(process.exitCode).toBe(1);
      // The Stop hook is still registered (that step happens before URI
      // validation and is idempotent), but the tracing config must not be
      // written with an invalid URI.
      expect(existsSync(join(tmpHome, '.qwen', 'mlflow-tracing.json'))).toBe(false);
    } finally {
      process.exitCode = originalExitCode;
    }
  });

  it('writes to ./.qwen/ when --project is passed', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'qwen-project-test-'));
    try {
      await runSetup([...NON_INTERACTIVE, '--project'], { home: tmpHome, cwd });
      const projectPath = join(cwd, '.qwen', 'settings.json');
      const projectTracingPath = join(cwd, '.qwen', 'mlflow-tracing.json');
      expect(existsSync(projectPath)).toBe(true);
      expect(existsSync(projectTracingPath)).toBe(true);
      expect(existsSync(join(tmpHome, '.qwen', 'settings.json'))).toBe(false);
      expect(existsSync(join(tmpHome, '.qwen', 'mlflow-tracing.json'))).toBe(false);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('interactive scope prompt picks project-local when the user selects Project', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'qwen-interactive-project-'));
    try {
      selectPromptMock.mockResolvedValueOnce(true);
      mockTextPrompts(['', '']);
      await runSetup([], { home: tmpHome, cwd });

      expect(selectPromptMock).toHaveBeenCalledTimes(1);
      expect(existsSync(join(cwd, '.qwen', 'settings.json'))).toBe(true);
      expect(existsSync(join(cwd, '.qwen', 'mlflow-tracing.json'))).toBe(true);
      expect(existsSync(join(tmpHome, '.qwen', 'settings.json'))).toBe(false);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('interactive scope prompt writes user-level when the user selects User', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'qwen-interactive-user-'));
    try {
      selectPromptMock.mockResolvedValueOnce(false);
      mockTextPrompts(['', '']);
      await runSetup([], { home: tmpHome, cwd });

      expect(existsSync(join(tmpHome, '.qwen', 'settings.json'))).toBe(true);
      expect(existsSync(join(tmpHome, '.qwen', 'mlflow-tracing.json'))).toBe(true);
      expect(existsSync(join(cwd, '.qwen', 'settings.json'))).toBe(false);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('skips the scope prompt when --project is explicitly passed', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'qwen-interactive-flag-'));
    try {
      mockTextPrompts(['', '']);
      await runSetup(['--project'], { home: tmpHome, cwd });

      expect(selectPromptMock).not.toHaveBeenCalled();
      expect(existsSync(join(cwd, '.qwen', 'settings.json'))).toBe(true);
      expect(existsSync(join(tmpHome, '.qwen', 'settings.json'))).toBe(false);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('skips the scope prompt in --non-interactive mode and defaults to project-local', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'qwen-non-interactive-scope-'));
    try {
      await runSetup(NON_INTERACTIVE, { home: tmpHome, cwd });

      expect(selectPromptMock).not.toHaveBeenCalled();
      expect(existsSync(join(cwd, '.qwen', 'settings.json'))).toBe(true);
      expect(existsSync(join(tmpHome, '.qwen', 'settings.json'))).toBe(false);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('reports a friendly error and exits 1 when settings.json is malformed', async () => {
    const settingsPath = join(tmpHome, '.qwen', 'settings.json');
    mkdirSync(join(tmpHome, '.qwen'), { recursive: true });
    writeFileSync(settingsPath, '{ not valid json', 'utf-8');
    const originalExitCode = process.exitCode;
    process.exitCode = undefined;

    try {
      await runSetup(NON_INTERACTIVE, { home: tmpHome, cwd: tmpHome });
      expect(process.exitCode).toBe(1);
      expect(readFileSync(settingsPath, 'utf-8')).toBe('{ not valid json');
      // mlflow-tracing.json must not have been written when setup bails early.
      expect(existsSync(join(tmpHome, '.qwen', 'mlflow-tracing.json'))).toBe(false);
    } finally {
      process.exitCode = originalExitCode;
    }
  });
});
