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

const HOOK_LINE = 'notify = ["mlflow-codex", "notify-hook"]';
const NON_INTERACTIVE = ['--non-interactive'];

describe('runSetup', () => {
  let tmpHome: string;
  let originalExitCode: number | string | undefined;

  beforeEach(() => {
    tmpHome = mkdtempSync(join(tmpdir(), 'codex-setup-test-'));
    originalExitCode = process.exitCode;
    process.exitCode = undefined;
  });

  afterEach(() => {
    rmSync(tmpHome, { recursive: true, force: true });
    process.exitCode = originalExitCode;
  });

  function readConfig(path: string): string {
    return readFileSync(path, 'utf-8');
  }

  function readJson(path: string): Record<string, unknown> {
    return JSON.parse(readFileSync(path, 'utf-8')) as Record<string, unknown>;
  }

  it('creates config.toml and mlflow-tracing.json with defaults when absent', async () => {
    await runSetup(NON_INTERACTIVE, { home: tmpHome, cwd: tmpHome });

    const configPath = join(tmpHome, '.codex', 'config.toml');
    const tracingPath = join(tmpHome, '.codex', 'mlflow-tracing.json');
    expect(existsSync(configPath)).toBe(true);
    expect(existsSync(tracingPath)).toBe(true);
    expect(readConfig(configPath)).toContain(HOOK_LINE);
    expect(readJson(tracingPath)).toEqual({
      trackingUri: 'http://localhost:5000',
      experimentId: '0',
    });
  });

  it('writes the supplied --tracking-uri and --experiment-id to mlflow-tracing.json', async () => {
    await runSetup(
      ['--non-interactive', '--tracking-uri', 'http://mlflow.example/', '--experiment-id', '42'],
      { home: tmpHome, cwd: tmpHome },
    );

    expect(readJson(join(tmpHome, '.codex', 'mlflow-tracing.json'))).toEqual({
      trackingUri: 'http://mlflow.example/',
      experimentId: '42',
    });
  });

  it('prepends the notify hook to an existing config that has no notify entry', async () => {
    const configPath = join(tmpHome, '.codex', 'config.toml');
    mkdirSync(join(tmpHome, '.codex'), { recursive: true });
    const existing = '[some.section]\nkey = "value"\n';
    writeFileSync(configPath, existing, 'utf-8');

    await runSetup(NON_INTERACTIVE, { home: tmpHome, cwd: tmpHome });

    const content = readConfig(configPath);
    expect(content).toContain(HOOK_LINE);
    expect(content).toContain('[some.section]');
    expect(content).toContain('key = "value"');
    // notify line must come before the section header so TOML parses correctly
    expect(content.indexOf('notify = ')).toBeLessThan(content.indexOf('[some.section]'));
  });

  it('is idempotent when the hook is already registered', async () => {
    await runSetup(NON_INTERACTIVE, { home: tmpHome, cwd: tmpHome });
    const configPath = join(tmpHome, '.codex', 'config.toml');
    const first = readConfig(configPath);

    await runSetup(NON_INTERACTIVE, { home: tmpHome, cwd: tmpHome });
    const second = readConfig(configPath);

    expect(second).toBe(first);
    expect(process.exitCode).toBeFalsy();
  });

  it('refuses to overwrite a different notify entry and signals exit 1', async () => {
    const configPath = join(tmpHome, '.codex', 'config.toml');
    mkdirSync(join(tmpHome, '.codex'), { recursive: true });
    writeFileSync(configPath, 'notify = ["some-other-tool"]\n', 'utf-8');

    await runSetup(NON_INTERACTIVE, { home: tmpHome, cwd: tmpHome });

    const content = readConfig(configPath);
    expect(content).toBe('notify = ["some-other-tool"]\n');
    expect(process.exitCode).toBe(1);
    // mlflow-tracing.json must not be written when we refuse to touch config.toml.
    expect(existsSync(join(tmpHome, '.codex', 'mlflow-tracing.json'))).toBe(false);
  });

  it('does not treat a comment mentioning "mlflow-codex" as an existing registration', async () => {
    const configPath = join(tmpHome, '.codex', 'config.toml');
    mkdirSync(join(tmpHome, '.codex'), { recursive: true });
    writeFileSync(
      configPath,
      '# TODO: switch to mlflow-codex\nnotify = ["some-other-tool"]\n',
      'utf-8',
    );

    await runSetup(NON_INTERACTIVE, { home: tmpHome, cwd: tmpHome });

    expect(process.exitCode).toBe(1);
    const content = readConfig(configPath);
    expect(content).toContain('notify = ["some-other-tool"]');
    expect(content).not.toContain(HOOK_LINE);
  });

  it('rejects a tracking URI that is missing an http(s) scheme and exits 1', async () => {
    await runSetup(
      ['--non-interactive', '--tracking-uri', 'localhost:5678', '--experiment-id', '48'],
      { home: tmpHome, cwd: tmpHome },
    );

    expect(process.exitCode).toBe(1);
    // The notify hook is still registered (that step happens before URI
    // validation and is idempotent), but the tracing config must not be
    // written with an invalid URI.
    expect(existsSync(join(tmpHome, '.codex', 'mlflow-tracing.json'))).toBe(false);
  });

  it('writes to ./.codex/ when --project is passed', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'codex-project-test-'));
    try {
      await runSetup([...NON_INTERACTIVE, '--project'], { home: tmpHome, cwd });
      expect(existsSync(join(cwd, '.codex', 'config.toml'))).toBe(true);
      expect(existsSync(join(cwd, '.codex', 'mlflow-tracing.json'))).toBe(true);
      expect(existsSync(join(tmpHome, '.codex', 'config.toml'))).toBe(false);
      expect(existsSync(join(tmpHome, '.codex', 'mlflow-tracing.json'))).toBe(false);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('interactive scope prompt picks project-local when the user selects Project', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'codex-interactive-project-'));
    try {
      selectPromptMock.mockResolvedValueOnce(true);
      mockTextPrompts(['', '']);
      await runSetup([], { home: tmpHome, cwd });

      expect(selectPromptMock).toHaveBeenCalledTimes(1);
      expect(existsSync(join(cwd, '.codex', 'config.toml'))).toBe(true);
      expect(existsSync(join(cwd, '.codex', 'mlflow-tracing.json'))).toBe(true);
      expect(existsSync(join(tmpHome, '.codex', 'config.toml'))).toBe(false);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('interactive scope prompt writes user-level when the user selects User', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'codex-interactive-user-'));
    try {
      selectPromptMock.mockResolvedValueOnce(false);
      mockTextPrompts(['', '']);
      await runSetup([], { home: tmpHome, cwd });

      expect(existsSync(join(tmpHome, '.codex', 'config.toml'))).toBe(true);
      expect(existsSync(join(tmpHome, '.codex', 'mlflow-tracing.json'))).toBe(true);
      expect(existsSync(join(cwd, '.codex', 'config.toml'))).toBe(false);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('skips the scope prompt when --project is explicitly passed', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'codex-interactive-flag-'));
    try {
      mockTextPrompts(['', '']);
      await runSetup(['--project'], { home: tmpHome, cwd });

      expect(selectPromptMock).not.toHaveBeenCalled();
      expect(existsSync(join(cwd, '.codex', 'config.toml'))).toBe(true);
      expect(existsSync(join(tmpHome, '.codex', 'config.toml'))).toBe(false);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('skips the scope prompt in --non-interactive mode and defaults to project-local', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'codex-non-interactive-scope-'));
    try {
      await runSetup(NON_INTERACTIVE, { home: tmpHome, cwd });

      expect(selectPromptMock).not.toHaveBeenCalled();
      expect(existsSync(join(cwd, '.codex', 'config.toml'))).toBe(true);
      expect(existsSync(join(tmpHome, '.codex', 'config.toml'))).toBe(false);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });
});
