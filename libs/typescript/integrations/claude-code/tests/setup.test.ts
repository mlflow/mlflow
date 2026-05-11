import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createInterface } from 'node:readline';

import { parseSetupArgs, runSetup } from '../src/commands/setup';
import { selectPrompt } from '../src/ui-select';

jest.mock('node:readline', () => ({ createInterface: jest.fn() }));
jest.mock('../src/ui-select', () => ({ selectPrompt: jest.fn() }));
jest.mock('../src/config', () => {
  const actual = jest.requireActual('../src/config');
  return {
    ...actual,
    resolveExperiment: jest.fn(async (_trackingUri: string, experimentId?: string, experimentName?: string) => ({
      experimentId: experimentId ?? 'resolved-id',
      experimentName,
      created: false,
    })),
  };
});

const createInterfaceMock = jest.mocked(createInterface);
const selectPromptMock = jest.mocked(selectPrompt);

function mockTextPrompts(answers: string[]): void {
  const queue = [...answers];
  createInterfaceMock.mockImplementation(
    () =>
      ({
        question: (_prompt: string, cb: (answer: string) => void) => {
          const answer = queue.shift();
          if (answer === undefined) {
            throw new Error('mockTextPrompts: ran out of answers');
          }
          cb(answer);
        },
        close: () => {},
      }) as unknown as ReturnType<typeof createInterface>,
  );
}

describe('mlflow-claude-code setup', () => {
  let tmpHome: string;
  let tmpCwd: string;
  let originalExitCode: number | string | undefined;

  beforeEach(() => {
    tmpHome = mkdtempSync(join(tmpdir(), 'claude-setup-home-'));
    tmpCwd = mkdtempSync(join(tmpdir(), 'claude-setup-cwd-'));
    originalExitCode = process.exitCode;
    process.exitCode = undefined;
    createInterfaceMock.mockReset();
    selectPromptMock.mockReset();
  });

  afterEach(() => {
    rmSync(tmpHome, { recursive: true, force: true });
    rmSync(tmpCwd, { recursive: true, force: true });
    process.exitCode = originalExitCode;
  });

  it('parses scope and experiment flags', () => {
    expect(
      parseSetupArgs([
        '--project',
        '--tracking-uri',
        'http://localhost:5000',
        '--experiment-name',
        'claude-code-traces',
      ]),
    ).toMatchObject({
      projectLocal: true,
      trackingUri: 'http://localhost:5000',
      experimentName: 'claude-code-traces',
    });
  });

  it('writes project settings in non-interactive mode with default experiment name', async () => {
    await runSetup(['--non-interactive'], { home: tmpHome, cwd: tmpCwd });

    const settingsPath = join(tmpCwd, '.claude', 'settings.json');
    expect(existsSync(settingsPath)).toBe(true);
    expect(JSON.parse(readFileSync(settingsPath, 'utf-8'))).toMatchObject({
      env: {
        MLFLOW_CLAUDE_TRACING_ENABLED: 'true',
        MLFLOW_TRACKING_URI: 'http://localhost:5000',
        MLFLOW_EXPERIMENT_ID: 'resolved-id',
        MLFLOW_EXPERIMENT_NAME: 'claude-code-traces',
      },
    });
  });

  it('writes user settings when --user is passed', async () => {
    await runSetup(
      [
        '--non-interactive',
        '--user',
        '--tracking-uri',
        'http://mlflow.example',
        '--experiment-id',
        '42',
      ],
      { home: tmpHome, cwd: tmpCwd },
    );

    expect(existsSync(join(tmpHome, '.claude', 'settings.json'))).toBe(true);
    expect(existsSync(join(tmpCwd, '.claude', 'settings.json'))).toBe(false);
  });

  it('prompts for scope and experiment selection in interactive mode', async () => {
    selectPromptMock.mockResolvedValueOnce(true).mockResolvedValueOnce('name');
    mockTextPrompts(['', 'my-experiment']);

    await runSetup([], { home: tmpHome, cwd: tmpCwd });

    expect(selectPromptMock).toHaveBeenCalledTimes(2);
    expect(JSON.parse(readFileSync(join(tmpCwd, '.claude', 'settings.json'), 'utf-8'))).toMatchObject({
      env: {
        MLFLOW_EXPERIMENT_NAME: 'my-experiment',
      },
    });
  });

  it('rejects conflicting experiment flags', async () => {
    await runSetup(
      [
        '--non-interactive',
        '--experiment-id',
        '1',
        '--experiment-name',
        'duplicate',
      ],
      { home: tmpHome, cwd: tmpCwd },
    );

    expect(process.exitCode).toBe(1);
  });
});
