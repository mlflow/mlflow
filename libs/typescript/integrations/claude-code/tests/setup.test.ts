import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { parseSetupArgs, runSetup } from '../src/commands/setup';

jest.mock('../src/config', () => {
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
  const actual = jest.requireActual('../src/config');
  // eslint-disable-next-line @typescript-eslint/no-unsafe-return
  return {
    ...actual,
    resolveExperiment: jest.fn(
      (_trackingUri: string, experimentId?: string, experimentName?: string) =>
        Promise.resolve({
          experimentId: experimentId ?? 'resolved-id',
          experimentName,
          created: false,
        }),
    ),
  };
});

describe('mlflow-claude-code setup', () => {
  let tmpHome: string;
  let tmpCwd: string;
  let originalExitCode: number | string | undefined;

  beforeEach(() => {
    tmpHome = mkdtempSync(join(tmpdir(), 'claude-setup-home-'));
    tmpCwd = mkdtempSync(join(tmpdir(), 'claude-setup-cwd-'));
    originalExitCode = process.exitCode;
    process.exitCode = undefined;
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

  it('writes project settings from flags', async () => {
    await runSetup(
      [
        '--project',
        '--tracking-uri',
        'http://localhost:5000',
        '--experiment-name',
        'my-experiment',
      ],
      { home: tmpHome, cwd: tmpCwd },
    );

    const settingsPath = join(tmpCwd, '.claude', 'settings.json');
    expect(existsSync(settingsPath)).toBe(true);
    expect(JSON.parse(readFileSync(settingsPath, 'utf-8'))).toMatchObject({
      env: {
        MLFLOW_CLAUDE_TRACING_ENABLED: 'true',
        MLFLOW_TRACKING_URI: 'http://localhost:5000',
        MLFLOW_EXPERIMENT_ID: 'resolved-id',
        MLFLOW_EXPERIMENT_NAME: 'my-experiment',
      },
    });
  });

  it('writes user settings when --user is passed', async () => {
    await runSetup(['--user', '--tracking-uri', 'http://mlflow.example', '--experiment-id', '42'], {
      home: tmpHome,
      cwd: tmpCwd,
    });

    expect(existsSync(join(tmpHome, '.claude', 'settings.json'))).toBe(true);
    expect(existsSync(join(tmpCwd, '.claude', 'settings.json'))).toBe(false);
  });

  it('rejects missing scope flag', async () => {
    await runSetup(['--tracking-uri', 'http://localhost:5000', '--experiment-name', 'x'], {
      home: tmpHome,
      cwd: tmpCwd,
    });
    expect(process.exitCode).toBe(1);
  });

  it('rejects missing tracking URI', async () => {
    await runSetup(['--project', '--experiment-name', 'x'], { home: tmpHome, cwd: tmpCwd });
    expect(process.exitCode).toBe(1);
  });

  it('rejects missing experiment', async () => {
    await runSetup(['--project', '--tracking-uri', 'http://localhost:5000'], {
      home: tmpHome,
      cwd: tmpCwd,
    });
    expect(process.exitCode).toBe(1);
  });

  it('rejects conflicting experiment flags', async () => {
    await runSetup(
      [
        '--project',
        '--tracking-uri',
        'http://localhost:5000',
        '--experiment-id',
        '1',
        '--experiment-name',
        'duplicate',
      ],
      { home: tmpHome, cwd: tmpCwd },
    );

    expect(process.exitCode).toBe(1);
  });

  it('rejects invalid tracking URI', async () => {
    await runSetup(['--project', '--tracking-uri', 'not-a-uri', '--experiment-name', 'x'], {
      home: tmpHome,
      cwd: tmpCwd,
    });
    expect(process.exitCode).toBe(1);
  });
});
