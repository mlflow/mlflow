import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  ensureInitialized,
  getEffectiveTracingConfig,
  parseTraceLocation,
  resolveSettingsPath,
  writeTracingSettings,
} from '../src/config';

const initMock = jest.fn();
const getExperimentByNameMock = jest.fn();
const createExperimentMock = jest.fn();
const createAuthProviderMock = jest.fn((_options: unknown) => ({
  getHost: () => 'http://localhost:5000',
}));

jest.mock('@mlflow/core', () => ({
  // eslint-disable-next-line @typescript-eslint/no-unsafe-return
  init: jest.fn((...args: unknown[]) => initMock(...(args as Parameters<typeof initMock>))),
  createAuthProvider: jest.fn((...args: unknown[]) =>
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    createAuthProviderMock(...(args as Parameters<typeof createAuthProviderMock>)),
  ),
  MlflowClient: jest.fn().mockImplementation(() => ({
    getExperimentByName: jest.fn((...args: unknown[]) =>
      // eslint-disable-next-line @typescript-eslint/no-unsafe-return
      getExperimentByNameMock(...(args as Parameters<typeof getExperimentByNameMock>)),
    ),
    createExperiment: jest.fn((...args: unknown[]) =>
      // eslint-disable-next-line @typescript-eslint/no-unsafe-return
      createExperimentMock(...(args as Parameters<typeof createExperimentMock>)),
    ),
  })),
}));

describe('Claude Code tracing config', () => {
  let tmpHome: string;
  let tmpCwd: string;
  let originalCwd: string;

  beforeEach(() => {
    tmpHome = mkdtempSync(join(tmpdir(), 'claude-config-home-'));
    tmpCwd = mkdtempSync(join(tmpdir(), 'claude-config-cwd-'));
    originalCwd = process.cwd();
    delete process.env.MLFLOW_CLAUDE_TRACING_ENABLED;
    delete process.env.MLFLOW_TRACKING_URI;
    delete process.env.MLFLOW_EXPERIMENT_ID;
    delete process.env.MLFLOW_EXPERIMENT_NAME;
    delete process.env.MLFLOW_TRACE_LOCATION;
    delete process.env.MLFLOW_WORKSPACE;
    initMock.mockReset();
    getExperimentByNameMock.mockReset();
    createExperimentMock.mockReset();
    createAuthProviderMock.mockClear();
  });

  afterEach(() => {
    process.chdir(originalCwd);
    rmSync(tmpHome, { recursive: true, force: true });
    rmSync(tmpCwd, { recursive: true, force: true });
  });

  it('reads project settings and preserves experiment name when writing config', () => {
    const settingsPath = resolveSettingsPath(true, { home: tmpHome, cwd: tmpCwd });
    writeTracingSettings(settingsPath, {
      trackingUri: 'http://localhost:5000',
      experimentId: '42',
      experimentName: 'claude-code-traces',
    });

    const stored = JSON.parse(readFileSync(settingsPath, 'utf-8')) as {
      env: Record<string, string>;
    };
    expect(stored.env).toMatchObject({
      MLFLOW_CLAUDE_TRACING_ENABLED: 'true',
      MLFLOW_TRACKING_URI: 'http://localhost:5000',
      MLFLOW_EXPERIMENT_ID: '42',
      MLFLOW_EXPERIMENT_NAME: 'claude-code-traces',
    });

    expect(getEffectiveTracingConfig({ home: tmpHome, cwd: tmpCwd })).toMatchObject({
      enabled: true,
      trackingUri: 'http://localhost:5000',
      experimentId: '42',
      experimentName: 'claude-code-traces',
      source: 'project',
    });
  });

  it('lets environment variables override saved settings', () => {
    writeTracingSettings(resolveSettingsPath(false, { home: tmpHome, cwd: tmpCwd }), {
      trackingUri: 'http://saved.example',
      experimentId: '7',
      experimentName: 'saved-exp',
    });

    process.env.MLFLOW_TRACKING_URI = 'http://override.example';
    process.env.MLFLOW_EXPERIMENT_ID = '99';
    process.env.MLFLOW_CLAUDE_TRACING_ENABLED = 'true';

    expect(getEffectiveTracingConfig({ home: tmpHome, cwd: tmpCwd })).toMatchObject({
      enabled: true,
      trackingUri: 'http://override.example',
      experimentId: '99',
      experimentName: 'saved-exp',
      source: 'environment',
    });
  });

  it('resolves experiment name to an ID and initializes the SDK', async () => {
    writeTracingSettings(resolveSettingsPath(true, { home: tmpHome, cwd: tmpCwd }), {
      trackingUri: 'http://localhost:5000',
      experimentId: '',
      experimentName: 'claude-code-traces',
    });
    process.chdir(tmpCwd);
    getExperimentByNameMock.mockResolvedValueOnce({
      experimentId: '123',
      name: 'claude-code-traces',
    });

    await expect(ensureInitialized()).resolves.toBe(true);
    expect(createAuthProviderMock).toHaveBeenCalledWith({ trackingUri: 'http://localhost:5000' });
    expect(initMock).toHaveBeenCalledWith({
      trackingUri: 'http://localhost:5000',
      experimentId: '123',
    });
  });

  it('writes and reads workspace in tracing settings', () => {
    const settingsPath = resolveSettingsPath(true, { home: tmpHome, cwd: tmpCwd });
    writeTracingSettings(settingsPath, {
      trackingUri: 'http://localhost:5000',
      experimentId: '42',
      workspace: 'my-workspace',
    });

    const stored = JSON.parse(readFileSync(settingsPath, 'utf-8')) as {
      env: Record<string, string>;
    };
    expect(stored.env).toMatchObject({
      MLFLOW_WORKSPACE: 'my-workspace',
    });

    expect(getEffectiveTracingConfig({ home: tmpHome, cwd: tmpCwd })).toMatchObject({
      workspace: 'my-workspace',
    });
  });

  it('lets MLFLOW_WORKSPACE env var override saved workspace', () => {
    writeTracingSettings(resolveSettingsPath(false, { home: tmpHome, cwd: tmpCwd }), {
      trackingUri: 'http://saved.example',
      experimentId: '7',
      workspace: 'saved-ws',
    });

    process.env.MLFLOW_WORKSPACE = 'env-ws';

    expect(getEffectiveTracingConfig({ home: tmpHome, cwd: tmpCwd })).toMatchObject({
      workspace: 'env-ws',
    });
  });

  it('omits MLFLOW_WORKSPACE from settings when workspace is not provided', () => {
    const settingsPath = resolveSettingsPath(true, { home: tmpHome, cwd: tmpCwd });
    writeTracingSettings(settingsPath, {
      trackingUri: 'http://localhost:5000',
      experimentId: '42',
    });

    const stored = JSON.parse(readFileSync(settingsPath, 'utf-8')) as {
      env: Record<string, string>;
    };
    expect(stored.env).not.toHaveProperty('MLFLOW_WORKSPACE');

    expect(getEffectiveTracingConfig({ home: tmpHome, cwd: tmpCwd }).workspace).toBeUndefined();
  });

  it('passes workspace to init during ensureInitialized', async () => {
    writeTracingSettings(resolveSettingsPath(true, { home: tmpHome, cwd: tmpCwd }), {
      trackingUri: 'http://localhost:5000',
      experimentId: '42',
      workspace: 'test-ws',
    });
    process.chdir(tmpCwd);

    await expect(ensureInitialized()).resolves.toBe(true);
    expect(initMock).toHaveBeenCalledWith(
      expect.objectContaining({
        workspace: 'test-ws',
      }),
    );
  });

  it('creates the experiment when a configured experiment name does not exist', async () => {
    writeTracingSettings(resolveSettingsPath(true, { home: tmpHome, cwd: tmpCwd }), {
      trackingUri: 'http://localhost:5000',
      experimentId: '',
      experimentName: 'new-experiment',
    });
    process.chdir(tmpCwd);
    getExperimentByNameMock.mockResolvedValueOnce(null);
    createExperimentMock.mockResolvedValueOnce('456');

    await expect(ensureInitialized()).resolves.toBe(true);
    expect(createExperimentMock).toHaveBeenCalledWith('new-experiment');
    expect(initMock).toHaveBeenCalledWith({
      trackingUri: 'http://localhost:5000',
      experimentId: '456',
    });
  });

  it('parses a catalog.schema.table_prefix trace location', () => {
    expect(parseTraceLocation('cat.sch.pfx')).toEqual({
      catalogName: 'cat',
      schemaName: 'sch',
      tablePrefix: 'pfx',
    });
    expect(parseTraceLocation('  cat.sch.pfx  ')).toEqual({
      catalogName: 'cat',
      schemaName: 'sch',
      tablePrefix: 'pfx',
    });
    expect(parseTraceLocation(undefined)).toBeNull();
    expect(parseTraceLocation('')).toBeNull();
    expect(parseTraceLocation('cat.sch')).toBeNull();
    expect(parseTraceLocation('cat.sch.pfx.extra')).toBeNull();
    expect(parseTraceLocation('cat..pfx')).toBeNull();
  });

  it('persists and surfaces the trace location and passes it to init', async () => {
    const settingsPath = resolveSettingsPath(true, { home: tmpHome, cwd: tmpCwd });
    writeTracingSettings(settingsPath, {
      trackingUri: 'databricks',
      experimentId: '42',
      traceLocation: 'my_catalog.my_schema.my_prefix',
    });
    process.chdir(tmpCwd);

    const stored = JSON.parse(readFileSync(settingsPath, 'utf-8')) as {
      env: Record<string, string>;
    };
    expect(stored.env.MLFLOW_TRACE_LOCATION).toBe('my_catalog.my_schema.my_prefix');

    expect(getEffectiveTracingConfig({ home: tmpHome, cwd: tmpCwd })).toMatchObject({
      trackingUri: 'databricks',
      experimentId: '42',
      traceLocation: 'my_catalog.my_schema.my_prefix',
    });

    await expect(ensureInitialized()).resolves.toBe(true);
    expect(initMock).toHaveBeenCalledWith({
      trackingUri: 'databricks',
      experimentId: '42',
      traceLocation: {
        catalogName: 'my_catalog',
        schemaName: 'my_schema',
        tablePrefix: 'my_prefix',
      },
    });
  });

  it('refuses to initialize when the trace location is malformed', async () => {
    writeTracingSettings(resolveSettingsPath(true, { home: tmpHome, cwd: tmpCwd }), {
      trackingUri: 'databricks',
      experimentId: '42',
      traceLocation: 'not-a-valid-location',
    });
    process.chdir(tmpCwd);

    await expect(ensureInitialized()).resolves.toBe(false);
    expect(initMock).not.toHaveBeenCalled();
  });
});
