import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { parseTraceLocation, resolveTracingConfig } from '../src/config';

jest.mock('@mlflow/core', () => ({ init: jest.fn() }));

describe('parseTraceLocation', () => {
  it('parses a catalog.schema.table_prefix value', () => {
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
  });

  it('returns null for empty or malformed values', () => {
    expect(parseTraceLocation(undefined)).toBeNull();
    expect(parseTraceLocation('')).toBeNull();
    expect(parseTraceLocation('cat.sch')).toBeNull();
    expect(parseTraceLocation('cat.sch.pfx.extra')).toBeNull();
    expect(parseTraceLocation('cat..pfx')).toBeNull();
  });
});

describe('resolveTracingConfig', () => {
  let tmpHome: string;
  let tmpCwd: string;

  beforeEach(() => {
    tmpHome = mkdtempSync(join(tmpdir(), 'codex-config-home-'));
    tmpCwd = mkdtempSync(join(tmpdir(), 'codex-config-cwd-'));
    delete process.env.MLFLOW_TRACKING_URI;
    delete process.env.MLFLOW_EXPERIMENT_ID;
    delete process.env.MLFLOW_TRACE_LOCATION;
  });

  afterEach(() => {
    rmSync(tmpHome, { recursive: true, force: true });
    rmSync(tmpCwd, { recursive: true, force: true });
    delete process.env.MLFLOW_TRACKING_URI;
    delete process.env.MLFLOW_EXPERIMENT_ID;
    delete process.env.MLFLOW_TRACE_LOCATION;
  });

  function writeUserConfig(config: Record<string, string>): void {
    mkdirSync(join(tmpHome, '.codex'), { recursive: true });
    writeFileSync(
      join(tmpHome, '.codex', 'mlflow-tracing.json'),
      JSON.stringify(config),
      'utf-8',
    );
  }

  it('reads the trace location from mlflow-tracing.json', () => {
    writeUserConfig({
      trackingUri: 'databricks',
      experimentId: '42',
      traceLocation: 'my_catalog.my_schema.my_prefix',
    });

    expect(resolveTracingConfig({ home: tmpHome, cwd: tmpCwd })).toEqual({
      trackingUri: 'databricks',
      experimentId: '42',
      traceLocation: 'my_catalog.my_schema.my_prefix',
    });
  });

  it('prefers MLFLOW_TRACE_LOCATION env over the config file', () => {
    writeUserConfig({
      trackingUri: 'databricks',
      experimentId: '42',
      traceLocation: 'file_catalog.file_schema.file_prefix',
    });
    process.env.MLFLOW_TRACE_LOCATION = 'env_catalog.env_schema.env_prefix';

    expect(resolveTracingConfig({ home: tmpHome, cwd: tmpCwd })).toMatchObject({
      traceLocation: 'env_catalog.env_schema.env_prefix',
    });
  });

  it('leaves the trace location undefined when not configured', () => {
    writeUserConfig({ trackingUri: 'http://localhost:5000', experimentId: '0' });

    expect(resolveTracingConfig({ home: tmpHome, cwd: tmpCwd }).traceLocation).toBeUndefined();
  });
});

describe('ensureInitialized', () => {
  let tmpHome: string;

  beforeEach(() => {
    tmpHome = mkdtempSync(join(tmpdir(), 'codex-init-home-'));
    delete process.env.MLFLOW_TRACKING_URI;
    delete process.env.MLFLOW_EXPERIMENT_ID;
    delete process.env.MLFLOW_TRACE_LOCATION;
    jest.resetModules();
  });

  afterEach(() => {
    rmSync(tmpHome, { recursive: true, force: true });
    delete process.env.MLFLOW_TRACKING_URI;
    delete process.env.MLFLOW_EXPERIMENT_ID;
    delete process.env.MLFLOW_TRACE_LOCATION;
  });

  it('passes a parsed UC trace location to init', () => {
    process.env.MLFLOW_TRACKING_URI = 'databricks';
    process.env.MLFLOW_EXPERIMENT_ID = '42';
    process.env.MLFLOW_TRACE_LOCATION = 'my_catalog.my_schema.my_prefix';

    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { init } = require('@mlflow/core') as { init: jest.Mock };
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { ensureInitialized } = require('../src/config') as {
      ensureInitialized: () => boolean;
    };

    expect(ensureInitialized()).toBe(true);
    expect(init).toHaveBeenCalledWith({
      trackingUri: 'databricks',
      experimentId: '42',
      traceLocation: {
        catalogName: 'my_catalog',
        schemaName: 'my_schema',
        tablePrefix: 'my_prefix',
      },
    });
  });

  it('omits traceLocation when not configured', () => {
    process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
    process.env.MLFLOW_EXPERIMENT_ID = '0';
    // Set to empty (not deleted) so a stray ~/.codex/mlflow-tracing.json on the
    // developer machine can't leak a trace location into this assertion.
    process.env.MLFLOW_TRACE_LOCATION = '';

    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { init } = require('@mlflow/core') as { init: jest.Mock };
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { ensureInitialized } = require('../src/config') as {
      ensureInitialized: () => boolean;
    };

    expect(ensureInitialized()).toBe(true);
    expect(init).toHaveBeenCalledWith({
      trackingUri: 'http://localhost:5000',
      experimentId: '0',
    });
  });

  it('refuses to initialize when the trace location is malformed', () => {
    process.env.MLFLOW_TRACKING_URI = 'databricks';
    process.env.MLFLOW_EXPERIMENT_ID = '42';
    process.env.MLFLOW_TRACE_LOCATION = 'not-a-valid-location';

    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { init } = require('@mlflow/core') as { init: jest.Mock };
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { ensureInitialized } = require('../src/config') as {
      ensureInitialized: () => boolean;
    };

    expect(ensureInitialized()).toBe(false);
    expect(init).not.toHaveBeenCalled();
  });
});
