import { tmpdir } from 'node:os';
import { join } from 'node:path';

type ProviderModule = typeof import('../../src/core/provider');

interface ProviderHarness {
  provider: ProviderModule;
  walExporterCtor: jest.Mock;
  syncExporterCtor: jest.Mock;
  resolveDaemonEntry: jest.Mock;
}

function loadProvider(): ProviderHarness {
  const walExporterCtor = jest.fn();
  const syncExporterCtor = jest.fn();
  const resolveDaemonEntry = jest.fn();

  let provider!: ProviderModule;
  jest.isolateModules(() => {
    jest.doMock('@opentelemetry/sdk-node', () => ({
      NodeSDK: jest.fn().mockImplementation(() => ({
        start: jest.fn(),
        shutdown: jest.fn().mockResolvedValue(undefined),
      })),
    }));
    jest.doMock('../../src/clients', () => ({
      MlflowClient: jest.fn(),
    }));
    jest.doMock('../../src/exporters/mlflow', () => ({
      MlflowSpanExporter: syncExporterCtor,
      MlflowSpanProcessor: jest.fn(),
    }));
    jest.doMock('../../src/exporters/uc_table', () => ({
      DatabricksUCTableSpanExporter: jest.fn(),
      DatabricksUCTableSpanProcessor: jest.fn(),
    }));
    jest.doMock('../../src/exporters/wal', () => ({
      MlflowWalSpanExporter: walExporterCtor,
    }));
    jest.doMock('../../src/exporters/wal/supervisor', () => ({
      resolveDaemonEntry,
    }));
    jest.doMock('../../src/core/config', () => ({
      getConfig: () => ({ trackingUri: 'http://localhost:5000', experimentId: '1' }),
      getAuthProvider: () => ({}),
    }));

    // eslint-disable-next-line @typescript-eslint/no-var-requires
    provider = require('../../src/core/provider') as ProviderModule;
  });

  return { provider, walExporterCtor, syncExporterCtor, resolveDaemonEntry };
}

describe('initializeSDK WAL daemon fallback', () => {
  const originalAsyncExport = process.env.MLFLOW_ENABLE_ASYNC_TRACE_LOGGING;

  beforeEach(() => {
    process.env.MLFLOW_ENABLE_ASYNC_TRACE_LOGGING = 'true';
  });

  afterEach(() => {
    if (originalAsyncExport === undefined) {
      delete process.env.MLFLOW_ENABLE_ASYNC_TRACE_LOGGING;
    } else {
      process.env.MLFLOW_ENABLE_ASYNC_TRACE_LOGGING = originalAsyncExport;
    }
    jest.restoreAllMocks();
  });

  it('falls back to sync export when the daemon entry cannot be resolved', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { provider, walExporterCtor, syncExporterCtor, resolveDaemonEntry } = loadProvider();
    resolveDaemonEntry.mockImplementation(() => {
      throw new Error("Cannot find module '@mlflow/core/package.json'");
    });

    provider.initializeSDK();

    expect(walExporterCtor).not.toHaveBeenCalled();
    expect(syncExporterCtor).toHaveBeenCalledTimes(1);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toContain('falling back to synchronous trace export');
  });

  it('falls back to sync export when the resolved daemon entry does not exist', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { provider, walExporterCtor, syncExporterCtor, resolveDaemonEntry } = loadProvider();
    resolveDaemonEntry.mockReturnValue(join(tmpdir(), 'missing-daemon.cjs'));

    provider.initializeSDK();

    expect(walExporterCtor).not.toHaveBeenCalled();
    expect(syncExporterCtor).toHaveBeenCalledTimes(1);
    expect(warnSpy).toHaveBeenCalledTimes(1);
  });

  it('uses the WAL exporter when the daemon entry exists', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { provider, walExporterCtor, syncExporterCtor, resolveDaemonEntry } = loadProvider();
    resolveDaemonEntry.mockReturnValue(__filename);

    provider.initializeSDK();

    expect(walExporterCtor).toHaveBeenCalledTimes(1);
    expect(syncExporterCtor).not.toHaveBeenCalled();
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('warns on every unavailable initialization when async export is enabled', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { provider, resolveDaemonEntry } = loadProvider();
    resolveDaemonEntry.mockReturnValue(join(tmpdir(), 'missing-daemon.cjs'));

    provider.initializeSDK();
    provider.initializeSDK();

    expect(warnSpy).toHaveBeenCalledTimes(2);
  });
});
