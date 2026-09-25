import { init, getConfig, resetConfig } from '../../src/core/config';
import { initializeSDK } from '../../src/core/provider';

jest.mock('../../src/core/provider', () => ({ initializeSDK: jest.fn() }));

describe('MLFLOW_TRACE_LOCATION configuration', () => {
  const originalEnv = process.env;
  const baseConfig = { trackingUri: 'databricks', experimentId: '123' };

  beforeEach(() => {
    process.env = {
      ...originalEnv,
      DATABRICKS_HOST: 'https://test-workspace.databricks.com',
      DATABRICKS_TOKEN: 'test-token',
    };
    delete process.env.MLFLOW_TRACE_LOCATION;
    resetConfig();
    jest.clearAllMocks();
  });

  afterAll(() => {
    process.env = originalEnv;
    resetConfig();
  });

  it('uses a three-part UC location from the environment', () => {
    process.env.MLFLOW_TRACE_LOCATION = ' cat . sch . prefix ';

    init(baseConfig);

    expect(getConfig().traceLocation).toEqual({
      catalogName: 'cat',
      schemaName: 'sch',
      tablePrefix: 'prefix',
    });
    expect(initializeSDK).toHaveBeenCalledTimes(1);
  });

  it.each(['', '  ', 'cat.sch', 'cat.sch.prefix.extra', 'cat..prefix', 'cat.sch.'])(
    'rejects invalid MLFLOW_TRACE_LOCATION %j before initializing',
    (value) => {
      process.env.MLFLOW_TRACE_LOCATION = value;

      expect(() => init(baseConfig)).toThrow(
        'Invalid MLFLOW_TRACE_LOCATION: expected catalog.schema.table_prefix',
      );
      expect(initializeSDK).not.toHaveBeenCalled();
      expect(() => getConfig()).toThrow('The MLflow Tracing client is not configured');
    },
  );

  it('prefers an explicit traceLocation over the environment', () => {
    process.env.MLFLOW_TRACE_LOCATION = 'invalid';
    const traceLocation = { catalogName: 'explicit', schemaName: 'sch', tablePrefix: 'prefix' };

    init({ ...baseConfig, traceLocation });

    expect(getConfig().traceLocation).toEqual(traceLocation);
  });

  it('keeps experiment-backed tracing when the variable is absent', () => {
    init(baseConfig);

    expect(getConfig().traceLocation).toBeUndefined();
  });
});
