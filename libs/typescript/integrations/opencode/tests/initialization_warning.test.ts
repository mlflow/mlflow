import { createHarness, idle } from './support';

jest.mock('@mlflow/core', () => ({ init: jest.fn() }));

describe('OpenCode initialization failure', () => {
  const originalEnv = process.env;
  let harness: ReturnType<typeof createHarness>;

  beforeEach(() => {
    jest.resetModules();
    process.env = {
      ...originalEnv,
      MLFLOW_TRACKING_URI: 'databricks',
      MLFLOW_EXPERIMENT_ID: '123',
    };
    harness = createHarness();
  });

  afterEach(async () => {
    await harness.stop();
    process.env = originalEnv;
    jest.restoreAllMocks();
  });

  it('warns once and retries initialization when the UC location is invalid', async () => {
    process.env.MLFLOW_TRACE_LOCATION = 'invalid';
    const { init } = await import('@mlflow/core');
    (init as jest.Mock).mockImplementation(() => {
      throw new Error('Invalid MLFLOW_TRACE_LOCATION: expected catalog.schema.table_prefix');
    });
    const warning = jest.spyOn(console, 'error').mockImplementation(() => {});
    await harness.start((await import('../src')).default);
    await harness.send(idle());
    await harness.send(idle());
    expect(init).toHaveBeenCalledTimes(2);
    expect(harness.context).not.toHaveBeenCalled();
    expect(warning).toHaveBeenCalledTimes(1);
    expect(warning).toHaveBeenCalledWith(
      '[mlflow] OpenCode tracing is disabled: Invalid MLFLOW_TRACE_LOCATION: expected catalog.schema.table_prefix',
    );
  });

  it('resumes message handling after a transient initialization failure', async () => {
    delete process.env.MLFLOW_TRACE_LOCATION;
    const { init } = await import('@mlflow/core');
    (init as jest.Mock).mockImplementationOnce(() => {
      throw new Error('temporary initialization failure');
    });
    const warning = jest.spyOn(console, 'error').mockImplementation(() => {});
    await harness.start((await import('../src')).default);
    await harness.send(idle());
    expect(init).toHaveBeenCalledTimes(1);
    expect(harness.context).not.toHaveBeenCalled();
    await harness.send(idle());
    await harness.send(idle());
    expect(init).toHaveBeenCalledTimes(2);
    expect(harness.context).toHaveBeenCalledTimes(2);
    expect(warning).toHaveBeenCalledTimes(1);
  });
});
