import type { PluginInput } from '@opencode-ai/plugin';
import { init } from '@mlflow/core';
import { MLflowTracingPlugin } from '../src';

jest.mock('@mlflow/core', () => ({ init: jest.fn() }));

describe('OpenCode initialization failure', () => {
  const originalEnv = process.env;

  afterAll(() => {
    process.env = originalEnv;
  });

  it('warns once instead of silently dropping traces when the UC location is invalid', async () => {
    process.env = {
      ...originalEnv,
      MLFLOW_TRACKING_URI: 'databricks',
      MLFLOW_EXPERIMENT_ID: '123',
      MLFLOW_TRACE_LOCATION: 'invalid',
    };
    (init as jest.Mock).mockImplementation(() => {
      throw new Error('Invalid MLFLOW_TRACE_LOCATION: expected catalog.schema.table_prefix');
    });
    const warning = jest.spyOn(console, 'error').mockImplementation(() => {});
    const messages = jest.fn();
    const hooks = await MLflowTracingPlugin({
      client: { session: { messages } },
    } as unknown as PluginInput);
    const idleEvent = {
      event: { type: 'session.idle', properties: { sessionID: 'session-1' } },
    } as Parameters<NonNullable<typeof hooks.event>>[0];

    try {
      await hooks.event!(idleEvent);
      await hooks.event!(idleEvent);

      expect(init).toHaveBeenCalledTimes(2);
      expect(messages).not.toHaveBeenCalled();
      expect(warning).toHaveBeenCalledTimes(1);
      expect(warning).toHaveBeenCalledWith(
        '[mlflow] OpenCode tracing is disabled: Invalid MLFLOW_TRACE_LOCATION: expected catalog.schema.table_prefix',
      );
    } finally {
      warning.mockRestore();
    }
  });

  it('retries after a transient initialization failure and resumes message handling', async () => {
    process.env = {
      ...originalEnv,
      MLFLOW_TRACKING_URI: 'databricks',
      MLFLOW_EXPERIMENT_ID: '123',
    };
    delete process.env.MLFLOW_TRACE_LOCATION;
    (init as jest.Mock).mockReset();
    (init as jest.Mock).mockImplementationOnce(() => {
      throw new Error('temporary initialization failure');
    });
    const warning = jest.spyOn(console, 'error').mockImplementation(() => {});
    const messages = jest.fn().mockResolvedValue({ data: [] });
    const hooks = await MLflowTracingPlugin({
      client: { session: { messages } },
    } as unknown as PluginInput);
    const idleEvent = {
      event: { type: 'session.idle', properties: { sessionID: 'session-2' } },
    } as Parameters<NonNullable<typeof hooks.event>>[0];

    try {
      await hooks.event!(idleEvent);
      expect(init).toHaveBeenCalledTimes(1);
      expect(messages).not.toHaveBeenCalled();

      await hooks.event!(idleEvent);
      expect(init).toHaveBeenCalledTimes(2);
      expect(messages).toHaveBeenCalledTimes(1);

      await hooks.event!(idleEvent);
      expect(init).toHaveBeenCalledTimes(2);
      expect(messages).toHaveBeenCalledTimes(2);
      expect(warning).toHaveBeenCalledTimes(1);
      expect(warning).toHaveBeenCalledWith(
        '[mlflow] OpenCode tracing is disabled: temporary initialization failure',
      );
    } finally {
      warning.mockRestore();
    }
  });
});
