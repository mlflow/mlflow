import type { OpenClawPluginApi } from 'openclaw/plugin-sdk/plugin-entry';

jest.mock('@mlflow/core', () => ({
  init: jest.fn(),
  startSpan: jest.fn(),
  flushTraces: jest.fn(),
  SpanStatusCode: {},
}));

import { init } from '@mlflow/core';
import { createMLflowService } from '../src/service';

describe('createMLflowService', () => {
  // The gateway process hosts provider SDKs that emit their own OpenTelemetry
  // spans. With a global MLflow tracer provider each of them becomes a trace.
  it('keeps the MLflow tracer provider out of the global OpenTelemetry API', () => {
    const api = {
      on: jest.fn(),
      logger: { info: jest.fn(), warn: jest.fn(), error: jest.fn(), debug: jest.fn() },
    } as unknown as OpenClawPluginApi;

    createMLflowService(api, {
      trackingUri: 'http://localhost:5000',
      experimentId: '0',
    }).registerHooks();

    expect(init).toHaveBeenCalledWith({
      trackingUri: 'http://localhost:5000',
      experimentId: '0',
      registerGlobalTracerProvider: false,
    });
  });
});
