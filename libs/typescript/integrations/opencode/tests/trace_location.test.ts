/**
 * Tests for Databricks Unity Catalog trace location support in the
 * MLflow OpenCode integration.
 */

import type { PluginInput } from '@opencode-ai/plugin';

interface PluginClient {
  session: {
    messages: jest.Mock;
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type EventParams = any;

jest.mock('@mlflow/core', () => {
  const mockSpan = {
    traceId: 'mock-trace-id',
    setAttribute: jest.fn(),
    setOutputs: jest.fn(),
    end: jest.fn(),
  };

  const mockTrace = {
    info: { requestPreview: '', responsePreview: '', traceMetadata: {} },
  };

  return {
    init: jest.fn(),
    startSpan: jest.fn(() => mockSpan),
    flushTraces: jest.fn().mockResolvedValue(undefined),
    SpanType: { LLM: 'LLM', TOOL: 'TOOL', AGENT: 'AGENT' },
    SpanAttributeKey: { TOKEN_USAGE: 'token_usage' },
    TraceMetadataKey: {
      TRACE_SESSION: 'mlflow.trace.session',
      TRACE_USER: 'mlflow.trace.user',
    },
    InMemoryTraceManager: {
      getInstance: jest.fn(() => ({ getTrace: jest.fn(() => mockTrace) })),
    },
  };
});

import { MLflowTracingPlugin, parseTraceLocation } from '../src';
import * as mlflowTracing from '@mlflow/core';

const createMockClient = (messagesData: unknown[] = []): PluginClient => ({
  session: {
    messages: jest.fn().mockResolvedValue({ data: messagesData }),
  },
});

const createPluginInput = (client: PluginClient): PluginInput =>
  ({ client }) as unknown as PluginInput;

const createSessionIdleEvent = (sessionID: string): EventParams => ({
  event: { type: 'session.idle', properties: { sessionID } },
});

const createUserMessage = (text: string) => ({
  info: { role: 'user', time: { created: 1000, completed: 1000 } },
  parts: [{ type: 'text', text }],
});

const createAssistantTextMessage = (text: string) => ({
  info: {
    role: 'assistant',
    modelID: 'claude-3-opus',
    providerID: 'anthropic',
    tokens: { input: 100, output: 50 },
    time: { created: 1100, completed: 2000 },
  },
  parts: [{ type: 'text', text }],
});

describe('parseTraceLocation', () => {
  it('parses a valid catalog.schema.table_prefix string', () => {
    expect(parseTraceLocation('my_catalog.my_schema.my_prefix')).toEqual({
      catalogName: 'my_catalog',
      schemaName: 'my_schema',
      tablePrefix: 'my_prefix',
    });
  });

  it('trims surrounding whitespace on each part', () => {
    expect(parseTraceLocation('  cat . sch . pref ')).toEqual({
      catalogName: 'cat',
      schemaName: 'sch',
      tablePrefix: 'pref',
    });
  });

  it.each([
    ['undefined', undefined],
    ['empty string', ''],
    ['whitespace only', '   '],
    ['two parts', 'catalog.schema'],
    ['four parts', 'a.b.c.d'],
    ['empty middle part', 'catalog..prefix'],
    ['trailing dot', 'catalog.schema.'],
  ])('returns null for %s', (_label, value) => {
    expect(parseTraceLocation(value)).toBeNull();
  });
});

describe('MLFLOW_TRACE_LOCATION wiring', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env = { ...originalEnv };
    delete process.env.MLFLOW_TRACKING_URI;
    delete process.env.MLFLOW_EXPERIMENT_ID;
    delete process.env.MLFLOW_TRACE_LOCATION;
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it('does not initialize when MLFLOW_TRACE_LOCATION is malformed', async () => {
    process.env.MLFLOW_TRACKING_URI = 'databricks';
    process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';
    process.env.MLFLOW_TRACE_LOCATION = 'not-a-valid-location';

    const messages = [createUserMessage('Hi'), createAssistantTextMessage('Hello')];
    const mockClient = createMockClient(messages);
    const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

    await hooks.event!(createSessionIdleEvent('bad-uc-session'));

    expect(mlflowTracing.init).not.toHaveBeenCalled();
    expect(mockClient.session.messages).not.toHaveBeenCalled();
  });

  it('passes the parsed UC trace location to init', async () => {
    process.env.MLFLOW_TRACKING_URI = 'databricks';
    process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';
    process.env.MLFLOW_TRACE_LOCATION = 'my_catalog.my_schema.my_prefix';

    const messages = [createUserMessage('Hi'), createAssistantTextMessage('Hello')];
    const mockClient = createMockClient(messages);
    const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

    await hooks.event!(createSessionIdleEvent('uc-session'));

    expect(mlflowTracing.init).toHaveBeenCalledWith({
      trackingUri: 'databricks',
      experimentId: 'exp-123',
      traceLocation: {
        catalogName: 'my_catalog',
        schemaName: 'my_schema',
        tablePrefix: 'my_prefix',
      },
    });
  });
});
