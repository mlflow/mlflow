// Track mock spans
let spanCounter = 0;
const mockSpans: Record<string, any> = {};
const mockTraceInfo: {
  traceMetadata: Record<string, string>;
  requestPreview?: string;
  responsePreview?: string;
} = {
  traceMetadata: {},
};

jest.mock('@mlflow/core', () => {
  return {
    init: jest.fn(),
    startSpan: jest.fn((options: any) => {
      const id = `span-${++spanCounter}`;
      const parentId = options.parent ? options.parent.spanId : null;
      const span = {
        name: options.name,
        traceId: 'mock-trace-id',
        spanId: id,
        parentId,
        spanType: options.spanType ?? 'UNKNOWN',
        inputs: options.inputs ?? {},
        outputs: {},
        attributes: { ...(options.attributes ?? {}) },
        setAttribute: jest.fn((key: string, value: any) => {
          span.attributes[key] = value;
        }),
        end: jest.fn((opts?: any) => {
          if (opts?.outputs) {
            span.outputs = opts.outputs;
          }
        }),
      };
      mockSpans[id] = span;
      return span;
    }),
    flushTraces: jest.fn().mockResolvedValue(undefined),
    SpanType: {
      LLM: 'LLM',
      AGENT: 'AGENT',
      TOOL: 'TOOL',
      UNKNOWN: 'UNKNOWN',
    },
    SpanAttributeKey: {
      TOKEN_USAGE: 'mlflow.chat.tokenUsage',
      MESSAGE_FORMAT: 'mlflow.message.format',
    },
    TraceMetadataKey: {
      TRACE_SESSION: 'mlflow.trace.session',
      TRACE_USER: 'mlflow.trace.user',
    },
    TokenUsageKey: {
      INPUT_TOKENS: 'input_tokens',
      OUTPUT_TOKENS: 'output_tokens',
      TOTAL_TOKENS: 'total_tokens',
    },
    InMemoryTraceManager: {
      getInstance: jest.fn(() => ({
        getTrace: jest.fn(() => ({
          info: mockTraceInfo,
        })),
      })),
    },
  };
});

import { processNotify } from '../src/tracing';
import { flushTraces } from '@mlflow/core';
import type { NotifyPayload } from '../src/types';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getSpans(): any[] {
  return Object.values(mockSpans);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getSpansByType(type: string): any[] {
  return getSpans().filter((s) => s.spanType === type);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getRootSpan(): any {
  return getSpans().find((s) => s.parentId == null);
}

function makeNotifyPayload(overrides?: Partial<NotifyPayload>): NotifyPayload {
  return {
    type: 'agent-turn-complete',
    'thread-id': 'test-thread-001',
    'turn-id': 'test-turn-001',
    cwd: '/tmp/test',
    client: 'codex-tui',
    'input-messages': ['what is 2+2'],
    'last-assistant-message': '4',
    ...overrides,
  };
}

describe('processNotify', () => {
  beforeEach(() => {
    spanCounter = 0;
    Object.keys(mockSpans).forEach((key) => delete mockSpans[key]);
    mockTraceInfo.traceMetadata = {};
    mockTraceInfo.requestPreview = undefined;
    mockTraceInfo.responsePreview = undefined;
    jest.clearAllMocks();
  });

  it('creates an AGENT root span with LLM child', async () => {
    await processNotify(makeNotifyPayload());

    const root = getRootSpan();
    expect(root).toBeDefined();
    expect(root.name).toBe('codex_conversation');
    expect(root.spanType).toBe('AGENT');

    const llmSpans = getSpansByType('LLM');
    expect(llmSpans.length).toBe(1);
    expect(llmSpans[0].parentId).toBe(root.spanId);
  });

  it('sets trace previews from notify payload', async () => {
    await processNotify(makeNotifyPayload());

    expect(mockTraceInfo.requestPreview).toBe('what is 2+2');
    expect(mockTraceInfo.responsePreview).toBe('4');
  });

  it('sets session metadata', async () => {
    await processNotify(makeNotifyPayload());

    expect(mockTraceInfo.traceMetadata['mlflow.trace.session']).toBe('test-thread-001');
    expect(mockTraceInfo.traceMetadata['mlflow.trace.user']).toBeDefined();
  });

  it('uses last input message only', async () => {
    await processNotify(
      makeNotifyPayload({
        'input-messages': ['first prompt', 'second prompt', 'third prompt'],
        'last-assistant-message': 'response to third',
      }),
    );

    expect(mockTraceInfo.requestPreview).toBe('third prompt');
    expect(mockTraceInfo.responsePreview).toBe('response to third');
  });

  it('skips when no user prompt', async () => {
    await processNotify(makeNotifyPayload({ 'input-messages': [] }));

    expect(getSpans().length).toBe(0);
    expect(flushTraces).not.toHaveBeenCalled();
  });

  it('calls flushTraces after processing', async () => {
    await processNotify(makeNotifyPayload());
    expect(flushTraces).toHaveBeenCalled();
  });

  it('includes response in root span outputs', async () => {
    await processNotify(makeNotifyPayload());

    const root = getRootSpan();
    expect(root.end).toHaveBeenCalled();
    const endCall = (root.end as jest.Mock).mock.calls[0][0];
    expect(endCall.outputs.response).toBe('4');
    expect(endCall.outputs.status).toBe('completed');
  });
});
