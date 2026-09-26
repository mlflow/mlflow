/**
 * In-memory stand-in for `@mlflow/core`, shared by the transcript tracing tests.
 *
 * Usage (the factory runs lazily, so requiring this module inside it is safe):
 *
 *   jest.mock('@mlflow/core', () =>
 *     jest.requireActual('./helpers/mlflowCoreMock').createMlflowCoreMock(),
 *   );
 *
 * Every span lands in `mockSpans`; all spans share one trace whose info is
 * `mockTraceInfo`. Call `resetMlflowCoreMock()` in `beforeEach`.
 */

export interface MockSpan {
  name: string;
  traceId: string;
  spanId: string;
  parentId: string | null;
  spanType: string;
  inputs: any;
  outputs: any;
  attributes: Record<string, any>;
  startTimeNs?: number;
  endTimeNs?: number;
  exceptions: Error[];
}

export const mockSpans: Record<string, MockSpan> = {};

export const mockTraceInfo: {
  traceMetadata: Record<string, string>;
  tags: Record<string, string>;
  requestPreview?: string;
  responsePreview?: string;
} = {
  traceMetadata: {},
  tags: {},
};

let spanCounter = 0;

export function resetMlflowCoreMock(): void {
  for (const key of Object.keys(mockSpans)) {
    delete mockSpans[key];
  }
  spanCounter = 0;
  mockTraceInfo.traceMetadata = {};
  mockTraceInfo.tags = {};
  mockTraceInfo.requestPreview = undefined;
  mockTraceInfo.responsePreview = undefined;
}

export function getSpans() {
  return Object.values(mockSpans);
}

export function getSpansByType(type: string) {
  return getSpans().filter((s) => s.spanType === type);
}

export function getSpansByName(name: string) {
  return getSpans().filter((s) => s.name === name);
}

export function getChildSpans(parentId: string) {
  return getSpans().filter((s) => s.parentId === parentId);
}

export function createMlflowCoreMock() {
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
        startTimeNs: options.startTimeNs,
        endTimeNs: undefined as number | undefined,
        exceptions: [] as Error[],
        setAttribute: jest.fn((key: string, value: any) => {
          span.attributes[key] = value;
        }),
        getAttribute: jest.fn((key: string): unknown => span.attributes[key] as unknown),
        setOutputs: jest.fn((outputs: any) => {
          span.outputs = outputs;
        }),
        end: jest.fn((opts?: { endTimeNs?: number }) => {
          span.endTimeNs = opts?.endTimeNs;
        }),
        recordException: jest.fn((err: Error) => {
          span.exceptions.push(err);
        }),
      };
      mockSpans[id] = span;
      return span;
    }),
    flushTraces: jest.fn().mockResolvedValue(undefined),
    SpanType: {
      LLM: 'LLM',
      CHAIN: 'CHAIN',
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
      TOKEN_USAGE: 'mlflow.trace.tokenUsage',
    },
    TokenUsageKey: {
      INPUT_TOKENS: 'input_tokens',
      OUTPUT_TOKENS: 'output_tokens',
      TOTAL_TOKENS: 'total_tokens',
      CACHE_READ_INPUT_TOKENS: 'cache_read_input_tokens',
      CACHE_CREATION_INPUT_TOKENS: 'cache_creation_input_tokens',
    },
    InMemoryTraceManager: {
      getInstance: jest.fn(() => ({
        getTrace: jest.fn(() => ({
          info: mockTraceInfo,
          spanDict: new Map(Object.entries(mockSpans)),
        })),
      })),
    },
  };
}
