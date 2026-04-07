import { resolve } from 'path';

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

import { processTranscript } from '../src/tracing';
import { flushTraces } from '@mlflow/core';

const FIXTURES_DIR = resolve(__dirname, 'fixtures');

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

describe('processTranscript', () => {
  beforeEach(() => {
    spanCounter = 0;
    Object.keys(mockSpans).forEach((key) => delete mockSpans[key]);
    mockTraceInfo.traceMetadata = {};
    mockTraceInfo.requestPreview = undefined;
    mockTraceInfo.responsePreview = undefined;
    jest.clearAllMocks();
  });

  it('creates AGENT root with LLM child for basic transcript', async () => {
    await processTranscript(resolve(FIXTURES_DIR, 'basic.jsonl'), 'test-session');

    const root = getRootSpan();
    expect(root).toBeDefined();
    expect(root.name).toBe('qwen_code_conversation');
    expect(root.spanType).toBe('AGENT');

    const llmSpans = getSpansByType('LLM');
    expect(llmSpans.length).toBe(1);
    expect(llmSpans[0].parentId).toBe(root.spanId);
  });

  it('sets trace previews', async () => {
    await processTranscript(resolve(FIXTURES_DIR, 'basic.jsonl'), 'test-session');

    expect(mockTraceInfo.requestPreview).toBe('what is 2+2');
    expect(mockTraceInfo.responsePreview).toBe('4');
  });

  it('sets session metadata', async () => {
    await processTranscript(resolve(FIXTURES_DIR, 'basic.jsonl'), 'test-session');

    expect(mockTraceInfo.traceMetadata['mlflow.trace.session']).toBe('test-session');
    expect(mockTraceInfo.traceMetadata['mlflow.trace.user']).toBeDefined();
  });

  it('creates TOOL spans for tool call transcript', async () => {
    await processTranscript(resolve(FIXTURES_DIR, 'with-tool-call.jsonl'), 'test-session');

    const toolSpans = getSpansByType('TOOL');
    expect(toolSpans.length).toBe(1);
    expect(toolSpans[0].name).toBe('tool_list_directory');

    const llmSpans = getSpansByType('LLM');
    expect(llmSpans.length).toBe(1);
  });

  it('sets token usage on LLM spans', async () => {
    await processTranscript(resolve(FIXTURES_DIR, 'basic.jsonl'), 'test-session');

    const llmSpan = getSpansByType('LLM')[0];
    expect(llmSpan.setAttribute).toHaveBeenCalledWith(
      'mlflow.chat.tokenUsage',
      expect.objectContaining({
        input_tokens: 100,
        output_tokens: 10,
        total_tokens: 110,
      }),
    );
  });

  it('skips null transcript path', async () => {
    await processTranscript(null);
    expect(getSpans().length).toBe(0);
    expect(flushTraces).not.toHaveBeenCalled();
  });

  it('calls flushTraces', async () => {
    await processTranscript(resolve(FIXTURES_DIR, 'basic.jsonl'));
    expect(flushTraces).toHaveBeenCalled();
  });
});
