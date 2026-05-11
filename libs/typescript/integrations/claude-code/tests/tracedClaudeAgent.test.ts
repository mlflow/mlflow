/**
 * Tests for createTracedQuery (Claude Agent SDK wrapper) and the
 * LiveTracingContext state machine it drives.
 */

// ============================================================================
// Mock @mlflow/core
// ============================================================================

interface MockSpan {
  name: string;
  spanId: string;
  parentId: string | null;
  spanType: string;
  inputs: Record<string, unknown>;
  outputs: unknown;
  attributes: Record<string, unknown>;
  status: { code: string; description?: string } | null;
  ended: boolean;
  traceId: string;
  setAttribute: (key: string, value: unknown) => void;
  setInputs: (inputs: Record<string, unknown>) => void;
  setOutputs: (outputs: unknown) => void;
  setStatus: (code: string, description?: string) => void;
  end: () => void;
}

const mockSpans: Record<string, MockSpan> = {};
let spanCounter = 0;

function resetMocks() {
  for (const key of Object.keys(mockSpans)) {
    delete mockSpans[key];
  }
  spanCounter = 0;
  mockTraceInfo.traceMetadata = {};
  mockTraceInfo.requestPreview = undefined;
  mockTraceInfo.responsePreview = undefined;
}

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
      const span: MockSpan = {
        name: options.name,
        spanId: id,
        parentId,
        spanType: options.spanType ?? 'UNKNOWN',
        inputs: options.inputs ?? {},
        outputs: undefined,
        attributes: { ...(options.attributes ?? {}) },
        status: null,
        ended: false,
        traceId: 'mock-trace-id',
        setAttribute: jest.fn((key: string, value: unknown) => {
          span.attributes[key] = value;
        }),
        setInputs: jest.fn((inputs: Record<string, unknown>) => {
          span.inputs = inputs;
        }),
        setOutputs: jest.fn((outputs: unknown) => {
          span.outputs = outputs;
        }),
        setStatus: jest.fn((code: string, description?: string) => {
          span.status = { code, description };
        }),
        end: jest.fn(() => {
          span.ended = true;
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
    SpanStatusCode: {
      OK: 'OK',
      ERROR: 'ERROR',
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
        getTrace: jest.fn(() => ({ info: mockTraceInfo })),
      })),
    },
  };
});

// Import after mock
import { createTracedQuery } from '../src/tracedClaudeAgent';

// ============================================================================
// Helpers
// ============================================================================

function getSpans() {
  return Object.values(mockSpans);
}

function spansByType(type: string) {
  return getSpans().filter((s) => s.spanType === type);
}

function spansByName(name: string) {
  return getSpans().filter((s) => s.name === name);
}

function rootSpan() {
  return getSpans().find((s) => s.parentId === null);
}

function childrenOf(spanId: string) {
  return getSpans().filter((s) => s.parentId === spanId);
}

/**
 * Build a mock query function that yields a scripted sequence of SDKMessages
 * and records the options passed in (so tests can verify forwardSubagentText).
 */
function mockQuery(messages: any[]) {
  const optionsSeen: Record<string, unknown>[] = [];
  const fn = (params: { prompt: unknown; options?: Record<string, unknown> }) => {
    optionsSeen.push(params.options ?? {});
    const iter = (async function* () {
      for (const m of messages) {
        yield m;
      }
    })();
    // Return an object that satisfies Symbol.asyncIterator plus interrupt/close stubs.
    const result: any = {
      [Symbol.asyncIterator]: () => iter,
      next: () => iter.next(),
      return: (v?: any) => iter.return(v),
      throw: (e?: any) => iter.throw(e),
      interrupt: jest.fn(),
    };
    return result;
  };
  return { fn, optionsSeen };
}

function mockQueryThatThrows(messages: any[], error: Error) {
  const fn = () => {
    const iter = (async function* () {
      for (const m of messages) yield m;
      throw error;
    })();
    return {
      [Symbol.asyncIterator]: () => iter,
      next: () => iter.next(),
      return: (v?: any) => iter.return(v),
      throw: (e?: any) => iter.throw(e),
      interrupt: jest.fn(),
    } as any;
  };
  return fn;
}

async function drain(query: any) {
  const seen: any[] = [];
  for await (const msg of query) {
    seen.push(msg);
  }
  return seen;
}

// ============================================================================
// Tests
// ============================================================================

beforeEach(() => {
  resetMocks();
  jest.clearAllMocks();
});

describe('createTracedQuery', () => {
  it('produces an AGENT root span with LLM and TOOL children for a basic run', async () => {
    const { fn } = mockQuery([
      { type: 'system', subtype: 'init', session_id: 'sess-abc', model: 'claude-haiku-4-5' },
      {
        type: 'assistant',
        parent_tool_use_id: null,
        message: {
          role: 'assistant',
          model: 'claude-haiku-4-5',
          content: [{ type: 'text', text: "I'll list the files." }],
          usage: { input_tokens: 10, output_tokens: 20 },
        },
      },
      {
        type: 'assistant',
        parent_tool_use_id: null,
        message: {
          role: 'assistant',
          model: 'claude-haiku-4-5',
          content: [
            { type: 'tool_use', id: 'tool-1', name: 'Bash', input: { command: 'ls' } },
          ],
          usage: { input_tokens: 30, output_tokens: 5 },
        },
      },
      {
        type: 'user',
        parent_tool_use_id: null,
        message: {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'tool-1', content: 'file1\nfile2', is_error: false },
          ],
        },
      },
      {
        type: 'assistant',
        parent_tool_use_id: null,
        message: {
          role: 'assistant',
          model: 'claude-haiku-4-5',
          content: [{ type: 'text', text: 'Two files: file1, file2.' }],
          usage: { input_tokens: 40, output_tokens: 10 },
        },
      },
      {
        type: 'result',
        subtype: 'success',
        duration_ms: 1234,
        usage: {
          input_tokens: 80,
          output_tokens: 35,
          cache_read_input_tokens: 100,
          cache_creation_input_tokens: 5,
        },
        result: 'Two files: file1, file2.',
      },
    ]);

    const traced = createTracedQuery(fn);
    await drain(traced({ prompt: 'list files' }));

    const root = rootSpan();
    expect(root).toBeDefined();
    expect(root!.spanType).toBe('AGENT');
    expect(root!.name).toBe('claude_code_conversation');
    expect(root!.ended).toBe(true);

    const llmSpans = spansByType('LLM');
    expect(llmSpans).toHaveLength(2);
    expect(llmSpans.every((s) => s.ended)).toBe(true);
    expect(llmSpans.every((s) => s.parentId === root!.spanId)).toBe(true);

    const toolSpans = spansByType('TOOL');
    expect(toolSpans).toHaveLength(1);
    expect(toolSpans[0].name).toBe('tool_Bash');
    expect(toolSpans[0].ended).toBe(true);
    expect(toolSpans[0].outputs).toEqual({ result: 'file1\nfile2' });

    // Root usage records cache tokens as separate keys (not collapsed).
    expect(root!.attributes['mlflow.chat.tokenUsage']).toEqual({
      input_tokens: 80,
      output_tokens: 35,
      total_tokens: 115,
      cache_read_input_tokens: 100,
      cache_creation_input_tokens: 5,
    });

    // Per-turn usage on LLM spans
    expect(llmSpans[0].attributes['mlflow.chat.tokenUsage']).toEqual({
      input_tokens: 10,
      output_tokens: 20,
      total_tokens: 30,
    });
  });

  it('creates AGENT(subagent) nested under TOOL(Agent) for Task tool calls', async () => {
    const { fn } = mockQuery([
      { type: 'system', subtype: 'init', session_id: 's', model: 'claude-haiku-4-5' },
      {
        type: 'assistant',
        parent_tool_use_id: null,
        message: {
          role: 'assistant',
          model: 'claude-haiku-4-5',
          content: [
            {
              type: 'tool_use',
              id: 'task-1',
              name: 'Agent',
              input: {
                subagent_type: 'general-purpose',
                description: 'List files',
                prompt: 'List files in /tmp',
              },
            },
          ],
        },
      },
      // Sub-agent inner messages
      {
        type: 'user',
        parent_tool_use_id: 'task-1',
        message: { role: 'user', content: [{ type: 'text', text: 'List files in /tmp' }] },
      },
      {
        type: 'assistant',
        parent_tool_use_id: 'task-1',
        message: {
          role: 'assistant',
          model: 'claude-haiku-4-5',
          content: [
            { type: 'tool_use', id: 'bash-1', name: 'Bash', input: { command: 'ls /tmp' } },
          ],
        },
      },
      {
        type: 'user',
        parent_tool_use_id: 'task-1',
        message: {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'bash-1', content: 'a\nb', is_error: false },
          ],
        },
      },
      {
        type: 'assistant',
        parent_tool_use_id: 'task-1',
        message: {
          role: 'assistant',
          model: 'claude-haiku-4-5',
          content: [{ type: 'text', text: 'Found 2 files.' }],
        },
      },
      // Task tool returns
      {
        type: 'user',
        parent_tool_use_id: null,
        message: {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'task-1', content: 'Found 2 files.', is_error: false },
          ],
        },
      },
      { type: 'result', subtype: 'success', duration_ms: 100, usage: { input_tokens: 0, output_tokens: 0 } },
    ]);

    await drain(createTracedQuery(fn)({ prompt: 'use agent' }));

    const root = rootSpan()!;
    const taskTool = spansByName('tool_Agent')[0];
    expect(taskTool).toBeDefined();
    expect(taskTool.parentId).toBe(root.spanId);

    const subagent = spansByName('subagent_general-purpose')[0];
    expect(subagent).toBeDefined();
    expect(subagent.spanType).toBe('AGENT');
    expect(subagent.parentId).toBe(taskTool.spanId);

    // Inner Bash tool span is nested under the subagent
    const innerBash = spansByName('tool_Bash')[0];
    expect(innerBash).toBeDefined();
    expect(innerBash.parentId).toBe(subagent.spanId);
    expect(innerBash.outputs).toEqual({ result: 'a\nb' });
    expect(innerBash.ended).toBe(true);

    // Inner LLM span(s) are also under the subagent
    const innerLlms = childrenOf(subagent.spanId).filter((s) => s.spanType === 'LLM');
    expect(innerLlms.length).toBeGreaterThanOrEqual(1);

    // Both subagent wrapper and Task tool are closed after the tool_result
    expect(subagent.ended).toBe(true);
    expect(taskTool.ended).toBe(true);
  });

  it('marks TOOL span as ERROR when tool_result has is_error=true', async () => {
    const { fn } = mockQuery([
      { type: 'system', subtype: 'init', session_id: 's', model: 'm' },
      {
        type: 'assistant',
        parent_tool_use_id: null,
        message: {
          role: 'assistant',
          model: 'm',
          content: [{ type: 'tool_use', id: 'tool-x', name: 'Bash', input: {} }],
        },
      },
      {
        type: 'user',
        parent_tool_use_id: null,
        message: {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'tool-x', content: 'command failed', is_error: true },
          ],
        },
      },
      { type: 'result', subtype: 'success', duration_ms: 1, usage: { input_tokens: 0, output_tokens: 0 } },
    ]);

    await drain(createTracedQuery(fn)({ prompt: 'p' }));

    const tool = spansByName('tool_Bash')[0];
    expect(tool.status?.code).toBe('ERROR');
    expect(tool.status?.description).toBe('command failed');
    expect(tool.ended).toBe(true);
  });

  it('closes all open spans with ERROR when the stream throws', async () => {
    const fn = mockQueryThatThrows(
      [
        { type: 'system', subtype: 'init', session_id: 's', model: 'm' },
        {
          type: 'assistant',
          parent_tool_use_id: null,
          message: {
            role: 'assistant',
            model: 'm',
            content: [{ type: 'tool_use', id: 'tool-x', name: 'Bash', input: {} }],
          },
        },
      ],
      new Error('rate limit'),
    );

    const traced = createTracedQuery(fn);
    await expect(drain(traced({ prompt: 'p' }))).rejects.toThrow('rate limit');

    const root = rootSpan()!;
    expect(root.status?.code).toBe('ERROR');
    expect(root.ended).toBe(true);

    const tool = spansByName('tool_Bash')[0];
    expect(tool.status?.code).toBe('ERROR');
    expect(tool.ended).toBe(true);
  });

  it('marks root as ERROR when result.subtype is not success', async () => {
    const { fn } = mockQuery([
      { type: 'system', subtype: 'init', session_id: 's', model: 'm' },
      {
        type: 'result',
        subtype: 'error_max_turns',
        duration_ms: 1,
        usage: { input_tokens: 1, output_tokens: 1 },
      },
    ]);

    await drain(createTracedQuery(fn)({ prompt: 'p' }));

    const root = rootSpan()!;
    expect(root.status?.code).toBe('ERROR');
    expect(root.status?.description).toBe('error_max_turns');
  });

  it('sets forwardSubagentText: true by default and preserves user-provided options', async () => {
    const { fn, optionsSeen } = mockQuery([
      { type: 'result', subtype: 'success', duration_ms: 1, usage: { input_tokens: 0, output_tokens: 0 } },
    ]);

    await drain(createTracedQuery(fn)({ prompt: 'p', options: { permissionMode: 'bypassPermissions' } }));

    expect(optionsSeen[0]).toMatchObject({
      forwardSubagentText: true,
      permissionMode: 'bypassPermissions',
    });
  });

  it('forces forwardSubagentText: true even when the caller passes false', async () => {
    const { fn, optionsSeen } = mockQuery([
      { type: 'result', subtype: 'success', duration_ms: 1, usage: { input_tokens: 0, output_tokens: 0 } },
    ]);

    await drain(createTracedQuery(fn)({ prompt: 'p', options: { forwardSubagentText: false } }));

    expect(optionsSeen[0]?.forwardSubagentText).toBe(true);
  });

  it('records hooks as a sanitized placeholder so functions are not serialised on the span', async () => {
    const { fn } = mockQuery([
      { type: 'result', subtype: 'success', duration_ms: 1, usage: { input_tokens: 0, output_tokens: 0 } },
    ]);
    const hookFn = jest.fn();

    await drain(createTracedQuery(fn)({ prompt: 'p', options: { hooks: { PreToolUse: [hookFn] } } }));

    const root = rootSpan()!;
    // Hooks should not appear in the recorded options
    expect((root.inputs.options as any).hooks).toBeUndefined();
  });
});
