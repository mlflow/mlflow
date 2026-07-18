/**
 * Integration tests for createTracedQuery against a real MLflow tracking server.
 *
 * Mirrors core/tests/core/api.test.ts: the global-server-setup spawns a
 * local MLflow server, we create a real experiment, drive the wrapper with
 * scripted SDKMessage sequences, flush, then fetch the trace via the client
 * API and assert on the real span tree.
 *
 * No mocks of @mlflow/core, LiveSpan, or InMemoryTraceManager — the same
 * exporter, span processor, and storage path that production uses.
 */

import * as mlflow from '@mlflow/core';
import { createAuthProvider } from '@mlflow/core/auth';
import { MlflowClient } from '@mlflow/core/clients';
import { SpanType } from '@mlflow/core/core/constants';
import { SpanStatusCode } from '@mlflow/core/core/entities/span_status';
import { TraceState } from '@mlflow/core/core/entities/trace_state';

import { createTracedQuery } from '../src/tracedClaudeAgent';

// The real `Query` type has 20+ methods (setModel, setPermissionMode, etc.)
// that the wrapper doesn't touch; cast the minimal stub for tests.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyQueryFn = any;

const TEST_TRACKING_URI = process.env.MLFLOW_TRACKING_URI ?? 'http://localhost:5000';

// ============================================================================
// Mock query (the only thing we mock — the SDK transport itself)
// ============================================================================

interface MockQueryParams {
  prompt: string | AsyncIterable<unknown>;
  options?: Record<string, unknown>;
}

interface MockQueryReturn {
  [Symbol.asyncIterator]: () => AsyncIterator<unknown>;
  next: () => Promise<IteratorResult<unknown>>;
  return: (v?: unknown) => Promise<IteratorResult<unknown>>;
  throw: (e?: unknown) => Promise<IteratorResult<unknown>>;
  interrupt: jest.Mock;
}

function mockQuery(messages: unknown[]) {
  const optionsSeen: Record<string, unknown>[] = [];
  const fn = (params: MockQueryParams): MockQueryReturn => {
    optionsSeen.push(params.options ?? {});
    const iter = (async function* () {
      // If the prompt is an AsyncIterable, drain it (mirrors real SDK).
      if (typeof params.prompt !== 'string') {
        for await (const _item of params.prompt) {
          // Discard — the wrapper observes via its capturing iterable.
        }
      }
      for (const m of messages) {
        yield m;
      }
    })();
    return {
      [Symbol.asyncIterator]: () => iter,
      next: () => iter.next(),
      return: (v?: unknown) => iter.return(v as never),
      throw: (e?: unknown) => iter.throw(e),
      interrupt: jest.fn(),
    };
  };
  return { fn, optionsSeen };
}

function mockQueryThatThrows(messages: unknown[], error: Error) {
  const fn = (): MockQueryReturn => {
    // eslint-disable-next-line @typescript-eslint/require-await
    const iter = (async function* () {
      for (const m of messages) {
        yield m;
      }
      throw error;
    })();
    return {
      [Symbol.asyncIterator]: () => iter,
      next: () => iter.next(),
      return: (v?: unknown) => iter.return(v as never),
      throw: (e?: unknown) => iter.throw(e),
      interrupt: jest.fn(),
    };
  };
  return fn;
}

async function drain(query: unknown) {
  for await (const _msg of query as AsyncIterable<unknown>) {
    // discard
  }
}

// ============================================================================
// Trace fetch helpers — operate on the real trace returned by the server
// ============================================================================

interface FetchedSpan {
  name: string;
  spanId: string;
  parentId: string | null;
  spanType: string;
  inputs: unknown;
  outputs: unknown;
  attributes: Record<string, unknown>;
  status: { statusCode: string; description?: string };
}

function rootSpan(spans: FetchedSpan[]): FetchedSpan {
  const root = spans.find((s) => s.parentId == null);
  if (!root) {
    throw new Error('No root span in trace');
  }
  return root;
}

function spansByName(spans: FetchedSpan[], name: string): FetchedSpan[] {
  return spans.filter((s) => s.name === name);
}

function spansByType(spans: FetchedSpan[], type: string): FetchedSpan[] {
  return spans.filter((s) => s.spanType === type);
}

function childrenOf(spans: FetchedSpan[], spanId: string): FetchedSpan[] {
  return spans.filter((s) => s.parentId === spanId);
}

// ============================================================================
// Suite setup — one experiment shared across tests
// ============================================================================

describe('createTracedQuery (integration)', () => {
  let client: MlflowClient;
  let experimentId: string;

  beforeAll(async () => {
    const authProvider = createAuthProvider({ trackingUri: TEST_TRACKING_URI });
    client = new MlflowClient({ trackingUri: TEST_TRACKING_URI, authProvider });
    const experimentName = `claude-code-agent-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    experimentId = await client.createExperiment(experimentName);
    mlflow.init({ trackingUri: TEST_TRACKING_URI, experimentId });
  });

  afterAll(async () => {
    if (experimentId) {
      await client.deleteExperiment(experimentId);
    }
  });

  async function runAndFetchTrace(
    wrappedQuery: ReturnType<typeof createTracedQuery>,
    params: MockQueryParams,
  ) {
    await drain(wrappedQuery(params));
    await mlflow.flushTraces();
    const traceId = mlflow.getLastActiveTraceId();
    if (!traceId) {
      throw new Error('No active trace id after run');
    }
    const trace = await client.getTrace(traceId);
    return {
      trace,
      spans: trace.data.spans as unknown as FetchedSpan[],
    };
  }

  // --------------------------------------------------------------------------
  // Basic span tree
  // --------------------------------------------------------------------------

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
          content: [{ type: 'tool_use', id: 'tool-1', name: 'Bash', input: { command: 'ls' } }],
          usage: { input_tokens: 30, output_tokens: 5 },
        },
      },
      {
        type: 'user',
        parent_tool_use_id: null,
        message: {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'tool-1',
              content: 'file1\nfile2',
              is_error: false,
            },
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

    const { trace, spans } = await runAndFetchTrace(createTracedQuery(fn as AnyQueryFn), {
      prompt: 'list files',
    });

    expect(trace.info.state).toBe(TraceState.OK);

    const root = rootSpan(spans);
    expect(root.spanType).toBe(SpanType.AGENT);
    expect(root.name).toBe('claude_code_conversation');

    const llmSpans = spansByType(spans, SpanType.LLM);
    expect(llmSpans).toHaveLength(2);
    expect(llmSpans.every((s) => s.parentId === root.spanId)).toBe(true);

    const toolSpans = spansByType(spans, SpanType.TOOL);
    expect(toolSpans).toHaveLength(1);
    expect(toolSpans[0].name).toBe('tool_Bash');
    expect(toolSpans[0].outputs).toEqual({ result: 'file1\nfile2' });

    // Aggregate token usage (with separate cache keys) is recorded on the root.
    const rootUsage = root.attributes['mlflow.chat.tokenUsage'] as Record<string, number>;
    expect(rootUsage).toEqual({
      input_tokens: 80,
      output_tokens: 35,
      total_tokens: 115,
      cache_read_input_tokens: 100,
      cache_creation_input_tokens: 5,
    });

    // Per-turn token usage on the first LLM span.
    expect(llmSpans[0].attributes['mlflow.chat.tokenUsage']).toEqual({
      input_tokens: 10,
      output_tokens: 20,
      total_tokens: 30,
    });
  });

  // --------------------------------------------------------------------------
  // Sub-agent nesting
  // --------------------------------------------------------------------------

  it('nests an AGENT(subagent) span under the Task tool for sub-agent invocations', async () => {
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
      {
        type: 'user',
        parent_tool_use_id: null,
        message: {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'task-1',
              content: 'Found 2 files.',
              is_error: false,
            },
          ],
        },
      },
      {
        type: 'result',
        subtype: 'success',
        duration_ms: 100,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    ]);

    const { spans } = await runAndFetchTrace(createTracedQuery(fn as AnyQueryFn), {
      prompt: 'use agent',
    });

    const root = rootSpan(spans);
    const taskTool = spansByName(spans, 'tool_Agent')[0];
    expect(taskTool).toBeDefined();
    expect(taskTool.parentId).toBe(root.spanId);

    const subagent = spansByName(spans, 'subagent_general-purpose')[0];
    expect(subagent).toBeDefined();
    expect(subagent.spanType).toBe(SpanType.AGENT);
    expect(subagent.parentId).toBe(taskTool.spanId);

    // Inner Bash span nests under the sub-agent wrapper, not under root.
    const innerBash = spansByName(spans, 'tool_Bash')[0];
    expect(innerBash.parentId).toBe(subagent.spanId);
    expect(innerBash.outputs).toEqual({ result: 'a\nb' });

    // At least one inner LLM span lives under the sub-agent.
    const innerLlms = childrenOf(spans, subagent.spanId).filter((s) => s.spanType === SpanType.LLM);
    expect(innerLlms.length).toBeGreaterThanOrEqual(1);
  });

  // --------------------------------------------------------------------------
  // Error handling
  // --------------------------------------------------------------------------

  it('marks a TOOL span ERROR when its tool_result has is_error=true', async () => {
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
            {
              type: 'tool_result',
              tool_use_id: 'tool-x',
              content: 'command failed',
              is_error: true,
            },
          ],
        },
      },
      {
        type: 'result',
        subtype: 'success',
        duration_ms: 1,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    ]);

    const { spans } = await runAndFetchTrace(createTracedQuery(fn as AnyQueryFn), { prompt: 'p' });

    const tool = spansByName(spans, 'tool_Bash')[0];
    expect(tool.status.statusCode).toBe(SpanStatusCode.ERROR);
    expect(tool.status.description).toBe('command failed');
  });

  it('closes every open span with ERROR and marks the trace ERROR on stream throw', async () => {
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

    const traced = createTracedQuery(fn as AnyQueryFn);
    await expect(drain(traced({ prompt: 'p' }))).rejects.toThrow('rate limit');
    await mlflow.flushTraces();
    const traceId = mlflow.getLastActiveTraceId();
    if (!traceId) {
      throw new Error('No trace id');
    }
    const trace = await client.getTrace(traceId);
    const spans = trace.data.spans as unknown as FetchedSpan[];

    expect(trace.info.state).toBe(TraceState.ERROR);
    const tool = spansByName(spans, 'tool_Bash')[0];
    expect(tool.status.statusCode).toBe(SpanStatusCode.ERROR);
  });

  it('marks the trace ERROR when result.subtype is not "success"', async () => {
    const { fn } = mockQuery([
      { type: 'system', subtype: 'init', session_id: 's', model: 'm' },
      {
        type: 'result',
        subtype: 'error_max_turns',
        duration_ms: 1,
        usage: { input_tokens: 1, output_tokens: 1 },
      },
    ]);

    const { trace } = await runAndFetchTrace(createTracedQuery(fn as AnyQueryFn), { prompt: 'p' });
    expect(trace.info.state).toBe(TraceState.ERROR);
  });

  // --------------------------------------------------------------------------
  // Wrapper behavior
  // --------------------------------------------------------------------------

  it('forces forwardSubagentText: true even when the caller passes false', async () => {
    const { fn, optionsSeen } = mockQuery([
      {
        type: 'result',
        subtype: 'success',
        duration_ms: 1,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    ]);

    await drain(
      createTracedQuery(fn as AnyQueryFn)({ prompt: 'p', options: { forwardSubagentText: false } }),
    );
    await mlflow.flushTraces();

    expect(optionsSeen[0]?.forwardSubagentText).toBe(true);
  });

  it('replaces callbacks in options with a sanitized placeholder on the span', async () => {
    const { fn } = mockQuery([
      {
        type: 'result',
        subtype: 'success',
        duration_ms: 1,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    ]);
    const hookFn = jest.fn();
    const canUseTool = jest.fn();

    const { spans } = await runAndFetchTrace(createTracedQuery(fn as AnyQueryFn), {
      prompt: 'p',
      options: {
        hooks: { PreToolUse: [hookFn] },
        canUseTool,
        permissionMode: 'bypassPermissions',
      },
    });

    const opts = (rootSpan(spans).inputs as { options: Record<string, unknown> }).options;
    expect((opts.hooks as { PreToolUse: unknown[] }).PreToolUse[0]).toBe('[function]');
    expect(opts.canUseTool).toBe('[function]');
    expect(opts.permissionMode).toBe('bypassPermissions');
  });

  it('finalizes the trace even when the consumer breaks out of the iterator early', async () => {
    const { fn } = mockQuery([
      { type: 'system', subtype: 'init', session_id: 's', model: 'm' },
      {
        type: 'assistant',
        parent_tool_use_id: null,
        message: { role: 'assistant', model: 'm', content: [{ type: 'text', text: 'hi' }] },
      },
      {
        type: 'assistant',
        parent_tool_use_id: null,
        message: {
          role: 'assistant',
          model: 'm',
          content: [{ type: 'text', text: 'never reached' }],
        },
      },
    ]);

    const stream = createTracedQuery(fn as AnyQueryFn)({ prompt: 'p' });
    for await (const msg of stream as AsyncIterable<{ type: string }>) {
      if (msg.type === 'assistant') {
        break;
      }
    }
    await mlflow.flushTraces();

    const traceId = mlflow.getLastActiveTraceId();
    if (!traceId) {
      throw new Error('No trace id after early break');
    }
    // If the post-loop finalize hadn't run, the root would never end and the
    // exporter wouldn't have pushed the trace — getTrace would 404.
    const trace = await client.getTrace(traceId);
    expect(trace.info.traceId).toBe(traceId);
    expect(trace.info.state).toBe(TraceState.OK);
  });

  it('captures streaming-prompt text on the root span as the SDK consumes it', async () => {
    const { fn } = mockQuery([
      {
        type: 'result',
        subtype: 'success',
        duration_ms: 1,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    ]);

    // eslint-disable-next-line @typescript-eslint/require-await
    async function* streamingPrompt() {
      yield { type: 'user', message: { role: 'user', content: 'first chunk' } };
      yield {
        type: 'user',
        message: { role: 'user', content: [{ type: 'text', text: 'second chunk' }] },
      };
    }

    const { spans } = await runAndFetchTrace(createTracedQuery(fn as AnyQueryFn), {
      prompt: streamingPrompt(),
    });

    const recorded = (rootSpan(spans).inputs as { prompt: string }).prompt;
    expect(recorded).toContain('first chunk');
    expect(recorded).toContain('second chunk');
  });
});
