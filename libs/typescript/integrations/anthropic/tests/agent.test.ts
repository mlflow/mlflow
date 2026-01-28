/**
 * Tests for MLflow Claude Agent SDK integration
 * Tests createTracedQuery and extractTokenUsage from tracedClaudeAgent.ts
 */

import * as mlflow from 'mlflow-tracing';
import { createTracedQuery } from '../src';
import { extractTokenUsage } from '../src/tracedClaudeAgent';
import { createAuthProvider } from 'mlflow-tracing/src/auth';

const TEST_TRACKING_URI = 'http://localhost:5000';

// Mock SDK message types
interface MockSDKMessage {
  type: string;
  subtype?: string;
  session_id?: string;
  model?: string;
  message?: { content?: Array<{ type: string; text?: string; name?: string; id?: string }> };
  result?: string;
  errors?: string[];
  total_cost_usd?: number;
  num_turns?: number;
  duration_ms?: number;
  usage?: { input_tokens?: number; output_tokens?: number };
  modelUsage?: Record<string, { inputTokens?: number; outputTokens?: number }>;
}

// Type for query function params matching the actual SDK
type MockQueryParams = {
  prompt: string | AsyncIterable<unknown>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  options?: Record<string, any>;
};

// Mock query function that simulates Claude Agent SDK behavior
function createMockQueryFunction(messages: MockSDKMessage[]) {
  return function mockQuery(params: MockQueryParams) {
    const hooks = params.options?.hooks as
      | Record<
          string,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          Array<{ hooks: Array<(input: any) => Promise<{ continue: boolean }>> }>
        >
      | undefined;
    async function* generator(): AsyncGenerator<MockSDKMessage, void> {
      for (const msg of messages) {
        // Call hooks if they exist
        if (msg.type === 'assistant' && msg.message?.content) {
          for (const block of msg.message.content) {
            if (block.type === 'tool_use' && hooks?.PreToolUse) {
              for (const matcher of hooks.PreToolUse) {
                for (const hook of matcher.hooks) {
                  await hook({
                    tool_name: block.name,
                    tool_input: { test: 'input' },
                    tool_use_id: block.id,
                  });
                }
              }
            }
          }
        }
        yield msg;
        // Simulate tool response after tool use
        if (msg.type === 'assistant' && msg.message?.content) {
          for (const block of msg.message.content) {
            if (block.type === 'tool_use' && hooks?.PostToolUse) {
              for (const matcher of hooks.PostToolUse) {
                for (const hook of matcher.hooks) {
                  await hook({
                    tool_name: block.name,
                    tool_input: { test: 'input' },
                    tool_use_id: block.id,
                    tool_response: { stdout: 'tool output' },
                  });
                }
              }
            }
          }
        }
      }
    }

    const gen = generator();
    return Object.assign(gen, {
      interrupt: (): Promise<void> => Promise.resolve(),
      close: (): void => {},
    });
  };
}

describe('createTracedQuery', () => {
  let experimentId: string;
  let client: mlflow.MlflowClient;

  beforeAll(async () => {
    const authProvider = createAuthProvider({ trackingUri: TEST_TRACKING_URI });
    client = new mlflow.MlflowClient({ trackingUri: TEST_TRACKING_URI, authProvider });

    const experimentName = `claude-agent-test-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    experimentId = await client.createExperiment(experimentName);

    mlflow.init({
      trackingUri: TEST_TRACKING_URI,
      experimentId,
    });
  });

  afterAll(async () => {
    await client.deleteExperiment(experimentId);
  });

  const getLastActiveTrace = async (): Promise<mlflow.Trace> => {
    await mlflow.flushTraces();
    const traceId = mlflow.getLastActiveTraceId();
    const trace = await client.getTrace(traceId!);
    return trace;
  };

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Basic query tracing', () => {
    it('should trace a simple query as AGENT span', async () => {
      const mockMessages: MockSDKMessage[] = [
        { type: 'system', subtype: 'init', session_id: 'sess-123', model: 'claude-3-5-sonnet' },
        {
          type: 'assistant',
          message: { content: [{ type: 'text', text: 'Hello! The answer is 4.' }] },
        },
        {
          type: 'result',
          subtype: 'success',
          result: 'Query completed',
          usage: { input_tokens: 50, output_tokens: 100 },
          num_turns: 1,
        },
      ];

      const mockQueryFn = createMockQueryFunction(mockMessages);
      const tracedQuery = createTracedQuery(mockQueryFn);

      const query = tracedQuery({ prompt: 'What is 2 + 2?' });

      const results: MockSDKMessage[] = [];
      for await (const msg of query) {
        results.push(msg);
      }

      expect(results).toHaveLength(3);
      expect(results[2].type).toBe('result');

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');

      const span = trace.data.spans[0];
      expect(span.name).toBe('claude_agent_query');
      expect(span.spanType).toBe(mlflow.SpanType.AGENT);
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(span.inputs).toEqual({
        model: 'claude-3-5-sonnet',
        messages: [
          { role: 'user', content: 'What is 2 + 2?' },
          { role: 'assistant', content: [{ type: 'text', text: 'Hello! The answer is 4.' }] },
        ],
      });

      // Check token usage
      const tokenUsage = span.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE];
      expect(tokenUsage).toBeDefined();
      expect(tokenUsage[mlflow.TokenUsageKey.INPUT_TOKENS]).toBe(50);
      expect(tokenUsage[mlflow.TokenUsageKey.OUTPUT_TOKENS]).toBe(100);
      expect(tokenUsage[mlflow.TokenUsageKey.TOTAL_TOKENS]).toBe(150);

      // Check message format
      const messageFormat = span.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
      expect(messageFormat).toBe('anthropic');
    });

    it('should trace query with tool use as nested TOOL spans', async () => {
      const mockMessages: MockSDKMessage[] = [
        { type: 'system', subtype: 'init', session_id: 'sess-456', model: 'claude-3-5-sonnet' },
        {
          type: 'assistant',
          message: {
            content: [{ type: 'tool_use', name: 'Bash', id: 'tool-1' }],
          },
        },
        {
          type: 'result',
          subtype: 'success',
          result: 'echo hello completed',
          usage: { input_tokens: 100, output_tokens: 200 },
          num_turns: 2,
        },
      ];

      const mockQueryFn = createMockQueryFunction(mockMessages);
      const tracedQuery = createTracedQuery(mockQueryFn);

      const query = tracedQuery({
        prompt: 'Run echo hello',
        options: { tools: ['Bash'] },
      });

      for await (const _msg of query) {
        // consume messages
      }

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(2);

      // Parent AGENT span
      const agentSpan = trace.data.spans[0];
      expect(agentSpan.name).toBe('claude_agent_query');
      expect(agentSpan.spanType).toBe(mlflow.SpanType.AGENT);

      // Child TOOL span
      const toolSpan = trace.data.spans[1];
      expect(toolSpan.name).toBe('tool_Bash');
      expect(toolSpan.spanType).toBe(mlflow.SpanType.TOOL);
      expect(toolSpan.attributes['tool.name']).toBe('Bash');
    });
  });

  describe('Error handling', () => {
    it('should handle query errors and mark span as ERROR', async () => {
      const mockQueryFn = (_params: MockQueryParams) => {
        async function* generator(): AsyncGenerator<MockSDKMessage, void> {
          await Promise.resolve();
          yield {
            type: 'system',
            subtype: 'init',
            session_id: 'sess-err',
            model: 'claude-3-5-sonnet',
          };
          throw new Error('API rate limit exceeded');
        }

        const gen = generator();
        return Object.assign(gen, {
          interrupt: (): Promise<void> => Promise.resolve(),
          close: (): void => {},
        });
      };

      const tracedQuery = createTracedQuery(mockQueryFn);
      const query = tracedQuery({ prompt: 'This should fail' });

      await expect(async () => {
        for await (const _msg of query) {
          // consume
        }
      }).rejects.toThrow('API rate limit exceeded');

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('ERROR');

      const span = trace.data.spans[0];
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
      expect(span.status.description).toContain('API rate limit exceeded');
    });

    it('should handle result with errors subtype', async () => {
      const mockMessages: MockSDKMessage[] = [
        { type: 'system', subtype: 'init', session_id: 'sess-789', model: 'claude-3-5-sonnet' },
        {
          type: 'result',
          subtype: 'error',
          errors: ['Tool execution failed'],
          usage: { input_tokens: 10, output_tokens: 20 },
        },
      ];

      const mockQueryFn = createMockQueryFunction(mockMessages);
      const tracedQuery = createTracedQuery(mockQueryFn);

      const query = tracedQuery({ prompt: 'This will have errors' });

      for await (const _msg of query) {
        // consume
      }

      const trace = await getLastActiveTrace();
      // The span should still complete but with error status
      const span = trace.data.spans[0];
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
    });
  });

  describe('Token usage with modelUsage', () => {
    it('should aggregate token usage from modelUsage field', async () => {
      const mockMessages: MockSDKMessage[] = [
        { type: 'system', subtype: 'init', session_id: 'sess-model', model: 'claude-3-5-sonnet' },
        {
          type: 'result',
          subtype: 'success',
          modelUsage: {
            'claude-3-5-sonnet': { inputTokens: 100, outputTokens: 200 },
            'claude-3-haiku': { inputTokens: 50, outputTokens: 100 },
          },
        },
      ];

      const mockQueryFn = createMockQueryFunction(mockMessages);
      const tracedQuery = createTracedQuery(mockQueryFn);

      const query = tracedQuery({ prompt: 'Multi-model query' });

      for await (const _msg of query) {
        // consume
      }

      const trace = await getLastActiveTrace();
      const span = trace.data.spans[0];

      const tokenUsage = span.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE];
      expect(tokenUsage).toBeDefined();
      expect(tokenUsage[mlflow.TokenUsageKey.INPUT_TOKENS]).toBe(150);
      expect(tokenUsage[mlflow.TokenUsageKey.OUTPUT_TOKENS]).toBe(300);
      expect(tokenUsage[mlflow.TokenUsageKey.TOTAL_TOKENS]).toBe(450);
    });
  });
});

describe('extractTokenUsage', () => {
  it('should extract token usage from direct usage field', () => {
    const response = {
      usage: {
        input_tokens: 100,
        output_tokens: 200,
      },
    };

    const usage = extractTokenUsage(response);

    expect(usage).toEqual({
      input_tokens: 100,
      output_tokens: 200,
      total_tokens: 300,
    });
  });

  it('should extract token usage from modelUsage field', () => {
    const response = {
      modelUsage: {
        'claude-3-5-sonnet': { inputTokens: 50, outputTokens: 75 },
        'claude-3-haiku': { inputTokens: 25, outputTokens: 25 },
      },
    };

    const usage = extractTokenUsage(response);

    expect(usage).toEqual({
      input_tokens: 75,
      output_tokens: 100,
      total_tokens: 175,
    });
  });

  it('should prefer usage field over modelUsage', () => {
    const response = {
      usage: {
        input_tokens: 100,
        output_tokens: 200,
      },
      modelUsage: {
        'claude-3-5-sonnet': { inputTokens: 50, outputTokens: 75 },
      },
    };

    const usage = extractTokenUsage(response);

    expect(usage).toEqual({
      input_tokens: 100,
      output_tokens: 200,
      total_tokens: 300,
    });
  });

  it('should return undefined for missing usage', () => {
    const response = { content: 'Hello' };

    const usage = extractTokenUsage(response);

    expect(usage).toBeUndefined();
  });

  it('should return undefined for null response', () => {
    const usage = extractTokenUsage(null);

    expect(usage).toBeUndefined();
  });

  it('should handle partial usage data', () => {
    const response = {
      usage: {
        input_tokens: 100,
        // output_tokens is missing
      },
    };

    const usage = extractTokenUsage(response);

    expect(usage).toEqual({
      input_tokens: 100,
      output_tokens: 0,
      total_tokens: 100,
    });
  });
});
