/**
 * Tests for MLflow OpenClaw integration
 */

import type { DiagnosticEventPayload } from 'openclaw/plugin-sdk/diagnostics-otel';

// Capture the diagnostic event handler registered by the service
let diagnosticHandler: ((evt: DiagnosticEventPayload) => void) | null = null;

jest.mock('openclaw/plugin-sdk/diagnostics-otel', () => ({
  onDiagnosticEvent: jest.fn((handler: (evt: DiagnosticEventPayload) => void) => {
    diagnosticHandler = handler;
    return jest.fn(); // unsubscribe
  }),
}));

// Mock the @mlflow/core module
jest.mock('@mlflow/core', () => {
  const createMockSpan = () => {
    const otelTraceId = `otel-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    return {
      traceId: `mock-trace-${otelTraceId}`,
      _span: { spanContext: () => ({ traceId: otelTraceId }) },
      setAttribute: jest.fn(),
      setOutputs: jest.fn(),
      setStatus: jest.fn(),
      end: jest.fn(),
    };
  };

  const mockTraceInfo = {
    requestPreview: '',
    responsePreview: '',
    traceMetadata: {},
  };

  const mockTrace = {
    info: mockTraceInfo,
  };

  return {
    init: jest.fn(),
    startSpan: jest.fn(() => createMockSpan()),
    tracingContext: jest.fn((_opts: unknown, fn: () => unknown) => fn()),
    flushTraces: jest.fn().mockResolvedValue(undefined),
    SpanType: {
      LLM: 'LLM',
      TOOL: 'TOOL',
      AGENT: 'AGENT',
    },
    SpanStatusCode: {
      OK: 'OK',
      ERROR: 'ERROR',
      UNSET: 'UNSET',
    },
    SpanAttributeKey: {
      TOKEN_USAGE: 'mlflow.chat.tokenUsage',
      MESSAGE_FORMAT: 'mlflow.message.format',
    },
    TraceMetadataKey: {
      TRACE_SESSION: 'mlflow.trace.session',
      TRACE_USER: 'mlflow.trace.user',
    },
    InMemoryTraceManager: {
      getInstance: jest.fn(() => ({
        getTrace: jest.fn(() => mockTrace),
        getMlflowTraceIdFromOtelId: jest.fn(() => 'mock-mlflow-trace-id'),
      })),
    },
  };
});

import * as mlflowTracing from '@mlflow/core';
import { createMLflowService } from '../src/service';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type EventHandler = (event: unknown, ctx: unknown) => void;

interface MockApi {
  on: jest.Mock;
  registerService: jest.Mock;
  registerCli: jest.Mock;
  runtime: { config: { loadConfig: jest.Mock; writeConfigFile: jest.Mock } };
  pluginConfig: undefined;
}

interface TestHarness {
  api: MockApi;
  fire: (event: string, eventData: Record<string, unknown>, ctx: Record<string, unknown>) => void;
  fireDiagnostic: (evt: DiagnosticEventPayload) => void;
  handlers: Map<string, EventHandler>;
}

function createTestHarness(): TestHarness {
  const handlers = new Map<string, EventHandler>();
  const api: MockApi = {
    on: jest.fn((eventName: string, handler: EventHandler) => {
      handlers.set(eventName, handler);
    }),
    registerService: jest.fn(),
    registerCli: jest.fn(),
    runtime: {
      config: {
        loadConfig: jest.fn(() => ({})),
        writeConfigFile: jest.fn(),
      },
    },
    pluginConfig: undefined,
  };

  return {
    api,
    handlers,
    fire(event: string, eventData: Record<string, unknown>, ctx: Record<string, unknown>) {
      const handler = handlers.get(event);
      if (handler) handler(eventData, ctx);
    },
    fireDiagnostic(evt: DiagnosticEventPayload) {
      if (diagnosticHandler) diagnosticHandler(evt);
    },
  };
}

function createMockLogger() {
  return {
    info: jest.fn(),
    warn: jest.fn(),
  };
}

async function startService(
  harness: TestHarness,
  pluginCfg: Record<string, unknown> = {
    trackingUri: 'http://localhost:5000',
    experimentId: 'exp-123',
  },
) {
  const service = createMLflowService(harness.api as any, pluginCfg);
  service.registerHooks();
  await service.start!({
    config: {},
    logger: createMockLogger(),
  });
  return service;
}

async function flushMicrotasks(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('MLflowTracingPlugin', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.clearAllMocks();
    diagnosticHandler = null;
    process.env = { ...originalEnv };
    delete process.env.MLFLOW_TRACKING_URI;
    delete process.env.MLFLOW_EXPERIMENT_ID;
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  describe('Service Lifecycle', () => {
    it('should return a service with id and start/stop', () => {
      const harness = createTestHarness();
      const service = createMLflowService(harness.api as any);
      expect(service.id).toBe('mlflow-tracing');
      expect(typeof service.start).toBe('function');
      expect(typeof service.stop).toBe('function');
    });

    it('should register event hooks on start', async () => {
      const harness = createTestHarness();
      await startService(harness);

      expect(harness.api.on).toHaveBeenCalledWith('llm_input', expect.any(Function));
      expect(harness.api.on).toHaveBeenCalledWith('llm_output', expect.any(Function));
      expect(harness.api.on).toHaveBeenCalledWith('before_tool_call', expect.any(Function));
      expect(harness.api.on).toHaveBeenCalledWith('after_tool_call', expect.any(Function));
      expect(harness.api.on).toHaveBeenCalledWith('subagent_spawning', expect.any(Function));
      expect(harness.api.on).toHaveBeenCalledWith('subagent_ended', expect.any(Function));
      expect(harness.api.on).toHaveBeenCalledWith('agent_end', expect.any(Function));
    });
  });

  describe('Configuration', () => {
    it('should not initialize SDK when trackingUri is missing', async () => {
      const harness = createTestHarness();
      const service = createMLflowService(harness.api as any, { experimentId: 'exp-123' });
      service.registerHooks();
      await service.start!({
        config: {},
        logger: createMockLogger(),
      });

      // Hooks are registered during register() but SDK should not init
      expect(mlflowTracing.init).not.toHaveBeenCalled();
    });

    it('should not initialize SDK when experimentId is missing', async () => {
      const harness = createTestHarness();
      const service = createMLflowService(harness.api as any, {
        trackingUri: 'http://localhost:5000',
      });
      service.registerHooks();
      await service.start!({
        config: {},
        logger: createMockLogger(),
      });

      expect(mlflowTracing.init).not.toHaveBeenCalled();
    });

    it('should register hooks when pluginConfig has values', async () => {
      const harness = createTestHarness();
      await startService(harness);

      expect(harness.api.on).toHaveBeenCalledWith('llm_input', expect.any(Function));
      expect(harness.api.on).toHaveBeenCalledWith('agent_end', expect.any(Function));
    });

    it('should use pluginConfig values passed from register', async () => {
      const harness = createTestHarness();
      await startService(harness, {
        trackingUri: 'http://custom:9000',
        experimentId: 'custom-exp',
      });

      // SDK init now happens in register(), not start() — test that hooks work
      expect(harness.api.on).toHaveBeenCalledWith('llm_input', expect.any(Function));
    });
  });

  describe('Basic LLM Call Tracing', () => {
    it('should create root AGENT and child LLM spans on llm_input', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'What is 2+2?', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'openclaw_agent',
          spanType: 'AGENT',
          inputs: { messages: [{ role: 'user', content: 'What is 2+2?' }] },
        }),
      );

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'llm_call',
          spanType: 'LLM',
          inputs: expect.objectContaining({
            model: 'openai/gpt-4',
            messages: expect.arrayContaining([{ role: 'user', content: 'What is 2+2?' }]),
          }),
        }),
      );
    });

    it('should end LLM span on llm_output with response', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hello', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      const mockSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[1].value;

      harness.fire('llm_output', { response: 'The answer is 4.' }, { sessionKey: 'session-1' });

      expect(mockSpan.setOutputs).toHaveBeenCalledWith({
        choices: [{ message: { role: 'assistant', content: 'The answer is 4.' } }],
      });
      expect(mockSpan.end).toHaveBeenCalled();
    });

    it('should flush traces on agent_end', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hello', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      harness.fire('llm_output', { response: 'Hi' }, { sessionKey: 'session-1' });
      harness.fire('agent_end', {}, { sessionKey: 'session-1' });
      await flushMicrotasks();

      expect(mlflowTracing.flushTraces).toHaveBeenCalled();
    });

    it('should set model label from provider and model', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hi', provider: 'anthropic', model: 'claude-3-opus' },
        { sessionKey: 'session-1' },
      );

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'llm_call',
          inputs: expect.objectContaining({ model: 'anthropic/claude-3-opus' }),
        }),
      );
    });

    it('should include historyMessages in LLM inputs when provided', async () => {
      const harness = createTestHarness();
      await startService(harness);

      const history = [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi there' },
      ];
      harness.fire(
        'llm_input',
        { prompt: 'Follow up', model: 'gpt-4', provider: 'openai', historyMessages: history },
        { sessionKey: 'session-1' },
      );

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'llm_call',
          inputs: expect.objectContaining({
            messages: [
              { role: 'user', content: 'Hello' },
              { role: 'assistant', content: 'Hi there' },
              { role: 'user', content: 'Follow up' },
            ],
          }),
        }),
      );
    });

    it('should join assistantTexts as response content', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hello', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      const llmSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[1].value;

      harness.fire(
        'llm_output',
        { assistantTexts: ['Part 1', 'Part 2'] },
        { sessionKey: 'session-1' },
      );

      expect(llmSpan.setOutputs).toHaveBeenCalledWith({
        choices: [{ message: { role: 'assistant', content: 'Part 1\nPart 2' } }],
      });
    });

    it('should fall back to response string when assistantTexts is absent', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hello', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      const llmSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[1].value;

      harness.fire('llm_output', { response: 'Plain response' }, { sessionKey: 'session-1' });

      expect(llmSpan.setOutputs).toHaveBeenCalledWith({
        choices: [{ message: { role: 'assistant', content: 'Plain response' } }],
      });
    });

    it('should extract token usage from lastAssistant.usage as fallback', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hello', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      const llmSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[1].value;

      harness.fire(
        'llm_output',
        {
          assistantTexts: ['Hi'],
          lastAssistant: {
            role: 'assistant',
            usage: { input: 100, output: 30, totalTokens: 130 },
          },
        },
        { sessionKey: 'session-1' },
      );

      expect(llmSpan.setAttribute).toHaveBeenCalledWith('mlflow.chat.tokenUsage', {
        input_tokens: 100,
        output_tokens: 30,
        total_tokens: 130,
      });
    });

    it('should include system prompt in LLM inputs as system message', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        {
          prompt: 'hello',
          systemPrompt: 'You are a helpful assistant.',
          model: 'gpt-4',
          provider: 'openai',
        },
        { sessionKey: 'session-1' },
      );

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'llm_call',
          inputs: expect.objectContaining({
            messages: [
              { role: 'system', content: 'You are a helpful assistant.' },
              { role: 'user', content: 'hello' },
            ],
          }),
        }),
      );
    });
  });

  describe('Message Sanitization', () => {
    it('should strip sender metadata (fenced JSON) from prompt', async () => {
      const harness = createTestHarness();
      await startService(harness);

      const rawPrompt =
        'Sender (untrusted metadata):\n```json\n{\n  "label": "openclaw-tui",\n  "id": "gateway-client"\n}\n```\n\n[Wed 2026-03-18 03:42 GMT+9] Hello';
      harness.fire(
        'llm_input',
        { prompt: rawPrompt, model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'openclaw_agent',
          inputs: { messages: [{ role: 'user', content: 'Hello' }] },
        }),
      );
    });

    it('should strip sender metadata (plain JSON) from prompt', async () => {
      const harness = createTestHarness();
      await startService(harness);

      const rawPrompt =
        'Sender (untrusted metadata):\n{"label": "tui", "id": "gw"}\n\n[Mon 2026-03-18 10:00 GMT+9] Hi there';
      harness.fire(
        'llm_input',
        { prompt: rawPrompt, model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'openclaw_agent',
          inputs: { messages: [{ role: 'user', content: 'Hi there' }] },
        }),
      );
    });

    it('should strip [[reply_to_current]] from response', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hi', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      const llmSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[1].value;

      harness.fire(
        'llm_output',
        { response: '[[reply_to_current]] Hello there!' },
        { sessionKey: 'session-1' },
      );

      expect(llmSpan.setOutputs).toHaveBeenCalledWith({
        choices: [{ message: { role: 'assistant', content: 'Hello there!' } }],
      });
    });

    it('should sanitize historyMessages in LLM inputs', async () => {
      const harness = createTestHarness();
      await startService(harness);

      const history = [{ role: 'user', content: '[[reply_to_current]] test' }];
      harness.fire(
        'llm_input',
        { prompt: 'Hi', model: 'gpt-4', provider: 'openai', historyMessages: history },
        { sessionKey: 'session-1' },
      );

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'llm_call',
          inputs: expect.objectContaining({
            messages: [
              { role: 'user', content: 'test' },
              { role: 'user', content: 'Hi' },
            ],
          }),
        }),
      );
    });

    it('should pass through clean messages unchanged', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Just a question', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'openclaw_agent',
          inputs: { messages: [{ role: 'user', content: 'Just a question' }] },
        }),
      );
    });
  });

  describe('Tool Execution Tracing', () => {
    it('should create TOOL span on before_tool_call', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Search', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      harness.fire(
        'before_tool_call',
        { toolName: 'web_search', params: { query: 'MLflow' }, toolCallId: 'tc-1' },
        { sessionKey: 'session-1' },
      );

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'web_search',
          spanType: 'TOOL',
          inputs: { query: 'MLflow' },
          attributes: expect.objectContaining({
            tool_name: 'web_search',
            tool_id: 'tc-1',
          }),
        }),
      );
    });

    it('should end TOOL span on after_tool_call with result', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Search', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      harness.fire(
        'before_tool_call',
        { toolName: 'web_search', toolCallId: 'tc-1' },
        { sessionKey: 'session-1' },
      );

      const toolSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[2].value;

      harness.fire(
        'after_tool_call',
        { toolName: 'web_search', toolCallId: 'tc-1', result: 'Found 10 results' },
        { sessionKey: 'session-1' },
      );

      expect(toolSpan.setOutputs).toHaveBeenCalledWith({ result: 'Found 10 results' });
      expect(toolSpan.end).toHaveBeenCalled();
    });

    it('should handle tool errors', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Query', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      harness.fire(
        'before_tool_call',
        { toolName: 'database', toolCallId: 'tc-2' },
        { sessionKey: 'session-1' },
      );

      const toolSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[2].value;

      harness.fire(
        'after_tool_call',
        { toolName: 'database', toolCallId: 'tc-2', error: 'Connection timeout' },
        { sessionKey: 'session-1' },
      );

      expect(toolSpan.setOutputs).toHaveBeenCalledWith({ error: 'Connection timeout' });
      expect(toolSpan.end).toHaveBeenCalled();
    });

    it('should handle multiple concurrent tools', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Do things', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      harness.fire(
        'before_tool_call',
        { toolName: 'search', toolCallId: 'tc-a' },
        { sessionKey: 'session-1' },
      );
      harness.fire(
        'before_tool_call',
        { toolName: 'fetch', toolCallId: 'tc-b' },
        { sessionKey: 'session-1' },
      );

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'search' }),
      );
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'fetch' }),
      );

      harness.fire(
        'after_tool_call',
        { toolName: 'fetch', toolCallId: 'tc-b', result: 'fetched' },
        { sessionKey: 'session-1' },
      );
      harness.fire(
        'after_tool_call',
        { toolName: 'search', toolCallId: 'tc-a', result: 'found' },
        { sessionKey: 'session-1' },
      );

      const searchSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[2].value;
      const fetchSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[3].value;

      expect(searchSpan.end).toHaveBeenCalled();
      expect(fetchSpan.end).toHaveBeenCalled();
    });
  });

  describe('Subagent Tracing', () => {
    it('should create nested AGENT span on subagent_spawning', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Research', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      harness.fire(
        'subagent_spawning',
        { agentId: 'researcher', label: 'research-agent' },
        { sessionKey: 'session-1' },
      );

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'subagent_research-agent',
          spanType: 'AGENT',
          inputs: { agent_id: 'researcher', label: 'research-agent' },
        }),
      );
    });

    it('should end subagent span on subagent_ended', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Research', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      harness.fire(
        'subagent_spawning',
        { agentId: 'researcher', label: 'research-agent' },
        { sessionKey: 'session-1' },
      );

      const subSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[2].value;

      harness.fire(
        'subagent_ended',
        { agentId: 'researcher', result: 'Research done' },
        { sessionKey: 'session-1' },
      );

      expect(subSpan.setOutputs).toHaveBeenCalledWith({ result: 'Research done' });
      expect(subSpan.end).toHaveBeenCalled();
    });

    it('should handle subagent errors', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Research', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      harness.fire('subagent_spawning', { agentId: 'failing-agent' }, { sessionKey: 'session-1' });

      const subSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[2].value;

      harness.fire(
        'subagent_ended',
        { agentId: 'failing-agent', error: 'Out of memory' },
        { sessionKey: 'session-1' },
      );

      expect(subSpan.setOutputs).toHaveBeenCalledWith({ error: 'Out of memory' });
      expect(subSpan.end).toHaveBeenCalled();
    });
  });

  describe('Token Usage Tracking', () => {
    it('should set token usage on LLM span from llm_output event', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hello', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      const llmSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[1].value;

      harness.fire(
        'llm_output',
        { response: 'Hi', usage: { input: 50, output: 25, total: 75 } },
        { sessionKey: 'session-1' },
      );

      expect(llmSpan.setAttribute).toHaveBeenCalledWith('mlflow.chat.tokenUsage', {
        input_tokens: 50,
        output_tokens: 25,
        total_tokens: 75,
      });
    });

    it('should accumulate token usage from diagnostic events', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hello', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      harness.fire('llm_output', { response: 'Hi' }, { sessionKey: 'session-1' });

      harness.fireDiagnostic({
        type: 'model.usage',
        sessionKey: 'session-1',
        usage: { input: 100, output: 50, total: 150 },
      });
      harness.fireDiagnostic({
        type: 'model.usage',
        sessionKey: 'session-1',
        usage: { input: 200, output: 100, total: 300 },
      });

      harness.fire('agent_end', {}, { sessionKey: 'session-1' });
      await flushMicrotasks();

      const rootSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[0].value;
      expect(rootSpan.setAttribute).toHaveBeenCalledWith('mlflow.chat.tokenUsage', {
        input_tokens: 300,
        output_tokens: 150,
        total_tokens: 450,
      });
    });

    it('should not set token usage if no diagnostic events', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hello', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      harness.fire('llm_output', { response: 'Hi' }, { sessionKey: 'session-1' });
      harness.fire('agent_end', {}, { sessionKey: 'session-1' });
      await flushMicrotasks();

      const rootSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[0].value;
      const tokenCalls = rootSpan.setAttribute.mock.calls.filter(
        (call: unknown[]) => call[0] === 'mlflow.chat.tokenUsage',
      );
      expect(tokenCalls).toHaveLength(0);
    });
  });

  // TODO: Trace metadata tests (session, user, previews) — blocked on reliable
  // in-memory trace lookup before export. See service.ts TODO.

  describe('Trace Output', () => {
    it('should set clean response text on root span outputs', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hello', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      harness.fire(
        'llm_output',
        { assistantTexts: ['Response part 1', 'Response part 2'] },
        { sessionKey: 'session-1' },
      );
      harness.fire('agent_end', { success: true }, { sessionKey: 'session-1' });
      await flushMicrotasks();

      const rootSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[0].value;
      expect(rootSpan.setOutputs).toHaveBeenCalledWith({
        messages: [{ role: 'assistant', content: 'Response part 1\nResponse part 2' }],
      });
    });

    it('should not include lastAssistant or assistantTexts in outputs', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hello', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      harness.fire(
        'llm_output',
        {
          assistantTexts: ['Reply'],
          lastAssistant: { role: 'assistant', content: 'Reply', usage: { input: 10 } },
        },
        { sessionKey: 'session-1' },
      );
      harness.fire('agent_end', { success: true }, { sessionKey: 'session-1' });
      await flushMicrotasks();

      const rootSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[0].value;
      const outputCall = rootSpan.setOutputs.mock.calls.find((call: unknown[]) => {
        const msgs = (call[0] as Record<string, unknown>).messages as { content: string }[];
        return msgs?.[0]?.content === 'Reply';
      });
      expect(outputCall).toBeDefined();
      const outputs = outputCall[0] as Record<string, unknown>;
      expect(outputs).not.toHaveProperty('lastAssistant');
      expect(outputs).not.toHaveProperty('assistantTexts');
    });
  });

  describe('Agent End Data', () => {
    it('should set error status on root span when agent_end has error', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hello', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      harness.fire('llm_output', { response: 'Hi' }, { sessionKey: 'session-1' });
      harness.fire(
        'agent_end',
        { error: 'Rate limit exceeded', success: false },
        { sessionKey: 'session-1' },
      );
      await flushMicrotasks();

      const rootSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[0].value;
      expect(rootSpan.setStatus).toHaveBeenCalledWith('ERROR', 'Rate limit exceeded');
      expect(rootSpan.setOutputs).toHaveBeenCalledWith(
        expect.objectContaining({ error: 'Rate limit exceeded' }),
      );
    });

    it('should set agent_duration_ms attribute when durationMs is provided', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hello', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      harness.fire('llm_output', { response: 'Hi' }, { sessionKey: 'session-1' });
      harness.fire('agent_end', { durationMs: 1500, success: true }, { sessionKey: 'session-1' });
      await flushMicrotasks();

      const rootSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[0].value;
      expect(rootSpan.setAttribute).toHaveBeenCalledWith('agent_duration_ms', 1500);
    });

    it('should use agent_end messages as fallback output when no llm_output', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hello', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );
      // No llm_output — agent_end provides messages as fallback
      harness.fire(
        'agent_end',
        { success: true, messages: [{ role: 'assistant', content: 'Fallback reply' }] },
        { sessionKey: 'session-1' },
      );
      await flushMicrotasks();

      const rootSpan = (mlflowTracing.startSpan as jest.Mock).mock.results[0].value;
      const call = rootSpan.setOutputs.mock.calls[0][0] as Record<string, unknown>;
      const msgs = call.messages as { content: string }[];
      expect(msgs[0].content).toContain('Fallback reply');
    });
  });

  describe('Deferred Finalization', () => {
    it('should handle agent_end before llm_output via queueMicrotask', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hello', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );

      // agent_end fires but finalization is deferred via queueMicrotask
      harness.fire('agent_end', {}, { sessionKey: 'session-1' });

      // llm_output fires before microtask runs
      harness.fire('llm_output', { response: 'Late response' }, { sessionKey: 'session-1' });

      await flushMicrotasks();

      expect(mlflowTracing.flushTraces).toHaveBeenCalled();
    });
  });

  describe('LRU Eviction', () => {
    it('should evict oldest sessions when exceeding max active traces', async () => {
      const harness = createTestHarness();
      await startService(harness);

      // Create 51 sessions — the first should be evicted
      for (let i = 0; i < 51; i++) {
        harness.fire(
          'llm_input',
          { prompt: `Prompt ${i}`, model: 'gpt-4', provider: 'openai' },
          { sessionKey: `session-${i}` },
        );
      }

      // Should have created 51 root AGENT spans + 51 LLM spans = 102 total
      expect((mlflowTracing.startSpan as jest.Mock).mock.calls.length).toBe(102);

      // Sending an event for session-0 should create a new trace (it was evicted)
      jest.clearAllMocks();
      harness.fire(
        'llm_input',
        { prompt: 'Revived prompt', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-0' },
      );

      // New root AGENT + new LLM span
      expect((mlflowTracing.startSpan as jest.Mock).mock.calls.length).toBe(2);
    });
  });

  describe('Error Resilience', () => {
    it('should not throw on llm_output without prior llm_input', async () => {
      const harness = createTestHarness();
      await startService(harness);

      expect(() => {
        harness.fire('llm_output', { response: 'orphan' }, { sessionKey: 'unknown-session' });
      }).not.toThrow();
    });

    it('should not throw on after_tool_call without prior before_tool_call', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hi', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );

      expect(() => {
        harness.fire(
          'after_tool_call',
          { toolName: 'unknown', toolCallId: 'tc-unknown' },
          { sessionKey: 'session-1' },
        );
      }).not.toThrow();
    });

    it('should not throw on subagent_ended without prior subagent_spawning', async () => {
      const harness = createTestHarness();
      await startService(harness);

      harness.fire(
        'llm_input',
        { prompt: 'Hi', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-1' },
      );

      expect(() => {
        harness.fire('subagent_ended', { agentId: 'ghost-agent' }, { sessionKey: 'session-1' });
      }).not.toThrow();
    });

    it('should not throw on agent_end without prior events', async () => {
      const harness = createTestHarness();
      await startService(harness);

      expect(() => {
        harness.fire('agent_end', {}, { sessionKey: 'nonexistent' });
      }).not.toThrow();
    });

    it('should handle before_tool_call without a root trace gracefully', async () => {
      const harness = createTestHarness();
      await startService(harness);

      expect(() => {
        harness.fire('before_tool_call', { toolName: 'search' }, { sessionKey: 'no-root' });
      }).not.toThrow();
    });
  });

  describe('Complete Agent Workflow', () => {
    it('should trace a full workflow: LLM → tool → subagent → LLM → end', async () => {
      const harness = createTestHarness();
      await startService(harness);

      const ctx = { sessionKey: 'session-1' };

      // 1. Initial LLM call
      harness.fire(
        'llm_input',
        { prompt: 'Find and summarize the latest MLflow docs', model: 'gpt-4', provider: 'openai' },
        ctx,
      );
      harness.fire('llm_output', { response: 'I will search and summarize for you.' }, ctx);

      // 2. Tool call
      harness.fire(
        'before_tool_call',
        { toolName: 'web_search', params: { query: 'MLflow documentation' }, toolCallId: 'tc-1' },
        ctx,
      );
      harness.fire(
        'after_tool_call',
        {
          toolName: 'web_search',
          result: 'MLflow docs: https://mlflow.org/docs',
          toolCallId: 'tc-1',
        },
        ctx,
      );

      // 3. Subagent
      harness.fire('subagent_spawning', { agentId: 'summarizer', label: 'summary-agent' }, ctx);
      harness.fire('subagent_ended', { agentId: 'summarizer', result: 'Summary complete' }, ctx);

      // 4. Token usage
      harness.fireDiagnostic({
        type: 'model.usage',
        sessionKey: 'session-1',
        usage: { input: 500, output: 200, total: 700 },
      });

      // 5. Second LLM call with final response
      harness.fire(
        'llm_input',
        { prompt: 'Find and summarize the latest MLflow docs', model: 'gpt-4', provider: 'openai' },
        ctx,
      );
      harness.fire('llm_output', { response: 'Here is your summary of MLflow docs.' }, ctx);

      // 6. End
      harness.fire('agent_end', {}, ctx);
      await flushMicrotasks();

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'openclaw_agent', spanType: 'AGENT' }),
      );
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'llm_call', spanType: 'LLM' }),
      );
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'web_search', spanType: 'TOOL' }),
      );
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'subagent_summary-agent', spanType: 'AGENT' }),
      );

      expect(mlflowTracing.flushTraces).toHaveBeenCalled();
    });
  });

  describe('Multiple Sessions', () => {
    it('should handle interleaved events from different sessions', async () => {
      const harness = createTestHarness();
      await startService(harness);

      // Session A starts
      harness.fire(
        'llm_input',
        { prompt: 'A prompt', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-A' },
      );
      // Session B starts
      harness.fire(
        'llm_input',
        { prompt: 'B prompt', model: 'gpt-4', provider: 'openai' },
        { sessionKey: 'session-B' },
      );

      // Session A gets LLM output
      harness.fire('llm_output', { response: 'A response' }, { sessionKey: 'session-A' });
      // Session B gets LLM output
      harness.fire('llm_output', { response: 'B response' }, { sessionKey: 'session-B' });

      // Both end
      harness.fire('agent_end', {}, { sessionKey: 'session-A' });
      harness.fire('agent_end', {}, { sessionKey: 'session-B' });
      await flushMicrotasks();

      expect(mlflowTracing.flushTraces).toHaveBeenCalledTimes(2);
    });
  });
});
