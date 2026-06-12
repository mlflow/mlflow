/**
 * Tests for MLflow OpenCode integration
 */

import { MLflowTracingPlugin } from '../src';
import type { PluginInput } from '@opencode-ai/plugin';

interface PluginClient {
  session: {
    messages: jest.Mock;
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type EventParams = any;

// Mock the @mlflow/core module
jest.mock('@mlflow/core', () => {
  const mockSpan = {
    traceId: 'mock-trace-id',
    setAttribute: jest.fn(),
    setOutputs: jest.fn(),
    end: jest.fn(),
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
    startSpan: jest.fn(() => mockSpan),
    flushTraces: jest.fn().mockResolvedValue(undefined),
    SpanType: {
      LLM: 'LLM',
      TOOL: 'TOOL',
      AGENT: 'AGENT',
    },
    SpanAttributeKey: {
      TOKEN_USAGE: 'token_usage',
    },
    TraceMetadataKey: {
      TRACE_SESSION: 'mlflow.trace.session',
      TRACE_USER: 'mlflow.trace.user',
    },
    InMemoryTraceManager: {
      getInstance: jest.fn(() => ({
        getTrace: jest.fn(() => mockTrace),
      })),
    },
  };
});

// Import mocked functions after mocking
import * as mlflowTracing from '@mlflow/core';

describe('MLflowTracingPlugin', () => {
  const originalEnv = process.env;

  // Helper to create mock client
  const createMockClient = (
    _sessionData: Record<string, unknown> = {},
    messagesData: unknown[] = [],
  ): PluginClient => ({
    session: {
      messages: jest.fn().mockResolvedValue({ data: messagesData }),
    },
  });

  // Helper to create plugin input
  const createPluginInput = (client: PluginClient): PluginInput =>
    ({
      client,
    }) as unknown as PluginInput;

  // Helper to create event params
  const createEventParams = (type: string, properties?: Record<string, unknown>): EventParams => ({
    event: { type, properties },
  });

  // Helper to create session idle event
  const createSessionIdleEvent = (sessionID: string): EventParams =>
    createEventParams('session.idle', { sessionID });

  // Helper to create a user message
  const createUserMessage = (text: string, time?: { created?: number; completed?: number }) => ({
    info: {
      role: 'user',
      time: time || { created: Date.now(), completed: Date.now() },
    },
    parts: [{ type: 'text', text }],
  });

  // Helper to create an assistant message with text
  const createAssistantTextMessage = (
    text: string,
    options?: {
      modelID?: string;
      providerID?: string;
      tokens?: {
        input?: number;
        output?: number;
        reasoning?: number;
        cache?: { read?: number; write?: number };
      };
      time?: { created?: number; completed?: number };
    },
  ) => ({
    info: {
      role: 'assistant',
      modelID: options?.modelID || 'claude-3-opus',
      providerID: options?.providerID || 'anthropic',
      tokens: options?.tokens || { input: 100, output: 50 },
      time: options?.time || { created: Date.now(), completed: Date.now() + 1000 },
    },
    parts: [{ type: 'text', text }],
  });

  // Helper to create an assistant message with tool call
  const createToolCallMessage = (
    toolName: string,
    options?: {
      callID?: string;
      status?: string;
      input?: Record<string, unknown>;
      output?: string;
      error?: string;
      title?: string;
      time?: { start?: number; end?: number };
      modelID?: string;
      providerID?: string;
    },
  ) => ({
    info: {
      role: 'assistant',
      modelID: options?.modelID || 'claude-3-opus',
      providerID: options?.providerID || 'anthropic',
      time: { created: Date.now(), completed: Date.now() + 1000 },
    },
    parts: [
      {
        type: 'tool',
        tool: toolName,
        callID: options?.callID || `call-${toolName}-${Date.now()}`,
        state: {
          status: options?.status || 'completed',
          input: options?.input || {},
          output: options?.output || '',
          error: options?.error,
          title: options?.title || `${toolName} completed`,
          time: options?.time || { start: Date.now(), end: Date.now() + 500 },
        },
      },
    ],
  });

  // Helper to create an assistant message with reasoning and optional text
  const createAssistantReasoningMessage = (
    reasoningText: string,
    text?: string,
    options?: {
      modelID?: string;
      providerID?: string;
      tokens?: {
        input?: number;
        output?: number;
        reasoning?: number;
        cache?: { read?: number; write?: number };
      };
      time?: { created?: number; completed?: number };
    },
  ) => ({
    info: {
      role: 'assistant',
      modelID: options?.modelID || 'claude-3-opus',
      providerID: options?.providerID || 'anthropic',
      tokens: options?.tokens || { input: 100, output: 50, reasoning: 30 },
      time: options?.time || { created: Date.now(), completed: Date.now() + 1000 },
    },
    parts: [
      ...(reasoningText
        ? [
            {
              type: 'reasoning',
              text: reasoningText,
              time: { start: Date.now(), end: Date.now() + 500 },
            },
          ]
        : []),
      ...(text ? [{ type: 'text', text }] : []),
    ],
  });

  // Helper to create an assistant message with both text and tool call
  const createAssistantMessageWithTextAndTool = (
    text: string,
    toolName: string,
    toolOptions?: {
      callID?: string;
      status?: string;
      input?: Record<string, unknown>;
      output?: string;
    },
  ) => ({
    info: {
      role: 'assistant',
      modelID: 'claude-3-opus',
      providerID: 'anthropic',
      tokens: { input: 100, output: 50 },
      time: { created: Date.now(), completed: Date.now() + 1000 },
    },
    parts: [
      { type: 'text', text },
      {
        type: 'tool',
        tool: toolName,
        callID: toolOptions?.callID || `call-${toolName}`,
        state: {
          status: toolOptions?.status || 'completed',
          input: toolOptions?.input || {},
          output: toolOptions?.output || '',
          title: `${toolName} completed`,
          time: { start: Date.now(), end: Date.now() + 500 },
        },
      },
    ],
  });

  beforeEach(() => {
    jest.clearAllMocks();
    jest.resetModules();
    process.env = { ...originalEnv };
    delete process.env.MLFLOW_TRACKING_URI;
    delete process.env.MLFLOW_EXPERIMENT_ID;
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  describe('Plugin Initialization', () => {
    it('should export MLflowTracingPlugin function', () => {
      expect(typeof MLflowTracingPlugin).toBe('function');
    });

    it('should return a promise with hooks when called', async () => {
      const mockClient = createMockClient();
      const result = MLflowTracingPlugin(createPluginInput(mockClient));

      expect(result).toBeInstanceOf(Promise);

      const hooks = await result;
      expect(hooks).toHaveProperty('event');
      expect(typeof hooks.event).toBe('function');
    });
  });

  describe('Environment Variable Handling', () => {
    it('should not process when MLFLOW_TRACKING_URI is not set', async () => {
      const mockClient = createMockClient();
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('test-session-123'));

      expect(mockClient.session.messages).not.toHaveBeenCalled();
    });

    it('should not process when MLFLOW_EXPERIMENT_ID is not set', async () => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      // MLFLOW_EXPERIMENT_ID is not set

      const mockClient = createMockClient();
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('test-session-123'));

      expect(mockClient.session.messages).not.toHaveBeenCalled();
    });

    it('should initialize SDK when both env variables are set', async () => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';

      const messages = [createUserMessage('Hello'), createAssistantTextMessage('Hi there!')];

      const mockClient = createMockClient({ title: 'Test Session' }, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('test-session-123'));

      expect(mlflowTracing.init).toHaveBeenCalledWith({
        trackingUri: 'http://localhost:5000',
        experimentId: 'exp-123',
      });
    });
  });

  describe('Event Filtering', () => {
    it('should not process non session.idle events', async () => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';

      const mockClient = createMockClient();
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createEventParams('message.updated', {}));

      expect(mockClient.session.messages).not.toHaveBeenCalled();
    });

    it('should not process events without sessionID', async () => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';

      const mockClient = createMockClient();
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createEventParams('session.idle', {}));

      expect(mockClient.session.messages).not.toHaveBeenCalled();
    });
  });

  describe('LLM Call Tracing', () => {
    beforeEach(() => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';
    });

    it('should create LLM span for assistant text response', async () => {
      const messages = [
        createUserMessage('What is 2+2?'),
        createAssistantTextMessage('The answer is 4.'),
      ];

      const mockClient = createMockClient({ title: 'Math Question' }, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('session-1'));

      // Should create parent span (AGENT type)
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'opencode_conversation',
          spanType: 'AGENT',
          inputs: { prompt: 'What is 2+2?' },
        }),
      );

      // Should create LLM span
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'llm_call',
          spanType: 'LLM',
          inputs: expect.objectContaining({
            model: 'anthropic/claude-3-opus',
          }),
        }),
      );

      expect(mlflowTracing.flushTraces).toHaveBeenCalled();
    });

    it('should track token usage in LLM spans', async () => {
      const messages = [
        createUserMessage('Tell me a story'),
        createAssistantTextMessage('Once upon a time...', {
          tokens: {
            input: 150,
            output: 200,
            reasoning: 50,
            cache: { read: 25, write: 10 },
          },
        }),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('session-tokens'));

      // Verify startSpan was called for LLM
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'llm_call',
          spanType: 'LLM',
        }),
      );
    });

    it('should create multiple LLM spans for multi-turn conversation', async () => {
      const messages = [
        createUserMessage('First question'),
        createAssistantTextMessage('First answer'),
        createUserMessage('Second question'),
        createAssistantTextMessage('Second answer'),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      // First idle event processes all messages
      await hooks.event!(createSessionIdleEvent('multi-turn-session'));

      // Should have created spans
      expect(mlflowTracing.startSpan).toHaveBeenCalled();
    });

    it('should handle assistant messages with different models', async () => {
      const messages = [
        createUserMessage('Compare models'),
        createAssistantTextMessage('GPT response', {
          modelID: 'gpt-4',
          providerID: 'openai',
        }),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('model-comparison'));

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          inputs: expect.objectContaining({
            model: 'openai/gpt-4',
          }),
        }),
      );
    });

    it('should include reasoning content in LLM span outputs', async () => {
      const messages = [
        createUserMessage('Are you conscious?'),
        createAssistantReasoningMessage(
          'The user is asking a philosophical question about AI consciousness.',
          "I don't know — that's the most honest answer I can give.",
        ),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('reasoning-session'));
      const mockSpan = (mlflowTracing.startSpan as jest.Mock)();

      expect(mockSpan.setOutputs).toHaveBeenCalledWith({
        choices: [
          {
            message: expect.objectContaining({
              role: 'assistant',
              content: "I don't know — that's the most honest answer I can give.",
              reasoning: 'The user is asking a philosophical question about AI consciousness.',
            }),
          },
        ],
      });
    });

    it('should not include reasoning field when no reasoning parts exist', async () => {
      const messages = [createUserMessage('Hello'), createAssistantTextMessage('Hi there!')];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('no-reasoning-session'));

      const mockSpan = (mlflowTracing.startSpan as jest.Mock)();
      // first argument when setOutputs got called first time
      const outputCall = mockSpan.setOutputs.mock.calls[0][0];
      expect(outputCall.choices[0].message).not.toHaveProperty('reasoning');
    });
  });

  describe('Tool Call Tracing', () => {
    beforeEach(() => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';
    });

    it('should create tool span for completed tool call', async () => {
      const messages = [
        createUserMessage('Read a file'),
        createToolCallMessage('Read', {
          callID: 'read-call-1',
          status: 'completed',
          input: { file_path: '/test/file.txt' },
          output: 'File contents here',
          title: 'Read file.txt',
        }),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('tool-session'));

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tool_Read',
          spanType: 'TOOL',
          inputs: { file_path: '/test/file.txt' },
          attributes: expect.objectContaining({
            tool_name: 'Read',
            tool_id: 'read-call-1',
            status: 'completed',
          }),
        }),
      );
    });

    it('should handle errored tool calls', async () => {
      const messages = [
        createUserMessage('Run a command'),
        createToolCallMessage('Bash', {
          callID: 'bash-call-1',
          status: 'error',
          input: { command: 'invalid-command' },
          error: 'Command not found',
        }),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('error-tool-session'));

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tool_Bash',
          spanType: 'TOOL',
          attributes: expect.objectContaining({
            status: 'error',
          }),
        }),
      );
    });

    it('should handle multiple tool calls in sequence', async () => {
      const messages = [
        createUserMessage('List and read files'),
        createToolCallMessage('Glob', {
          callID: 'glob-call-1',
          input: { pattern: '*.ts' },
          output: 'file1.ts\nfile2.ts',
        }),
        createToolCallMessage('Read', {
          callID: 'read-call-1',
          input: { file_path: 'file1.ts' },
          output: 'console.log("hello");',
        }),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('multi-tool-session'));

      // Should create spans for both tools
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tool_Glob',
        }),
      );
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tool_Read',
        }),
      );
    });

    it('should handle message with both text and tool call', async () => {
      const messages = [
        createUserMessage('Explain and read file'),
        createAssistantMessageWithTextAndTool("I'll read the file for you.", 'Read', {
          input: { file_path: '/test/example.ts' },
          output: 'const x = 1;',
        }),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('text-and-tool-session'));

      // Should create both LLM and tool spans
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'llm_call',
          spanType: 'LLM',
        }),
      );
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tool_Read',
          spanType: 'TOOL',
        }),
      );
    });

    it('should handle Edit tool with file modifications', async () => {
      const messages = [
        createUserMessage('Fix the bug'),
        createToolCallMessage('Edit', {
          callID: 'edit-call-1',
          status: 'completed',
          input: {
            file_path: '/src/index.ts',
            old_string: 'const x = 1',
            new_string: 'const x = 2',
          },
          output: 'File edited successfully',
          title: 'Edit index.ts',
        }),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('edit-session'));

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tool_Edit',
          spanType: 'TOOL',
          inputs: expect.objectContaining({
            file_path: '/src/index.ts',
          }),
        }),
      );
    });

    it('should handle Write tool for file creation', async () => {
      const messages = [
        createUserMessage('Create a new file'),
        createToolCallMessage('Write', {
          callID: 'write-call-1',
          status: 'completed',
          input: {
            file_path: '/src/new-file.ts',
            content: 'export const hello = "world";',
          },
          output: 'File created',
        }),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('write-session'));

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tool_Write',
          spanType: 'TOOL',
        }),
      );
    });

    it('should handle Bash tool execution', async () => {
      const messages = [
        createUserMessage('Run the tests'),
        createToolCallMessage('Bash', {
          callID: 'bash-call-1',
          status: 'completed',
          input: { command: 'npm test' },
          output: 'All tests passed',
        }),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('bash-session'));

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tool_Bash',
          spanType: 'TOOL',
          inputs: { command: 'npm test' },
        }),
      );
    });
  });

  describe('Agent Workflow Tracing', () => {
    beforeEach(() => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';
    });

    it('should trace complete agent workflow with LLM + tools + response', async () => {
      const messages = [
        createUserMessage('Find all TypeScript files and count them'),
        createToolCallMessage('Glob', {
          callID: 'glob-1',
          input: { pattern: '**/*.ts' },
          output: 'src/index.ts\nsrc/utils.ts\nsrc/types.ts',
        }),
        createAssistantTextMessage('I found 3 TypeScript files in the project.'),
      ];

      const mockClient = createMockClient({ title: 'File Count' }, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('agent-workflow'));

      // Should create parent AGENT span
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'opencode_conversation',
          spanType: 'AGENT',
        }),
      );

      // Should create TOOL span
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tool_Glob',
          spanType: 'TOOL',
        }),
      );

      // Should create LLM span
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'llm_call',
          spanType: 'LLM',
        }),
      );
    });

    it('should trace complex multi-step agent workflow', async () => {
      const messages = [
        createUserMessage('Implement a new function'),
        createToolCallMessage('Read', {
          callID: 'read-1',
          input: { file_path: '/src/utils.ts' },
          output: 'export function existing() {}',
        }),
        createToolCallMessage('Edit', {
          callID: 'edit-1',
          input: { file_path: '/src/utils.ts', old_string: '{}', new_string: '{ return true; }' },
          output: 'Edited',
        }),
        createAssistantTextMessage('I have updated the function.'),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('multi-step-agent'));

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'tool_Read' }),
      );
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'tool_Edit' }),
      );
    });
  });

  describe('Edge Cases', () => {
    beforeEach(() => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';
    });

    it('should handle empty messages array', async () => {
      const mockClient = createMockClient({}, []);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('empty-session'));

      // Should not create any spans
      expect(mlflowTracing.startSpan).not.toHaveBeenCalled();
    });

    it('should handle session with no user messages', async () => {
      const messages = [createAssistantTextMessage('Hello!')];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('no-user-session'));

      // Should not create trace without user message
      expect(mlflowTracing.startSpan).not.toHaveBeenCalled();
    });

    it('should handle user message with empty text', async () => {
      const messages = [
        { info: { role: 'user' }, parts: [{ type: 'text', text: '' }] },
        createAssistantTextMessage('Response'),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('empty-user-text'));

      // Should not create trace with empty user prompt
      expect(mlflowTracing.startSpan).not.toHaveBeenCalled();
    });

    it('should handle messages with missing parts', async () => {
      const messages = [
        { info: { role: 'user' }, parts: undefined },
        createAssistantTextMessage('Response'),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('missing-parts'));

      // Should handle gracefully
      expect(mlflowTracing.startSpan).not.toHaveBeenCalled();
    });

    it('should handle tool call with missing state', async () => {
      const messages = [
        createUserMessage('Do something'),
        {
          info: { role: 'assistant' },
          parts: [{ type: 'tool', tool: 'Unknown', callID: 'unknown-1' }],
        },
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('missing-state'));

      // Should still create tool span with defaults
      expect(mlflowTracing.startSpan).toHaveBeenCalled();
    });

    it('should handle messages fetch failure', async () => {
      const mockClient = {
        session: {
          messages: jest.fn().mockResolvedValue({ data: null }),
        },
      };

      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient as PluginClient));

      await hooks.event!(createSessionIdleEvent('messages-failure'));

      expect(mlflowTracing.startSpan).not.toHaveBeenCalled();
    });
  });

  describe('Duplicate Turn Prevention', () => {
    beforeEach(() => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';
    });

    it('should not process the same turn twice', async () => {
      const messages = [createUserMessage('Hello'), createAssistantTextMessage('Hi!')];

      const mockClient = createMockClient({ title: 'Test' }, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      // First event
      await hooks.event!(createSessionIdleEvent('dup-session'));

      const firstCallCount = (mlflowTracing.startSpan as jest.Mock).mock.calls.length;

      // Same session, same message count - should be skipped
      await hooks.event!(createSessionIdleEvent('dup-session'));

      expect((mlflowTracing.startSpan as jest.Mock).mock.calls.length).toBe(firstCallCount);
    });

    it('should process new messages in existing session', async () => {
      const initialMessages = [
        createUserMessage('First'),
        createAssistantTextMessage('First response'),
      ];

      let currentMessages = [...initialMessages];

      const mockClient = {
        session: {
          messages: jest.fn().mockImplementation(() => Promise.resolve({ data: currentMessages })),
        },
      };

      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient as PluginClient));

      // First turn
      await hooks.event!(createSessionIdleEvent('incremental-session'));

      const firstCallCount = (mlflowTracing.startSpan as jest.Mock).mock.calls.length;

      // Add new messages
      currentMessages = [
        ...initialMessages,
        createUserMessage('Second'),
        createAssistantTextMessage('Second response'),
      ];

      // Second turn with new messages
      await hooks.event!(createSessionIdleEvent('incremental-session'));

      // Should have more spans now
      expect((mlflowTracing.startSpan as jest.Mock).mock.calls.length).toBeGreaterThan(
        firstCallCount,
      );
    });
  });

  describe('Trace Metadata', () => {
    beforeEach(() => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';
      process.env.USER = 'test-user';
    });

    it('should set trace metadata with session info', async () => {
      const messages = [
        createUserMessage('Test prompt'),
        createAssistantTextMessage('Test response'),
      ];

      const mockClient = createMockClient(
        {
          title: 'Test Session Title',
          directory: '/custom/directory',
        },
        messages,
      );

      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('metadata-session'));

      // Verify InMemoryTraceManager was used
      // eslint-disable-next-line @typescript-eslint/unbound-method
      expect(mlflowTracing.InMemoryTraceManager.getInstance).toHaveBeenCalled();
    });
  });

  describe('Timing Information', () => {
    beforeEach(() => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';
    });

    it('should pass timing information to spans', async () => {
      const startTime = Date.now();
      const endTime = startTime + 5000;

      const messages = [
        createUserMessage('Timed request', { created: startTime, completed: startTime + 100 }),
        createAssistantTextMessage('Timed response', {
          time: { created: startTime + 100, completed: endTime },
        }),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('timing-session'));

      // Verify spans were created with timing
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          startTimeNs: expect.any(Number),
        }),
      );
    });

    it('should handle tool timing information', async () => {
      const toolStart = Date.now();
      const toolEnd = toolStart + 1000;

      const messages = [
        createUserMessage('Run tool'),
        createToolCallMessage('Read', {
          time: { start: toolStart, end: toolEnd },
          input: { file_path: '/test.ts' },
          output: 'content',
        }),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('tool-timing-session'));

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tool_Read',
          startTimeNs: expect.any(Number),
        }),
      );
    });
  });

  describe('Conversation History Reconstruction', () => {
    beforeEach(() => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';
    });

    it('should reconstruct conversation history for LLM inputs', async () => {
      const messages = [
        createUserMessage('What is TypeScript?'),
        createAssistantTextMessage('TypeScript is a typed superset of JavaScript.'),
        createUserMessage('How do I install it?'),
        createAssistantTextMessage('Run npm install -g typescript'),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('history-session'));

      // The second LLM call should include the full conversation history
      expect(mlflowTracing.startSpan).toHaveBeenCalled();
    });

    it('should include tool results in conversation history', async () => {
      const messages = [
        createUserMessage('List files'),
        {
          info: { role: 'assistant' },
          parts: [
            {
              type: 'tool',
              tool: 'Glob',
              callID: 'glob-1',
              state: {
                status: 'completed',
                input: { pattern: '*.ts' },
                output: 'index.ts\nutils.ts',
              },
            },
          ],
        },
        createUserMessage('What do they contain?'),
        createAssistantTextMessage('The files contain TypeScript code.'),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('tool-history-session'));

      expect(mlflowTracing.startSpan).toHaveBeenCalled();
    });

    it('should include reasoning from prior assistant messages in conversation history', async () => {
      // The plugin only traces messages after the last user message.
      // So messages[3] (assistant) is the only LLM span created,
      // but input to this span (inputs.messages) should include the conversation history
      // from messages [0]-[2], including reasoning part as well from message[1].
      const messages = [
        createUserMessage('Are you conscious?'),
        createAssistantReasoningMessage(
          'The user is asking a philosophical question.',
          "I don't know.",
        ),
        createUserMessage('Tell me more'),
        createAssistantTextMessage('Here is more detail.'),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('reasoning-history-session'));

      // Find the llm_call span — its inputs.messages should contain
      // the prior assistant message's reasoning in the conversation history
      const llmCalls = (mlflowTracing.startSpan as jest.Mock).mock.calls.filter(
        (call: [{ name: string }]) => call[0].name === 'llm_call',
      );
      expect(llmCalls.length).toBe(1);

      const inputMessages = llmCalls[0][0].inputs.messages;
      const assistantHistoryMsg = inputMessages.find(
        (m: { role: string }) => m.role === 'assistant',
      );
      expect(assistantHistoryMsg).toBeDefined();
      expect(assistantHistoryMsg.reasoning).toBe('The user is asking a philosophical question.');
      expect(assistantHistoryMsg.content).toBe("I don't know.");
    });

    it('should reconstruct full conversation history with reasoning, tools, and continuation messages', async () => {
      // Modeled after a real OpenCode session where the assistant acts
      // autonomously across multiple messages:
      //   user asks → assistant reasons + calls tool → assistant continues
      //   autonomously (analyzes tool output, calls more tools) → user
      //   asks follow-up → assistant responds (traced span)
      //
      // Key pattern: messages [3] and [4] are both assistant with no user
      // message between them — [3] calls a tool, [4] autonomously analyzes
      // the result and calls another tool.
      const messages = [
        // [0] Turn 1: user asks a question
        createUserMessage('Are you conscious?'),

        // [1] Turn 1: assistant responds with reasoning + text
        createAssistantReasoningMessage(
          'The user is asking about consciousness, not a coding question.',
          "I don't know. I process language but can't verify subjective experience.",
        ),

        // [2] Turn 2: user asks to find files
        createUserMessage('Find all files in the parent directory'),

        // [3] Turn 2: assistant reasons, explains, and calls read tool
        {
          info: {
            role: 'assistant',
            modelID: 'claude-opus',
            providerID: 'anthropic',
            tokens: { input: 200, output: 100, reasoning: 40 },
            time: { created: 1000, completed: 2000 },
          },
          parts: [
            {
              type: 'reasoning',
              text: 'The user wants to list files. I should use the read tool.',
              time: { start: 1000, end: 1200 },
            },
            { type: 'text', text: "I'll read the parent directory for you." },
            {
              type: 'tool',
              tool: 'read',
              callID: 'call-read-1',
              state: {
                status: 'completed',
                input: { path: '..' },
                output: 'src/\npackage.json\nREADME.md',
                title: 'Read directory',
                time: { start: 1200, end: 1500 },
              },
            },
          ],
        },

        // [4] Turn 2 continued: assistant autonomously analyzes the tool
        // output and calls another tool — no user message in between
        {
          info: {
            role: 'assistant',
            modelID: 'claude-opus',
            providerID: 'anthropic',
            tokens: { input: 300, output: 80 },
            time: { created: 2001, completed: 3000 },
          },
          parts: [
            {
              type: 'reasoning',
              text: 'I see 3 entries. Let me also check the src/ directory contents.',
              time: { start: 2001, end: 2200 },
            },
            { type: 'text', text: 'Found 3 entries. Let me also check what is inside src/.' },
            {
              type: 'tool',
              tool: 'read',
              callID: 'call-read-2',
              state: {
                status: 'completed',
                input: { path: '../src' },
                output: 'index.ts\nutils.ts\nconfig.ts',
                title: 'Read src directory',
                time: { start: 2200, end: 2500 },
              },
            },
          ],
        },

        // [5] Turn 2 continued: assistant gives final summary
        createAssistantTextMessage(
          'The parent directory has 3 entries. The src/ folder contains index.ts, utils.ts, and config.ts.',
        ),

        // [6] Turn 3: user asks follow-up
        createUserMessage('What does index.ts export?'),

        // [7] Turn 3: assistant responds (this is the span we trace)
        createAssistantTextMessage('index.ts exports the main plugin function.'),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin(createPluginInput(mockClient));

      await hooks.event!(createSessionIdleEvent('full-history-session'));

      // The plugin creates spans from lastUserIdx (6) + 1 = message[7].
      // message[7]'s LLM span inputs.messages should contain the full
      // conversation history from messages [0]-[6].
      const llmCalls = (mlflowTracing.startSpan as jest.Mock).mock.calls.filter(
        (call: [{ name: string }]) => call[0].name === 'llm_call',
      );
      expect(llmCalls.length).toBe(1);

      const history = llmCalls[0][0].inputs.messages;

      // Expected history structure:
      //   user: "Are you conscious?"
      //   assistant: text + reasoning (turn 1)
      //   user: "Find all files..."
      //   assistant: text + reasoning (turn 2 step 1, called read tool)
      //   tool: read result "src/\npackage.json\nREADME.md"
      //   assistant: text + reasoning (turn 2 step 2, autonomous continuation)
      //   tool: read result "index.ts\nutils.ts\nconfig.ts"
      //   assistant: text only (turn 2 final summary)
      //   user: "What does index.ts export?"

      // Check user messages
      const userMsgs = history.filter((m: { role: string }) => m.role === 'user');
      expect(userMsgs.length).toBe(3);
      expect(userMsgs[0].content).toBe('Are you conscious?');
      expect(userMsgs[1].content).toBe('Find all files in the parent directory');
      expect(userMsgs[2].content).toBe('What does index.ts export?');

      // Check assistant messages — should include reasoning where it existed
      const assistantMsgs = history.filter((m: { role: string }) => m.role === 'assistant');
      expect(assistantMsgs.length).toBe(4);

      // Turn 1 assistant: had reasoning
      expect(assistantMsgs[0].reasoning).toBe(
        'The user is asking about consciousness, not a coding question.',
      );
      expect(assistantMsgs[0].content).toBe(
        "I don't know. I process language but can't verify subjective experience.",
      );

      // Turn 2 step 1: had reasoning + tool call
      expect(assistantMsgs[1].reasoning).toBe(
        'The user wants to list files. I should use the read tool.',
      );
      expect(assistantMsgs[1].content).toBe("I'll read the parent directory for you.");

      // Turn 2 step 2: autonomous continuation with reasoning
      expect(assistantMsgs[2].reasoning).toBe(
        'I see 3 entries. Let me also check the src/ directory contents.',
      );
      expect(assistantMsgs[2].content).toBe(
        'Found 3 entries. Let me also check what is inside src/.',
      );

      // Turn 2 final: summary, no reasoning
      expect(assistantMsgs[3].content).toBe(
        'The parent directory has 3 entries. The src/ folder contains index.ts, utils.ts, and config.ts.',
      );
      expect(assistantMsgs[3].reasoning).toBeUndefined();

      // Check tool results — both tool calls should be in history
      const toolMsgs = history.filter((m: { role: string }) => m.role === 'tool');
      expect(toolMsgs.length).toBe(2);
      expect(toolMsgs[0].tool_call_id).toBe('call-read-1');
      expect(toolMsgs[0].content).toBe('src/\npackage.json\nREADME.md');
      expect(toolMsgs[1].tool_call_id).toBe('call-read-2');
      expect(toolMsgs[1].content).toBe('index.ts\nutils.ts\nconfig.ts');
    });
  });
});
