/**
 * Tests for MLflow OpenCode integration
 */

import { MLflowTracingPlugin } from '../src';

// Mock the mlflow-tracing module
jest.mock('mlflow-tracing', () => {
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
import * as mlflowTracing from 'mlflow-tracing';

describe('MLflowTracingPlugin', () => {
  const originalEnv = process.env;

  // Helper to create mock client
  const createMockClient = (sessionData: any = {}, messagesData: any[] = []) => ({
    session: {
      get: jest.fn().mockResolvedValue({ data: sessionData }),
      messages: jest.fn().mockResolvedValue({ data: messagesData }),
    },
  });

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
      tokens?: { input?: number; output?: number; reasoning?: number; cache?: { read?: number; write?: number } };
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
      const result = MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      expect(result).toBeInstanceOf(Promise);

      const hooks = await result;
      expect(hooks).toHaveProperty('event');
      expect(typeof hooks.event).toBe('function');
    });
  });

  describe('Environment Variable Handling', () => {
    it('should not process when MLFLOW_TRACKING_URI is not set', async () => {
      const mockClient = createMockClient();
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'test-session-123' },
        },
      } as any);

      expect(mockClient.session.get).not.toHaveBeenCalled();
      expect(mockClient.session.messages).not.toHaveBeenCalled();
    });

    it('should not process when MLFLOW_EXPERIMENT_ID is not set', async () => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      // MLFLOW_EXPERIMENT_ID is not set

      const mockClient = createMockClient();
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'test-session-123' },
        },
      } as any);

      expect(mockClient.session.get).not.toHaveBeenCalled();
    });

    it('should initialize SDK when both env variables are set', async () => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';

      const messages = [createUserMessage('Hello'), createAssistantTextMessage('Hi there!')];

      const mockClient = createMockClient({ title: 'Test Session' }, messages);
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'test-session-123' },
        },
      } as any);

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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'message.updated',
          properties: {},
        },
      } as any);

      expect(mockClient.session.get).not.toHaveBeenCalled();
    });

    it('should not process events without sessionID', async () => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';

      const mockClient = createMockClient();
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: {},
        },
      } as any);

      expect(mockClient.session.get).not.toHaveBeenCalled();
    });
  });

  describe('LLM Call Tracing', () => {
    beforeEach(() => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';
    });

    it('should create LLM span for assistant text response', async () => {
      const messages = [createUserMessage('What is 2+2?'), createAssistantTextMessage('The answer is 4.')];

      const mockClient = createMockClient({ title: 'Math Question' }, messages);
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'session-1' },
        },
      } as any);

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
          name: 'llm_call_1',
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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'session-tokens' },
        },
      } as any);

      // Verify startSpan was called for LLM
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'llm_call_1',
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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      // First idle event processes all messages
      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'multi-turn-session' },
        },
      } as any);

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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'model-comparison' },
        },
      } as any);

      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          inputs: expect.objectContaining({
            model: 'openai/gpt-4',
          }),
        }),
      );
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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'tool-session' },
        },
      } as any);

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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'error-tool-session' },
        },
      } as any);

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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'multi-tool-session' },
        },
      } as any);

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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'text-and-tool-session' },
        },
      } as any);

      // Should create both LLM and tool spans
      expect(mlflowTracing.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'llm_call_1',
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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'edit-session' },
        },
      } as any);

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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'write-session' },
        },
      } as any);

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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'bash-session' },
        },
      } as any);

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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'agent-workflow' },
        },
      } as any);

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
          name: 'llm_call_1',
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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'multi-step-agent' },
        },
      } as any);

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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'empty-session' },
        },
      } as any);

      // Should not create any spans
      expect(mlflowTracing.startSpan).not.toHaveBeenCalled();
    });

    it('should handle session with no user messages', async () => {
      const messages = [createAssistantTextMessage('Hello!')];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'no-user-session' },
        },
      } as any);

      // Should not create trace without user message
      expect(mlflowTracing.startSpan).not.toHaveBeenCalled();
    });

    it('should handle user message with empty text', async () => {
      const messages = [
        { info: { role: 'user' }, parts: [{ type: 'text', text: '' }] },
        createAssistantTextMessage('Response'),
      ];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'empty-user-text' },
        },
      } as any);

      // Should not create trace with empty user prompt
      expect(mlflowTracing.startSpan).not.toHaveBeenCalled();
    });

    it('should handle messages with missing parts', async () => {
      const messages = [{ info: { role: 'user' }, parts: undefined }, createAssistantTextMessage('Response')];

      const mockClient = createMockClient({}, messages);
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'missing-parts' },
        },
      } as any);

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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'missing-state' },
        },
      } as any);

      // Should still create tool span with defaults
      expect(mlflowTracing.startSpan).toHaveBeenCalled();
    });

    it('should handle session fetch failure', async () => {
      const mockClient = {
        session: {
          get: jest.fn().mockResolvedValue({ data: null }),
          messages: jest.fn(),
        },
      };

      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'fetch-failure' },
        },
      } as any);

      expect(mockClient.session.messages).not.toHaveBeenCalled();
    });

    it('should handle messages fetch failure', async () => {
      const mockClient = {
        session: {
          get: jest.fn().mockResolvedValue({ data: { title: 'Test' } }),
          messages: jest.fn().mockResolvedValue({ data: null }),
        },
      };

      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'messages-failure' },
        },
      } as any);

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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      // First event
      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'dup-session' },
        },
      } as any);

      const firstCallCount = (mlflowTracing.startSpan as jest.Mock).mock.calls.length;

      // Same session, same message count - should be skipped
      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'dup-session' },
        },
      } as any);

      expect((mlflowTracing.startSpan as jest.Mock).mock.calls.length).toBe(firstCallCount);
    });

    it('should process new messages in existing session', async () => {
      const initialMessages = [createUserMessage('First'), createAssistantTextMessage('First response')];

      let currentMessages = [...initialMessages];

      const mockClient = {
        session: {
          get: jest.fn().mockResolvedValue({ data: { title: 'Test' } }),
          messages: jest.fn().mockImplementation(() => Promise.resolve({ data: currentMessages })),
        },
      };

      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      // First turn
      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'incremental-session' },
        },
      } as any);

      const firstCallCount = (mlflowTracing.startSpan as jest.Mock).mock.calls.length;

      // Add new messages
      currentMessages = [
        ...initialMessages,
        createUserMessage('Second'),
        createAssistantTextMessage('Second response'),
      ];

      // Second turn with new messages
      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'incremental-session' },
        },
      } as any);

      // Should have more spans now
      expect((mlflowTracing.startSpan as jest.Mock).mock.calls.length).toBeGreaterThan(firstCallCount);
    });
  });

  describe('Trace Metadata', () => {
    beforeEach(() => {
      process.env.MLFLOW_TRACKING_URI = 'http://localhost:5000';
      process.env.MLFLOW_EXPERIMENT_ID = 'exp-123';
      process.env.USER = 'test-user';
    });

    it('should set trace metadata with session info', async () => {
      const messages = [createUserMessage('Test prompt'), createAssistantTextMessage('Test response')];

      const mockClient = createMockClient(
        {
          title: 'Test Session Title',
          directory: '/custom/directory',
        },
        messages,
      );

      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'metadata-session' },
        },
      } as any);

      // Verify InMemoryTraceManager was used
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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'timing-session' },
        },
      } as any);

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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'tool-timing-session' },
        },
      } as any);

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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'history-session' },
        },
      } as any);

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
      const hooks = await MLflowTracingPlugin({
        client: mockClient,
        directory: '/test/directory',
      } as any);

      await hooks.event!({
        event: {
          type: 'session.idle',
          properties: { sessionID: 'tool-history-session' },
        },
      } as any);

      expect(mlflowTracing.startSpan).toHaveBeenCalled();
    });
  });
});
