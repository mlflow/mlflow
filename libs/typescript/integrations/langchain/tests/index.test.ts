/**
 * Tests for MLflow LangChain integration
 */

import * as mlflow from '@mlflow/core';
import { tracedModel } from '../src';
import { createAuthProvider } from '@mlflow/core/src/auth';

const TEST_TRACKING_URI = 'http://localhost:5000';

/**
 * Create a mock LangChain AIMessage-like object
 */
function createMockAIMessage(
  content: string,
  options?: {
    tool_calls?: Array<{ id: string; name: string; args: Record<string, unknown> }>;
    usage_metadata?: {
      input_tokens: number;
      output_tokens: number;
    };
    response_metadata?: Record<string, unknown>;
  },
) {
  return {
    content,
    tool_calls: options?.tool_calls ?? [],
    usage_metadata: options?.usage_metadata ?? { input_tokens: 10, output_tokens: 20 },
    response_metadata: options?.response_metadata ?? {},
    _getType() {
      return 'ai';
    },
    // concat for chunk aggregation
    concat(other: any) {
      return createMockAIMessage(
        content + (other.content || ''),
        {
          tool_calls: [...(this.tool_calls || []), ...(other.tool_calls || [])],
          usage_metadata: other.usage_metadata ?? this.usage_metadata,
          response_metadata: other.response_metadata ?? this.response_metadata,
        },
      );
    },
  };
}

/**
 * Create a mock LangChain HumanMessage-like object
 */
function createMockHumanMessage(content: string) {
  return {
    content,
    _getType() {
      return 'human';
    },
  };
}

/**
 * Create a mock LangChain BaseChatModel
 */
function createMockModel(
  className: string,
  options?: {
    invokeResult?: any;
    streamChunks?: any[];
    invokeError?: Error;
    streamError?: Error;
  },
) {
  const invokeResult = options?.invokeResult ?? createMockAIMessage('Hello!');
  const streamChunks = options?.streamChunks ?? [
    createMockAIMessage('Hel', { usage_metadata: undefined as any }),
    createMockAIMessage('lo!', { usage_metadata: { input_tokens: 10, output_tokens: 20 } }),
  ];

  const model = {
    invoke: async (...args: any[]) => {
      if (options?.invokeError) throw options.invokeError;
      return invokeResult;
    },
    stream: async (...args: any[]) => {
      if (options?.streamError) throw options.streamError;
      // Return an async iterable
      return {
        [Symbol.asyncIterator]: async function* () {
          for (const chunk of streamChunks) {
            yield chunk;
          }
        },
      };
    },
    bindTools: (tools: any[], toolConfig?: any) => {
      // Return a new mock model with tools bound
      return createMockModel(className, options);
    },
  };

  // Set the constructor name to simulate real LangChain classes
  Object.defineProperty(model.constructor, 'name', { value: className });
  // Also set via a custom class approach
  const ctor = { name: className };
  Object.setPrototypeOf(model, { constructor: ctor });

  return model;
}

describe('tracedModel', () => {
  let experimentId: string;
  let client: mlflow.MlflowClient;

  beforeAll(async () => {
    const authProvider = createAuthProvider({ trackingUri: TEST_TRACKING_URI });
    client = new mlflow.MlflowClient({ trackingUri: TEST_TRACKING_URI, authProvider });

    const experimentName = `langchain-test-${Date.now()}-${Math.random().toString(36).slice(2)}`;
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

  it('should return non-objects unchanged', () => {
    expect(tracedModel(null)).toBe(null);
    expect(tracedModel(undefined)).toBe(undefined);
    expect(tracedModel(42 as any)).toBe(42);
  });

  it('should trace invoke() calls', async () => {
    const mockModel = createMockModel('ChatAnthropic');
    const traced = tracedModel(mockModel);

    const messages = [createMockHumanMessage('Hello Claude!')];
    const result = await traced.invoke(messages);

    expect(result.content).toBe('Hello!');

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('OK');

    const span = trace.data.spans[0];
    expect(span.name).toBe('ChatModel');
    expect(span.spanType).toBe(mlflow.SpanType.LLM);
    expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);

    // Verify inputs are Anthropic-formatted {messages: [...]}
    expect(span.inputs).toBeDefined();
    expect(span.inputs).toHaveProperty('messages');
    expect(span.inputs.messages[0]).toHaveProperty('role', 'user');
    expect(span.inputs.messages[0]).toHaveProperty('content', 'Hello Claude!');

    // Verify outputs are Anthropic-formatted {type: "message", role: "assistant", content: [...]}
    expect(span.outputs).toBeDefined();
    expect(span.outputs).toHaveProperty('type', 'message');
    expect(span.outputs).toHaveProperty('role', 'assistant');
    expect(span.outputs.content[0]).toHaveProperty('type', 'text');
    expect(span.outputs.content[0]).toHaveProperty('text', 'Hello!');

    // Verify token usage
    const tokenUsage = span.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE];
    expect(tokenUsage).toBeDefined();
    expect(tokenUsage[mlflow.TokenUsageKey.INPUT_TOKENS]).toBe(10);
    expect(tokenUsage[mlflow.TokenUsageKey.OUTPUT_TOKENS]).toBe(20);
    expect(tokenUsage[mlflow.TokenUsageKey.TOTAL_TOKENS]).toBe(30);

    // Verify message format
    const messageFormat = span.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
    expect(messageFormat).toBe('anthropic');
  });

  it('should trace invoke() with tool calls', async () => {
    const mockModel = createMockModel('ChatOpenAI', {
      invokeResult: createMockAIMessage('', {
        tool_calls: [{ id: 'tc1', name: 'get_weather', args: { city: 'SF' } }],
        usage_metadata: { input_tokens: 15, output_tokens: 25 },
      }),
    });
    const traced = tracedModel(mockModel);

    const result = await traced.invoke([createMockHumanMessage('What is the weather?')]);

    expect(result.tool_calls).toHaveLength(1);

    const trace = await getLastActiveTrace();
    const span = trace.data.spans[0];

    // OpenAI format: tool_calls are inside choices[0].message
    expect(span.outputs).toHaveProperty('choices');
    expect(span.outputs.choices[0].message).toHaveProperty('tool_calls');
    expect(span.outputs.choices[0].message.tool_calls).toHaveLength(1);

    const messageFormat = span.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
    expect(messageFormat).toBe('openai');
  });

  it('should handle invoke() errors', async () => {
    const mockModel = createMockModel('ChatAnthropic', {
      invokeError: new Error('Rate limit exceeded'),
    });
    const traced = tracedModel(mockModel);

    await expect(traced.invoke([createMockHumanMessage('Hello')])).rejects.toThrow(
      'Rate limit exceeded',
    );

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('ERROR');

    const span = trace.data.spans[0];
    expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
  });

  it('should trace stream() with async iteration', async () => {
    const mockModel = createMockModel('ChatOpenAI');
    const traced = tracedModel(mockModel);

    const stream = await traced.stream([createMockHumanMessage('Hello streaming!')]);
    const chunks: any[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    expect(chunks).toHaveLength(2);

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('OK');

    const span = trace.data.spans[0];
    expect(span.name).toBe('ChatModel');
    expect(span.spanType).toBe(mlflow.SpanType.LLM);
    expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);

    // Verify outputs are OpenAI-formatted {choices: [{message: {content, role}}]}
    expect(span.outputs).toBeDefined();
    expect(span.outputs.choices[0].message).toHaveProperty('content', 'Hello!');
    expect(span.outputs.choices[0].message).toHaveProperty('role', 'assistant');

    // Verify token usage from aggregated result
    const tokenUsage = span.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE];
    expect(tokenUsage).toBeDefined();
    expect(tokenUsage[mlflow.TokenUsageKey.INPUT_TOKENS]).toBe(10);
    expect(tokenUsage[mlflow.TokenUsageKey.OUTPUT_TOKENS]).toBe(20);

    const messageFormat = span.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
    expect(messageFormat).toBe('openai');
  });

  it('should handle stream() errors', async () => {
    const errorChunks = [createMockAIMessage('Hel')];
    const mockModel = createMockModel('ChatAnthropic', {
      streamChunks: errorChunks,
    });

    // Override stream to produce an error during iteration
    mockModel.stream = async () => ({
      [Symbol.asyncIterator]: async function* () {
        yield createMockAIMessage('Hel');
        throw new Error('Stream interrupted');
      },
    });

    const traced = tracedModel(mockModel);
    const stream = await traced.stream([createMockHumanMessage('Hello')]);

    await expect(async () => {
      for await (const _event of stream) {
        // consume
      }
    }).rejects.toThrow('Stream interrupted');

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('ERROR');

    const span = trace.data.spans[0];
    expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
  });

  it('should trace invoke() wrapped in a parent span', async () => {
    const mockModel = createMockModel('ChatAnthropic');
    const traced = tracedModel(mockModel);

    const result = await mlflow.withSpan(
      async (_span) => {
        return traced.invoke([createMockHumanMessage('Hello from parent span.')]);
      },
      {
        name: 'predict',
        spanType: mlflow.SpanType.CHAIN,
        inputs: 'Hello from parent span.',
      },
    );

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('OK');
    expect(trace.data.spans.length).toBe(2);

    const parentSpan = trace.data.spans[0];
    expect(parentSpan.name).toBe('predict');
    expect(parentSpan.spanType).toBe(mlflow.SpanType.CHAIN);

    const childSpan = trace.data.spans[1];
    expect(childSpan.name).toBe('ChatModel');
    expect(childSpan.spanType).toBe(mlflow.SpanType.LLM);

    const messageFormat = childSpan.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
    expect(messageFormat).toBe('anthropic');
  });

  it('should return traced model from bindTools()', async () => {
    const mockModel = createMockModel('ChatOpenAI');
    const traced = tracedModel(mockModel);

    const withTools = traced.bindTools([{ name: 'test', description: 'test tool' }]);

    // The bound model should still be traced
    const result = await withTools.invoke([createMockHumanMessage('Use the tool')]);
    expect(result.content).toBe('Hello!');

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('OK');

    const span = trace.data.spans[0];
    expect(span.name).toBe('ChatModel');
    expect(span.spanType).toBe(mlflow.SpanType.LLM);
  });

  it('should detect message format for ChatXAI', async () => {
    const mockModel = createMockModel('ChatXAI');
    const traced = tracedModel(mockModel);

    await traced.invoke([createMockHumanMessage('Hello Grok!')]);

    const trace = await getLastActiveTrace();
    const span = trace.data.spans[0];

    const messageFormat = span.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
    expect(messageFormat).toBe('openai');
  });

  it('should use fallback format for unknown model classes', async () => {
    const mockModel = createMockModel('ChatCustomModel');
    const traced = tracedModel(mockModel);

    await traced.invoke([createMockHumanMessage('Hello custom!')]);

    const trace = await getLastActiveTrace();
    const span = trace.data.spans[0];

    const messageFormat = span.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
    expect(messageFormat).toBe('langchain');
  });
});
