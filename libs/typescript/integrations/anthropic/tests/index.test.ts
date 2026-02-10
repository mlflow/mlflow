/**
 * Tests for MLflow Anthropic integration with MSW mock server
 */

import * as mlflow from 'mlflow-tracing';
import Anthropic from '@anthropic-ai/sdk';
import { tracedAnthropic } from '../src';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { anthropicMockHandlers, createStreamingErrorHandler } from './mockAnthropicServer';
import { createAuthProvider } from 'mlflow-tracing/src/auth';

const TEST_TRACKING_URI = 'http://localhost:5000';

describe('tracedAnthropic', () => {
  let experimentId: string;
  let client: mlflow.MlflowClient;
  let server: ReturnType<typeof setupServer>;

  beforeAll(async () => {
    server = setupServer(...anthropicMockHandlers);
    server.listen();

    const authProvider = createAuthProvider({ trackingUri: TEST_TRACKING_URI });
    client = new mlflow.MlflowClient({ trackingUri: TEST_TRACKING_URI, authProvider });

    const experimentName = `anthropic-test-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    experimentId = await client.createExperiment(experimentName);

    mlflow.init({
      trackingUri: TEST_TRACKING_URI,
      experimentId,
    });
  });

  afterAll(async () => {
    server.close();
    await client.deleteExperiment(experimentId);
  });

  const getLastActiveTrace = async (): Promise<mlflow.Trace> => {
    await mlflow.flushTraces();
    const traceId = mlflow.getLastActiveTraceId();
    const trace = await client.getTrace(traceId!);
    return trace;
  };

  beforeEach(() => {
    server.resetHandlers();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should trace messages.create()', async () => {
    const anthropic = new Anthropic({ apiKey: 'test-key' });
    const wrappedAnthropic = tracedAnthropic(anthropic);

    const result = await wrappedAnthropic.messages.create({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 256,
      messages: [{ role: 'user', content: 'Hello Claude!' }],
    });

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('OK');

    const tokenUsage = trace.info.tokenUsage;
    expect(tokenUsage).toBeDefined();
    expect(typeof tokenUsage?.input_tokens).toBe('number');
    expect(typeof tokenUsage?.output_tokens).toBe('number');
    expect(typeof tokenUsage?.total_tokens).toBe('number');

    const span = trace.data.spans[0];
    expect(span.name).toBe('Messages');
    expect(span.spanType).toBe(mlflow.SpanType.LLM);
    expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
    expect(span.inputs).toEqual({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 256,
      messages: [{ role: 'user', content: 'Hello Claude!' }],
    });
    expect(span.outputs).toEqual(result);

    const spanTokenUsage = span.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE];
    expect(spanTokenUsage).toBeDefined();
    expect(typeof spanTokenUsage[mlflow.TokenUsageKey.INPUT_TOKENS]).toBe('number');
    expect(typeof spanTokenUsage[mlflow.TokenUsageKey.OUTPUT_TOKENS]).toBe('number');
    expect(typeof spanTokenUsage[mlflow.TokenUsageKey.TOTAL_TOKENS]).toBe('number');

    const messageFormat = span.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
    expect(messageFormat).toBe('anthropic');
  });

  it('should handle Anthropic errors properly', async () => {
    server.use(
      http.post('https://api.anthropic.com/v1/messages', () =>
        HttpResponse.json(
          {
            error: {
              type: 'rate_limit',
              message: 'Rate limit exceeded',
            },
          },
          { status: 429 },
        ),
      ),
    );

    // Disable retries to prevent the SDK from retrying on errors.
    // SDK 0.50+ calls CancelReadableStream on error responses before retrying,
    // which hangs indefinitely with MSW mock responses.
    const anthropic = new Anthropic({ apiKey: 'test-key', maxRetries: 0 });
    const wrappedAnthropic = tracedAnthropic(anthropic);

    await expect(
      wrappedAnthropic.messages.create({
        model: 'claude-3-7-sonnet-20250219',
        max_tokens: 128,
        messages: [{ role: 'user', content: 'This request should fail.' }],
      }),
    ).rejects.toThrow();

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('ERROR');

    const span = trace.data.spans[0];
    expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
    expect(span.inputs).toEqual({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 128,
      messages: [{ role: 'user', content: 'This request should fail.' }],
    });
    expect(span.outputs).toBeUndefined();
  });

  it('should trace Anthropic request wrapped in a parent span', async () => {
    const anthropic = new Anthropic({ apiKey: 'test-key' });
    const wrappedAnthropic = tracedAnthropic(anthropic);

    const result = await mlflow.withSpan(
      async (_span) => {
        const response = await wrappedAnthropic.messages.create({
          model: 'claude-3-7-sonnet-20250219',
          max_tokens: 128,
          messages: [{ role: 'user', content: 'Hello from parent span.' }],
        });
        return response.content[0];
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
    expect(parentSpan.outputs).toEqual(result);

    const childSpan = trace.data.spans[1];
    expect(childSpan.name).toBe('Messages');
    expect(childSpan.spanType).toBe(mlflow.SpanType.LLM);
    expect(childSpan.inputs).toEqual({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 128,
      messages: [{ role: 'user', content: 'Hello from parent span.' }],
    });
    expect(childSpan.outputs).toBeDefined();

    const messageFormat = childSpan.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
    expect(messageFormat).toBe('anthropic');
  });

  it('should trace messages.stream() with async iteration', async () => {
    const anthropic = new Anthropic({ apiKey: 'test-key' });
    const wrappedAnthropic = tracedAnthropic(anthropic);

    const events: any[] = [];
    const stream = wrappedAnthropic.messages.stream({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 256,
      messages: [{ role: 'user', content: 'Hello streaming!' }],
    });

    for await (const event of stream) {
      events.push(event);
    }

    expect(events.length).toBeGreaterThan(0);

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('OK');

    const span = trace.data.spans[0];
    expect(span.name).toBe('Messages');
    expect(span.spanType).toBe(mlflow.SpanType.LLM);
    expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
    expect(span.inputs).toMatchObject({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 256,
      messages: [{ role: 'user', content: 'Hello streaming!' }],
    });

    // Verify outputs contain actual message content, not a stream object
    expect(span.outputs).toBeDefined();
    expect(span.outputs).toHaveProperty('id');
    expect(span.outputs).toHaveProperty('type', 'message');
    expect(span.outputs).toHaveProperty('content');
    expect(span.outputs.content[0]).toHaveProperty('type', 'text');

    // Verify token usage is captured for streaming
    const spanTokenUsage = span.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE];
    expect(spanTokenUsage).toBeDefined();
    expect(typeof spanTokenUsage[mlflow.TokenUsageKey.INPUT_TOKENS]).toBe('number');
    expect(typeof spanTokenUsage[mlflow.TokenUsageKey.OUTPUT_TOKENS]).toBe('number');

    const messageFormat = span.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
    expect(messageFormat).toBe('anthropic');
  });

  it('should trace messages.stream() with finalMessage()', async () => {
    const anthropic = new Anthropic({ apiKey: 'test-key' });
    const wrappedAnthropic = tracedAnthropic(anthropic);

    const stream = wrappedAnthropic.messages.stream({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 256,
      messages: [{ role: 'user', content: 'Hello finalMessage!' }],
    });

    const finalMessage = await stream.finalMessage();

    expect(finalMessage).toBeDefined();
    expect(finalMessage.content).toBeDefined();

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('OK');

    const span = trace.data.spans[0];
    expect(span.name).toBe('Messages');
    expect(span.spanType).toBe(mlflow.SpanType.LLM);
    expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
    expect(span.inputs).toEqual({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 256,
      messages: [{ role: 'user', content: 'Hello finalMessage!' }],
    });
    expect(span.outputs).toEqual(finalMessage);

    const messageFormat = span.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
    expect(messageFormat).toBe('anthropic');
  });

  it('should handle errors during streaming async iteration', async () => {
    server.use(createStreamingErrorHandler());

    const anthropic = new Anthropic({ apiKey: 'test-key', maxRetries: 0 });
    const wrappedAnthropic = tracedAnthropic(anthropic);

    const stream = wrappedAnthropic.messages.stream({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 256,
      messages: [{ role: 'user', content: 'Hello streaming error!' }],
    });

    await expect(async () => {
      for await (const _event of stream) {
        // consume events until error
      }
    }).rejects.toThrow();

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('ERROR');

    const span = trace.data.spans[0];
    expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
    expect(span.inputs).toMatchObject({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 256,
      messages: [{ role: 'user', content: 'Hello streaming error!' }],
    });
  });

  it('should handle errors during streaming finalMessage()', async () => {
    server.use(createStreamingErrorHandler());

    const anthropic = new Anthropic({ apiKey: 'test-key', maxRetries: 0 });
    const wrappedAnthropic = tracedAnthropic(anthropic);

    const stream = wrappedAnthropic.messages.stream({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 256,
      messages: [{ role: 'user', content: 'Hello finalMessage error!' }],
    });

    await expect(stream.finalMessage()).rejects.toThrow();

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('ERROR');

    const span = trace.data.spans[0];
    expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
    expect(span.inputs).toEqual({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 256,
      messages: [{ role: 'user', content: 'Hello finalMessage error!' }],
    });
  });

  it('should create only one span when async iteration is followed by finalMessage()', async () => {
    const anthropic = new Anthropic({ apiKey: 'test-key' });
    const wrappedAnthropic = tracedAnthropic(anthropic);

    const stream = wrappedAnthropic.messages.stream({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 256,
      messages: [{ role: 'user', content: 'Hello dedup!' }],
    });

    // First consume via async iteration (claims tracing)
    for await (const _event of stream) {
      // consume all events
    }

    // Then call finalMessage() on the same stream — should NOT create a second span
    const finalMessage = await stream.finalMessage();
    expect(finalMessage).toBeDefined();

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('OK');

    // Only one LLM span should exist (no duplicate from finalMessage)
    const llmSpans = trace.data.spans.filter((s) => s.spanType === mlflow.SpanType.LLM);
    expect(llmSpans.length).toBe(1);
  });

  it('should not trace messages.create() when stream: true is passed directly', async () => {
    const anthropic = new Anthropic({ apiKey: 'test-key' });
    const wrappedAnthropic = tracedAnthropic(anthropic);

    // Calling create() with stream: true is skipped by the tracing wrapper
    // because the SDK's stream() method internally calls create({ stream: true }),
    // and tracing is handled by the stream wrapper instead.
    // Users should use messages.stream() for traced streaming.
    const response = await wrappedAnthropic.messages.create({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 256,
      messages: [{ role: 'user', content: 'Hello direct stream!' }],
      stream: true,
    });

    // The SDK returns a Stream object when stream: true
    expect(response).toBeDefined();

    // Consume the stream to completion
    for await (const _event of response as any) {
      // consume
    }

    // No traced span should have been created by the create() wrapper
    await mlflow.flushTraces();
    const traceId = mlflow.getLastActiveTraceId();
    // The last active trace should be from a previous test, not this call,
    // OR if it is from this call, it should not have a Messages LLM span
    // created by wrapWithTracing (since create with stream:true is skipped).
    if (traceId) {
      const trace = await client.getTrace(traceId);
      const matchingSpans = trace.data.spans.filter(
        (s) =>
          s.spanType === mlflow.SpanType.LLM &&
          s.inputs?.messages?.[0]?.content === 'Hello direct stream!',
      );
      expect(matchingSpans.length).toBe(0);
    }
  });

  it('should trace streaming request wrapped in a parent span', async () => {
    const anthropic = new Anthropic({ apiKey: 'test-key' });
    const wrappedAnthropic = tracedAnthropic(anthropic);

    const result = await mlflow.withSpan(
      async (_span) => {
        const stream = wrappedAnthropic.messages.stream({
          model: 'claude-3-7-sonnet-20250219',
          max_tokens: 128,
          messages: [{ role: 'user', content: 'Hello streaming parent!' }],
        });

        const message = await stream.finalMessage();
        return message.content[0];
      },
      {
        name: 'streaming-predict',
        spanType: mlflow.SpanType.CHAIN,
        inputs: 'Hello streaming parent!',
      },
    );

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('OK');
    // At least 2 spans: parent chain span and child LLM span
    expect(trace.data.spans.length).toBeGreaterThanOrEqual(2);

    const parentSpan = trace.data.spans[0];
    expect(parentSpan.name).toBe('streaming-predict');
    expect(parentSpan.spanType).toBe(mlflow.SpanType.CHAIN);
    expect(parentSpan.outputs).toEqual(result);

    // Find child spans with LLM type
    const llmSpans = trace.data.spans.filter((s) => s.spanType === mlflow.SpanType.LLM);
    expect(llmSpans.length).toBeGreaterThanOrEqual(1);

    const llmSpan = llmSpans[0];
    expect(llmSpan.outputs).toBeDefined();

    const messageFormat = llmSpan.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
    expect(messageFormat).toBe('anthropic');
  });
});
