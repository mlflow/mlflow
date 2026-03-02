/**
 * Tests for MLflow Anthropic integration with MSW mock server
 */

import * as mlflow from '@mlflow/core';
import Anthropic from '@anthropic-ai/sdk';
import { tracedAnthropic } from '../src';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import {
  anthropicMockHandlers,
  createStreamingErrorHandler,
  createStreamingWithUnsupportedContentHandler,
} from './mockAnthropicServer';
import { createAuthProvider } from '@mlflow/core/src/auth';

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

  describe('content sanitization for MLflow UI', () => {
    it('should replace redacted_thinking blocks with text placeholders in inputs', async () => {
      const anthropic = new Anthropic({ apiKey: 'test-key' });
      const wrappedAnthropic = tracedAnthropic(anthropic);

      await wrappedAnthropic.messages.create({
        model: 'claude-3-7-sonnet-20250219',
        max_tokens: 256,
        messages: [
          { role: 'user', content: 'Think carefully.' },
          {
            role: 'assistant',
            content: [
              { type: 'thinking', thinking: 'Deep thought...', signature: 'sig123' },
              { type: 'redacted_thinking', data: 'base64encodeddata' },
              { type: 'text', text: 'My answer.' },
            ],
          },
          { role: 'user', content: 'Continue.' },
        ] as any,
      });

      const trace = await getLastActiveTrace();
      const span = trace.data.spans[0];

      const assistantMsg = span.inputs.messages[1];
      expect(assistantMsg.content).toEqual([
        { type: 'thinking', thinking: 'Deep thought...', signature: 'sig123' },
        { type: 'text', text: '[redacted_thinking]' },
        { type: 'text', text: 'My answer.' },
      ]);

      // Other messages should be unchanged
      expect(span.inputs.messages[0]).toEqual({ role: 'user', content: 'Think carefully.' });
      expect(span.inputs.messages[2]).toEqual({ role: 'user', content: 'Continue.' });
    });

    it('should replace document blocks with text placeholders in inputs', async () => {
      const anthropic = new Anthropic({ apiKey: 'test-key' });
      const wrappedAnthropic = tracedAnthropic(anthropic);

      await wrappedAnthropic.messages.create({
        model: 'claude-3-7-sonnet-20250219',
        max_tokens: 256,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'document',
                source: { type: 'base64', media_type: 'application/pdf', data: 'base64pdfdata' },
              },
              { type: 'text', text: 'Summarize this document.' },
            ],
          },
        ] as any,
      });

      const trace = await getLastActiveTrace();
      const span = trace.data.spans[0];

      const userMsg = span.inputs.messages[0];
      expect(userMsg.content).toEqual([
        { type: 'text', text: '[document]' },
        { type: 'text', text: 'Summarize this document.' },
      ]);
    });

    it('should preserve all supported content types in inputs unchanged', async () => {
      const anthropic = new Anthropic({ apiKey: 'test-key' });
      const wrappedAnthropic = tracedAnthropic(anthropic);

      const messages = [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Hello' },
            {
              type: 'image',
              source: { type: 'base64', media_type: 'image/png', data: 'imagedata' },
            },
          ],
        },
        {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'Thinking...' },
            { type: 'text', text: 'Response' },
            { type: 'tool_use', id: 'tool_1', name: 'search', input: { query: 'test' } },
          ],
        },
        {
          role: 'user',
          content: [{ type: 'tool_result', tool_use_id: 'tool_1', content: 'Result' }],
        },
      ];

      await wrappedAnthropic.messages.create({
        model: 'claude-3-7-sonnet-20250219',
        max_tokens: 256,
        messages: messages as any,
      });

      const trace = await getLastActiveTrace();
      const span = trace.data.spans[0];

      // All content should be preserved as-is since all types are supported
      expect(span.inputs.messages).toEqual(messages);
    });

    it('should pass through string message content without modification', async () => {
      const anthropic = new Anthropic({ apiKey: 'test-key' });
      const wrappedAnthropic = tracedAnthropic(anthropic);

      await wrappedAnthropic.messages.create({
        model: 'claude-3-7-sonnet-20250219',
        max_tokens: 256,
        messages: [{ role: 'user', content: 'Just a plain string message.' }],
      });

      const trace = await getLastActiveTrace();
      const span = trace.data.spans[0];

      expect(span.inputs.messages[0]).toEqual({
        role: 'user',
        content: 'Just a plain string message.',
      });
    });

    it('should replace unsupported content blocks in outputs', async () => {
      server.use(
        http.post('https://api.anthropic.com/v1/messages', async ({ request }) => {
          const body = (await request.json()) as any;
          return HttpResponse.json({
            id: 'msg_test_sanitize_output',
            type: 'message',
            role: 'assistant',
            // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
            model: body.model,
            stop_reason: 'end_turn',
            stop_sequence: null,
            usage: { input_tokens: 100, output_tokens: 200, total_tokens: 300 },
            content: [
              { type: 'thinking', thinking: 'Let me think about this...' },
              { type: 'redacted_thinking', data: 'base64signaturedata' },
              { type: 'text', text: 'Here is my response.' },
            ],
          });
        }),
      );

      const anthropic = new Anthropic({ apiKey: 'test-key' });
      const wrappedAnthropic = tracedAnthropic(anthropic);

      await wrappedAnthropic.messages.create({
        model: 'claude-3-7-sonnet-20250219',
        max_tokens: 256,
        messages: [{ role: 'user', content: 'Test output sanitization.' }],
      });

      const trace = await getLastActiveTrace();
      const span = trace.data.spans[0];

      expect(span.outputs.content).toEqual([
        { type: 'thinking', thinking: 'Let me think about this...' },
        { type: 'text', text: '[redacted_thinking]' },
        { type: 'text', text: 'Here is my response.' },
      ]);
    });

    it('should sanitize outputs when streaming via async iteration', async () => {
      server.use(createStreamingWithUnsupportedContentHandler());

      const anthropic = new Anthropic({ apiKey: 'test-key' });
      const wrappedAnthropic = tracedAnthropic(anthropic);

      const stream = wrappedAnthropic.messages.stream({
        model: 'claude-3-7-sonnet-20250219',
        max_tokens: 256,
        messages: [{ role: 'user', content: 'Test streaming output sanitization.' }],
      });

      for await (const _event of stream) {
        // consume all events
      }

      const trace = await getLastActiveTrace();
      const span = trace.data.spans[0];

      // The streamed final message contained a redacted_thinking block;
      // outputs stored on the span should have it replaced with a text placeholder
      expect(span.outputs.content).toEqual([
        { type: 'thinking', thinking: 'Let me think...' },
        { type: 'text', text: '[redacted_thinking]' },
        { type: 'text', text: 'Here is my response.' },
      ]);
    });

    it('should sanitize outputs when streaming via finalMessage()', async () => {
      server.use(createStreamingWithUnsupportedContentHandler());

      const anthropic = new Anthropic({ apiKey: 'test-key' });
      const wrappedAnthropic = tracedAnthropic(anthropic);

      const stream = wrappedAnthropic.messages.stream({
        model: 'claude-3-7-sonnet-20250219',
        max_tokens: 256,
        messages: [{ role: 'user', content: 'Test streaming finalMessage output sanitization.' }],
      });

      const message = await stream.finalMessage();
      expect(message).toBeDefined();

      const trace = await getLastActiveTrace();
      const span = trace.data.spans[0];

      // Outputs stored on the span should have redacted_thinking replaced
      expect(span.outputs.content).toEqual([
        { type: 'thinking', thinking: 'Let me think...' },
        { type: 'text', text: '[redacted_thinking]' },
        { type: 'text', text: 'Here is my response.' },
      ]);
    });

    it('should sanitize inputs when streaming', async () => {
      const anthropic = new Anthropic({ apiKey: 'test-key' });
      const wrappedAnthropic = tracedAnthropic(anthropic);

      const stream = wrappedAnthropic.messages.stream({
        model: 'claude-3-7-sonnet-20250219',
        max_tokens: 256,
        messages: [
          { role: 'user', content: 'Think about this.' },
          {
            role: 'assistant',
            content: [
              { type: 'thinking', thinking: 'Hmm...' },
              { type: 'redacted_thinking', data: 'redacteddata' },
              { type: 'text', text: 'Previous answer.' },
            ],
          },
          { role: 'user', content: 'Now continue.' },
        ] as any,
      });

      const message = await stream.finalMessage();
      expect(message).toBeDefined();

      const trace = await getLastActiveTrace();
      const span = trace.data.spans[0];

      // Inputs should have redacted_thinking replaced
      const assistantMsg = span.inputs.messages[1];
      expect(assistantMsg.content).toEqual([
        { type: 'thinking', thinking: 'Hmm...' },
        { type: 'text', text: '[redacted_thinking]' },
        { type: 'text', text: 'Previous answer.' },
      ]);
    });
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
