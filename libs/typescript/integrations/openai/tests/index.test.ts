/**
 * Tests for MLflow OpenAI integration with MSW mock server
 */

import * as mlflow from '@mlflow/core';
import { tracedOpenAI } from '../src';
import { OpenAI } from 'openai';
import { Stream } from 'openai/streaming';
import { http, HttpResponse } from 'msw';
import { openAIMswServer, useMockOpenAIServer } from '../../helpers/openaiTestHelper';
import { createAuthProvider } from '@mlflow/core/src/auth';

const TEST_TRACKING_URI = 'http://localhost:5000';

describe('tracedOpenAI', () => {
  useMockOpenAIServer();

  let experimentId: string;
  let client: mlflow.MlflowClient;

  beforeAll(async () => {
    // Setup MLflow client and experiment
    const authProvider = createAuthProvider({ trackingUri: TEST_TRACKING_URI });
    client = new mlflow.MlflowClient({ trackingUri: TEST_TRACKING_URI, authProvider });

    // Create a new experiment
    const experimentName = `test-experiment-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
    experimentId = await client.createExperiment(experimentName);
    mlflow.init({
      trackingUri: TEST_TRACKING_URI,
      experimentId: experimentId,
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

  describe('Chat Completions', () => {
    it('should trace chat.completions.create()', async () => {
      const openai = new OpenAI({ apiKey: 'test-key' });
      const wrappedOpenAI = tracedOpenAI(openai);

      const result = await wrappedOpenAI.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello!' }],
      });

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');

      const tokenUsage = trace.info.tokenUsage;
      expect(tokenUsage).toBeDefined();
      expect(typeof tokenUsage?.input_tokens).toBe('number');
      expect(typeof tokenUsage?.output_tokens).toBe('number');
      expect(typeof tokenUsage?.total_tokens).toBe('number');

      const span = trace.data.spans[0];
      expect(span.name).toBe('Completions');
      expect(span.spanType).toBe(mlflow.SpanType.LLM);
      expect(span.logLevel).toBe(mlflow.SpanLogLevel.INFO);
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(span.inputs).toEqual({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello!' }],
      });
      expect(span.outputs).toEqual(result);
      expect(span.startTime).toBeDefined();
      expect(span.endTime).toBeDefined();

      // Check that token usage is stored at span level
      const spanTokenUsage = span.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE];
      expect(spanTokenUsage).toBeDefined();
      expect(typeof spanTokenUsage[mlflow.TokenUsageKey.INPUT_TOKENS]).toBe('number');
      expect(typeof spanTokenUsage[mlflow.TokenUsageKey.OUTPUT_TOKENS]).toBe('number');
      expect(typeof spanTokenUsage[mlflow.TokenUsageKey.TOTAL_TOKENS]).toBe('number');
    });

    it('should handle chat completion errors properly', async () => {
      // Configure MSW to return rate limit error
      openAIMswServer.use(
        http.post('https://api.openai.com/v1/chat/completions', () => {
          return HttpResponse.json(
            {
              error: {
                type: 'requests',
                message: 'Rate limit exceeded',
              },
            },
            { status: 429 },
          );
        }),
      );

      const openai = new OpenAI({ apiKey: 'test-key' });
      const wrappedOpenAI = tracedOpenAI(openai);

      await expect(
        wrappedOpenAI.chat.completions.create({
          model: 'gpt-4',
          messages: [{ role: 'user', content: 'This should fail' }],
        }),
      ).rejects.toThrow();

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('ERROR');

      const span = trace.data.spans[0];
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
      expect(span.inputs).toEqual({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'This should fail' }],
      });
      expect(span.outputs).toBeUndefined();
      expect(span.startTime).toBeDefined();
      expect(span.endTime).toBeDefined();
    });

    it('should keep spans open while streaming chat completions', async () => {
      const openai = new OpenAI({ apiKey: 'test-key' });
      const wrappedOpenAI = tracedOpenAI(openai);

      const stream = await wrappedOpenAI.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello!' }],
        stream: true,
        stream_options: { include_usage: true },
      });

      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      const trace = await getLastActiveTrace();
      const span = trace.data.spans[0];
      expect(chunks).toHaveLength(4);
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(span.outputs).toEqual({
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'Test response',
              tool_calls: [
                {
                  id: 'call_123',
                  type: 'function',
                  function: { name: 'get_weather', arguments: '{"city":"Paris"}' },
                },
              ],
            },
            finish_reason: 'tool_calls',
          },
        ],
      });
      expect(span.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE]).toEqual({
        [mlflow.TokenUsageKey.INPUT_TOKENS]: 10,
        [mlflow.TokenUsageKey.OUTPUT_TOKENS]: 20,
        [mlflow.TokenUsageKey.TOTAL_TOKENS]: 30,
      });
    });

    it('should end spans when streaming chat completions are terminated early', async () => {
      const openai = new OpenAI({ apiKey: 'test-key' });
      const wrappedOpenAI = tracedOpenAI(openai);

      const stream = await wrappedOpenAI.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello!' }],
        stream: true,
      });

      for await (const _chunk of stream) {
        break;
      }

      const trace = await getLastActiveTrace();
      const span = trace.data.spans[0];
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(span.outputs).toEqual({
        choices: [{ index: 0, message: { role: 'assistant', content: 'Test ' } }],
      });
      expect(span.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE]).toBeUndefined();
    });

    it('should end spans when a stream is consumed through tee()', async () => {
      const wrappedOpenAI = tracedOpenAI(new OpenAI({ apiKey: 'test-key' }));

      const stream = await wrappedOpenAI.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello tee!' }],
        stream: true,
        stream_options: { include_usage: true },
      });

      const [left] = stream.tee();
      const chunks = [];
      for await (const chunk of left) {
        chunks.push(chunk);
      }

      const trace = await getLastActiveTrace();
      const span = trace.data.spans[0];
      expect(span.inputs.messages).toEqual([{ role: 'user', content: 'Hello tee!' }]);
      expect(chunks).toHaveLength(4);
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(span.outputs).toBeDefined();
      expect(span.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE]).toEqual({
        [mlflow.TokenUsageKey.INPUT_TOKENS]: 10,
        [mlflow.TokenUsageKey.OUTPUT_TOKENS]: 20,
        [mlflow.TokenUsageKey.TOTAL_TOKENS]: 30,
      });
    });

    it('should end spans when a stream is consumed through toReadableStream()', async () => {
      const wrappedOpenAI = tracedOpenAI(new OpenAI({ apiKey: 'test-key' }));

      const stream = await wrappedOpenAI.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello readable!' }],
        stream: true,
      });

      const reader = stream.toReadableStream().getReader();
      while (!(await reader.read()).done) {
        // Drain the stream
      }

      const trace = await getLastActiveTrace();
      const span = trace.data.spans[0];
      expect(span.inputs.messages).toEqual([{ role: 'user', content: 'Hello readable!' }]);
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(span.outputs).toBeDefined();
    });

    it('should end spans when a stream is abandoned without being consumed', async () => {
      const wrappedOpenAI = tracedOpenAI(new OpenAI({ apiKey: 'test-key' }));

      const stream = await wrappedOpenAI.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello abandoned!' }],
        stream: true,
      });

      stream.controller.abort();

      const trace = await getLastActiveTrace();
      const span = trace.data.spans[0];
      expect(span.inputs.messages).toEqual([{ role: 'user', content: 'Hello abandoned!' }]);
      expect(span.endTime).toBeDefined();
      expect(span.outputs).toEqual({ choices: [] });
    });

    it('should preserve the identity of the returned stream', async () => {
      const wrappedOpenAI = tracedOpenAI(new OpenAI({ apiKey: 'test-key' }));

      const stream = await wrappedOpenAI.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello!' }],
        stream: true,
      });

      expect(stream).toBeInstanceOf(Stream);
      expect(stream.constructor).toBe(Stream);

      stream.controller.abort();
    });

    it('should mark spans as errors when streaming chat completions fail', async () => {
      class Completions {
        create(_params: unknown) {
          return Promise.resolve({
            async *[Symbol.asyncIterator]() {
              yield await Promise.resolve({
                choices: [{ index: 0, delta: { content: 'Test ' }, finish_reason: null }],
              });
              throw new Error('Stream failed');
            },
          });
        }
      }

      const wrappedOpenAI = tracedOpenAI({ chat: { completions: new Completions() } });
      const stream = await wrappedOpenAI.chat.completions.create({ stream: true });

      await expect(
        (async () => {
          for await (const _chunk of stream) {
            expect(_chunk).toBeDefined();
          }
        })(),
      ).rejects.toThrow('Stream failed');

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('ERROR');
      expect(trace.data.spans[0].status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
    });

    it('should mark spans as errors when creating a stream throws synchronously', async () => {
      class Completions {
        create(_params: unknown) {
          throw new Error('Stream creation failed');
        }
      }

      const wrappedOpenAI = tracedOpenAI({ chat: { completions: new Completions() } });

      await expect(wrappedOpenAI.chat.completions.create({ stream: true })).rejects.toThrow(
        'Stream creation failed',
      );

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('ERROR');
      expect(trace.data.spans[0].status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
    });

    it('should trace OpenAI request wrapped in a parent span', async () => {
      const openai = new OpenAI({ apiKey: 'test-key' });
      const wrappedOpenAI = tracedOpenAI(openai);

      const result = await mlflow.withSpan(
        async (_span) => {
          const response = await wrappedOpenAI.chat.completions.create({
            model: 'gpt-4',
            messages: [{ role: 'user', content: 'Hello!' }],
          });
          return response.choices[0].message.content;
        },
        {
          name: 'predict',
          spanType: mlflow.SpanType.CHAIN,
          inputs: 'Hello!',
        },
      );

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(2);

      const parentSpan = trace.data.spans[0];
      expect(parentSpan.name).toBe('predict');
      expect(parentSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(parentSpan.spanType).toBe(mlflow.SpanType.CHAIN);
      // CHAIN spans default to DEBUG; LLM children to INFO (asserted below).
      expect(parentSpan.logLevel).toBe(mlflow.SpanLogLevel.DEBUG);
      expect(parentSpan.inputs).toEqual('Hello!');
      expect(parentSpan.outputs).toEqual(result);
      expect(parentSpan.startTime).toBeDefined();
      expect(parentSpan.endTime).toBeDefined();

      const childSpan = trace.data.spans[1];
      expect(childSpan.name).toBe('Completions');
      expect(childSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(childSpan.spanType).toBe(mlflow.SpanType.LLM);
      expect(childSpan.logLevel).toBe(mlflow.SpanLogLevel.INFO);
      expect(childSpan.inputs).toEqual({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello!' }],
      });
      expect(childSpan.outputs).toBeDefined();
      expect(childSpan.startTime).toBeDefined();
      expect(childSpan.endTime).toBeDefined();
    });
  });

  describe('Responses API', () => {
    it('should trace responses.create()', async () => {
      const openai = new OpenAI({ apiKey: 'test-key' });
      const wrappedOpenAI = tracedOpenAI(openai);

      const response = await wrappedOpenAI.responses.create({
        input: 'Hello!',
        model: 'gpt-4o',
        temperature: 0,
      });

      // Verify response
      expect((response as any).id).toBe('responses-123');

      // Get and verify the trace
      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.info.tokenUsage?.input_tokens).toBe(response.usage?.input_tokens);
      expect(trace.info.tokenUsage?.output_tokens).toBe(response.usage?.output_tokens);
      expect(trace.info.tokenUsage?.total_tokens).toBe(response.usage?.total_tokens);
      expect(trace.data.spans.length).toBe(1);

      const span = trace.data.spans[0];
      expect(span.spanType).toBe(mlflow.SpanType.LLM);
      expect(span.inputs).toEqual({
        input: 'Hello!',
        model: 'gpt-4o',
        temperature: 0,
      });
      expect(span.outputs).toEqual(response);
    });
  });

  describe('Embeddings API', () => {
    it('should trace embeddings.create() with input: %p', async () => {
      const openai = new OpenAI({ apiKey: 'test-key' });
      const wrappedOpenAI = tracedOpenAI(openai);

      const response = await wrappedOpenAI.embeddings.create({
        model: 'text-embedding-3-small',
        input: ['Hello', 'world'],
      });

      expect(response.object).toBe('list');
      expect(response.data.length).toBe(2);
      expect(response.data[0].object).toBe('embedding');
      expect(response.data[0].embedding.length).toBeGreaterThan(0);
      expect(response.model).toBe('text-embedding-3-small');

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(1);

      const tokenUsage = trace.info.tokenUsage;
      expect(tokenUsage).toBeDefined();
      expect(tokenUsage?.input_tokens).toBe(response.usage.prompt_tokens);
      expect(tokenUsage?.output_tokens).toBe(0);
      expect(tokenUsage?.total_tokens).toBe(response.usage.total_tokens);

      const span = trace.data.spans[0];
      expect(span.name).toBe('Embeddings');
      expect(span.spanType).toBe(mlflow.SpanType.EMBEDDING);
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(span.inputs).toEqual({
        model: 'text-embedding-3-small',
        input: ['Hello', 'world'],
      });
      expect(span.outputs).toEqual(response);
      expect(span.startTime).toBeDefined();
      expect(span.endTime).toBeDefined();

      const spanTokenUsage = span.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE];
      expect(spanTokenUsage).toBeDefined();
      expect(spanTokenUsage[mlflow.TokenUsageKey.INPUT_TOKENS]).toBe(response.usage.prompt_tokens);
      expect(spanTokenUsage[mlflow.TokenUsageKey.OUTPUT_TOKENS]).toBe(0);
      expect(spanTokenUsage[mlflow.TokenUsageKey.TOTAL_TOKENS]).toBe(response.usage.total_tokens);
    });
  });
});
