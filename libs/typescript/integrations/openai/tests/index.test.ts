/**
 * Tests for MLflow OpenAI integration with MSW mock server
 */

import * as mlflow from 'mlflow-tracing';
import { tracedOpenAI } from '../src';
import { OpenAI } from 'openai';
import { http, HttpResponse } from 'msw';
import { openAIMswServer, useMockOpenAIServer } from '../../helpers/openaiTestHelper';

const TEST_TRACKING_URI = 'http://localhost:5000';

describe('tracedOpenAI', () => {
  useMockOpenAIServer();

  let experimentId: string;
  let client: mlflow.MlflowClient;

  beforeAll(async () => {
    // Setup MLflow client and experiment
    client = new mlflow.MlflowClient({ trackingUri: TEST_TRACKING_URI, host: TEST_TRACKING_URI });

    // Create a new experiment
    const experimentName = `test-experiment-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
    experimentId = await client.createExperiment(experimentName);
    mlflow.init({
      trackingUri: TEST_TRACKING_URI,
      experimentId: experimentId
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
        messages: [{ role: 'user', content: 'Hello!' }]
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
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(span.inputs).toEqual({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello!' }]
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
                message: 'Rate limit exceeded'
              }
            },
            { status: 429 }
          );
        })
      );

      const openai = new OpenAI({ apiKey: 'test-key' });
      const wrappedOpenAI = tracedOpenAI(openai);

      await expect(
        wrappedOpenAI.chat.completions.create({
          model: 'gpt-4',
          messages: [{ role: 'user', content: 'This should fail' }]
        })
      ).rejects.toThrow();

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('ERROR');

      const span = trace.data.spans[0];
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
      expect(span.inputs).toEqual({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'This should fail' }]
      });
      expect(span.outputs).toBeUndefined();
      expect(span.startTime).toBeDefined();
      expect(span.endTime).toBeDefined();
    });

    it('should trace OpenAI request wrapped in a parent span', async () => {
      const openai = new OpenAI({ apiKey: 'test-key' });
      const wrappedOpenAI = tracedOpenAI(openai);

      const result = await mlflow.withSpan(
        async (_span) => {
          const response = await wrappedOpenAI.chat.completions.create({
            model: 'gpt-4',
            messages: [{ role: 'user', content: 'Hello!' }]
          });
          return response.choices[0].message.content;
        },
        {
          name: 'predict',
          spanType: mlflow.SpanType.CHAIN,
          inputs: 'Hello!'
        }
      );

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(2);

      const parentSpan = trace.data.spans[0];
      expect(parentSpan.name).toBe('predict');
      expect(parentSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(parentSpan.spanType).toBe(mlflow.SpanType.CHAIN);
      expect(parentSpan.inputs).toEqual('Hello!');
      expect(parentSpan.outputs).toEqual(result);
      expect(parentSpan.startTime).toBeDefined();
      expect(parentSpan.endTime).toBeDefined();

      const childSpan = trace.data.spans[1];
      expect(childSpan.name).toBe('Completions');
      expect(childSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(childSpan.spanType).toBe(mlflow.SpanType.LLM);
      expect(childSpan.inputs).toEqual({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello!' }]
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
        temperature: 0
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
        temperature: 0
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
        input: ['Hello', 'world']
      });

      expect(response.object).toBe('list');
      expect(response.data.length).toBe(2);
      expect(response.data[0].object).toBe('embedding');
      expect(response.data[0].embedding.length).toBeGreaterThan(0);
      expect(response.model).toBe('text-embedding-3-small');

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(1);

      const span = trace.data.spans[0];
      expect(span.name).toBe('Embeddings');
      expect(span.spanType).toBe(mlflow.SpanType.EMBEDDING);
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(span.inputs).toEqual({
        model: 'text-embedding-3-small',
        input: ['Hello', 'world']
      });
      expect(span.outputs).toEqual(response);
      expect(span.startTime).toBeDefined();
      expect(span.endTime).toBeDefined();
    });
  });
});
