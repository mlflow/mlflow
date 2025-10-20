/**
 * Tests for MLflow OpenAI integration with MSW mock server
 */

import * as mlflow from 'mlflow-tracing';
import { generateText, streamText } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { http, HttpResponse } from 'msw';
import { openAIMswServer, useMockOpenAIServer } from '../../../tests/helpers/mockOpenAIServer';

const TEST_TRACKING_URI = 'http://localhost:5000';

describe('vercel autologging integration', () => {
  useMockOpenAIServer();

  let experimentId: string;
  let client: mlflow.MlflowClient;
  let openai: any;

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

    // Create OpenAI client
    openai = createOpenAI({ apiKey: 'test-key' });
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

  describe('generateText', () => {
    it('should trace ai.generatetext()', async () => {
      const result = await generateText({
        model: openai('gpt-4o-mini'),
        prompt: 'What is mlflow?',
        experimental_telemetry: { isEnabled: true }
      });

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');

      const tokenUsage = trace.info.tokenUsage;
      expect(tokenUsage).toBeDefined();
      expect(typeof tokenUsage?.input_tokens).toBe('number');
      expect(typeof tokenUsage?.output_tokens).toBe('number');
      expect(typeof tokenUsage?.total_tokens).toBe('number');

      expect(trace.data.spans.length).toBe(2);
      const span = trace.data.spans[0];
      expect(span.name).toBe('ai.generateText');
      expect(span.spanType).toBe(mlflow.SpanType.LLM);
      expect(span.inputs).toEqual({ prompt: 'What is mlflow?' });
      expect(span.outputs).toEqual({ text: result.text });
      expect(span.startTime).toBeDefined();
      expect(span.endTime).toBeDefined();

      expect(trace.data.spans[1].name).toBe('ai.generateText.doGenerate');
      expect(trace.data.spans[1].spanType).toBe(mlflow.SpanType.LLM);
      expect(trace.data.spans[1].inputs).toHaveProperty('messages');
      expect(trace.data.spans[1].startTime).toBeDefined();
      expect(trace.data.spans[1].endTime).toBeDefined();
    });

    it('should trace ai.generatetext() with system prompt', async () => {
      const result = await generateText({
        model: openai('gpt-4o-mini'),
        system: 'You are a helpful assistant.',
        prompt: 'What is mlflow?',
        experimental_telemetry: { isEnabled: true }
      });

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');

      const span = trace.data.spans[0];
      expect(span.name).toBe('ai.generateText');
      expect(span.spanType).toBe(mlflow.SpanType.LLM);
      expect(span.inputs).toEqual({
        system: 'You are a helpful assistant.',
        prompt: 'What is mlflow?'
      });
      expect(span.outputs).toEqual({ text: result.text });
      expect(span.startTime).toBeDefined();
      expect(span.endTime).toBeDefined();
    });

    it('should trace ai.generatetext() with messages', async () => {
      const result = await generateText({
        model: openai('gpt-4o-mini'),
        messages: [{ role: 'user', content: 'What is mlflow?' }],
        experimental_telemetry: { isEnabled: true }
      });

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');

      const span = trace.data.spans[0];
      expect(span.name).toBe('ai.generateText');
      expect(span.spanType).toBe(mlflow.SpanType.LLM);
      expect(span.inputs).toEqual({ messages: [{ role: 'user', content: 'What is mlflow?' }] });
      expect(span.outputs).toEqual({ text: result.text });
      expect(span.startTime).toBeDefined();
      expect(span.endTime).toBeDefined();
    });

    it('should handle ai.streamText()', async () => {
      const result = streamText({
        model: openai('gpt-4o-mini'),
        prompt: 'What is mlflow?',
        experimental_telemetry: { isEnabled: true }
      });

      for await (const textPart of result.textStream) {
        console.log(textPart);
      }

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');

      const span = trace.data.spans[0];
      expect(span.name).toBe('ai.streamText');
      expect(span.spanType).toBe(mlflow.SpanType.LLM);
      expect(span.inputs).toEqual({ prompt: 'What is mlflow?' });
      // Vercel AI SDK doesn't propagate streaming response to telemetry
      expect(span.outputs).toBeUndefined();
      expect(span.startTime).toBeDefined();
      expect(span.endTime).toBeDefined();
    });

    it('should handle endpoint errors properly', async () => {
      // Configure MSW to return rate limit error
      openAIMswServer.use(
        http.post('https://api.openai.com/v1/responses', () => {
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

      const openai = createOpenAI({ apiKey: 'test-key' });
      await expect(
        generateText({
          model: openai('gpt-4o-mini'),
          prompt: 'What is mlflow?',
          experimental_telemetry: { isEnabled: true }
        })
      ).rejects.toThrow();

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('ERROR');

      const span = trace.data.spans[0];
      expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
      expect(span.inputs).toEqual({ prompt: 'What is mlflow?' });
      expect(span.outputs).toBeUndefined();
      expect(span.startTime).toBeDefined();
      expect(span.endTime).toBeDefined();
    });

    it('should trace request wrapped in a parent span', async () => {
      const result = await mlflow.withSpan(
        async (_span) => {
          const response = await generateText({
            model: openai('gpt-4o-mini'),
            prompt: 'What is mlflow?',
            experimental_telemetry: { isEnabled: true }
          });
          return response.text;
        },
        {
          name: 'predict',
          spanType: mlflow.SpanType.CHAIN,
          inputs: 'Hello!'
        }
      );

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(3);

      const parentSpan = trace.data.spans[0];
      expect(parentSpan.name).toBe('predict');
      expect(parentSpan.spanType).toBe(mlflow.SpanType.CHAIN);

      const childSpan = trace.data.spans[1];
      expect(childSpan.name).toBe('ai.generateText');
      expect(childSpan.spanType).toBe(mlflow.SpanType.LLM);
      expect(childSpan.inputs).toEqual({ prompt: 'What is mlflow?' });
      expect(childSpan.outputs).toEqual({ text: result });
    });
  });
});
