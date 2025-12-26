/**
 * Tests for MLflow Gemini integration with MSW mock server
 */

import * as mlflow from 'mlflow-tracing';
import { tracedGemini } from '../src';
import { GoogleGenAI } from '@google/genai';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { geminiMockHandlers } from './mockGeminiServer';
import { createAuthProvider } from 'mlflow-tracing/src/auth';

const TEST_TRACKING_URI = 'http://localhost:5000';

describe('tracedGemini', () => {
  let experimentId: string;
  let client: mlflow.MlflowClient;
  let server: ReturnType<typeof setupServer>;

  beforeAll(async () => {
    const authProvider = createAuthProvider({ trackingUri: TEST_TRACKING_URI });
    client = new mlflow.MlflowClient({ trackingUri: TEST_TRACKING_URI, authProvider });

    const experimentName = `test-experiment-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
    experimentId = await client.createExperiment(experimentName);
    mlflow.init({
      trackingUri: TEST_TRACKING_URI,
      experimentId: experimentId
    });

    server = setupServer(...geminiMockHandlers);
    server.listen();
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

  describe('Generate Content', () => {
    it('should trace models.generateContent() without parent span', async () => {
      const gemini = new GoogleGenAI({ apiKey: 'test-key' });
      const wrappedGemini = tracedGemini(gemini);

      const result = await wrappedGemini.models.generateContent({
        model: 'gemini-2.0-flash-001',
        contents: 'Hello Gemini'
      });

      expect(result).toBeDefined();
      expect(result.usageMetadata).toBeDefined();

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');

      expect(trace.data.spans.length).toBe(1);

      const llmSpan = trace.data.spans[0];
      expect(llmSpan).toBeDefined();
      expect(llmSpan.name).toBe('generateContent');
      expect(llmSpan.spanType).toBe(mlflow.SpanType.LLM);
      expect(llmSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(llmSpan.inputs).toEqual({
        model: 'gemini-2.0-flash-001',
        contents: 'Hello Gemini'
      });
      expect(llmSpan.outputs).toEqual(result);
      expect(llmSpan.startTime).toBeDefined();
      expect(llmSpan.endTime).toBeDefined();

      const spanTokenUsage = llmSpan.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE];
      expect(spanTokenUsage).toBeDefined();
      expect(typeof spanTokenUsage[mlflow.TokenUsageKey.INPUT_TOKENS]).toBe('number');
      expect(typeof spanTokenUsage[mlflow.TokenUsageKey.OUTPUT_TOKENS]).toBe('number');
      expect(typeof spanTokenUsage[mlflow.TokenUsageKey.TOTAL_TOKENS]).toBe('number');
      expect(spanTokenUsage[mlflow.TokenUsageKey.INPUT_TOKENS]).toBe(10);
      expect(spanTokenUsage[mlflow.TokenUsageKey.OUTPUT_TOKENS]).toBe(5);
      expect(spanTokenUsage[mlflow.TokenUsageKey.TOTAL_TOKENS]).toBe(15);

      const messageFormat = llmSpan.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
      expect(messageFormat).toBe('gemini');
    });

    it('should trace models.generateContent()', async () => {
      const gemini = new GoogleGenAI({ apiKey: 'test-key' });
      const wrappedGemini = tracedGemini(gemini);

      const result = await mlflow.withSpan(
        async () => {
          return await wrappedGemini.models.generateContent({
            model: 'gemini-2.0-flash-001',
            contents: 'Hello Gemini'
          });
        },
        {
          name: 'test-trace',
          spanType: mlflow.SpanType.CHAIN
        }
      );

      expect(result).toBeDefined();
      expect(result.usageMetadata).toBeDefined();

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');

      expect(trace.data.spans.length).toBe(2);

      const llmSpan = trace.data.spans.find((s) => s.spanType === mlflow.SpanType.LLM)!;
      expect(llmSpan).toBeDefined();
      expect(llmSpan.name).toBe('generateContent');
      expect(llmSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(llmSpan.inputs).toEqual({
        model: 'gemini-2.0-flash-001',
        contents: 'Hello Gemini'
      });
      expect(llmSpan.outputs).toEqual(result);
      expect(llmSpan.startTime).toBeDefined();
      expect(llmSpan.endTime).toBeDefined();

      const spanTokenUsage = llmSpan.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE];
      expect(spanTokenUsage).toBeDefined();
      expect(typeof spanTokenUsage[mlflow.TokenUsageKey.INPUT_TOKENS]).toBe('number');
      expect(typeof spanTokenUsage[mlflow.TokenUsageKey.OUTPUT_TOKENS]).toBe('number');
      expect(typeof spanTokenUsage[mlflow.TokenUsageKey.TOTAL_TOKENS]).toBe('number');
      expect(spanTokenUsage[mlflow.TokenUsageKey.INPUT_TOKENS]).toBe(10);
      expect(spanTokenUsage[mlflow.TokenUsageKey.OUTPUT_TOKENS]).toBe(5);
      expect(spanTokenUsage[mlflow.TokenUsageKey.TOTAL_TOKENS]).toBe(15);

      const messageFormat = llmSpan.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT];
      expect(messageFormat).toBe('gemini');
    });

    it('should handle generateContent errors properly', async () => {
      server.use(
        http.post(
          'https://generativelanguage.googleapis.com/v1beta/models/*\\:generateContent',
          () => {
            return HttpResponse.json(
              {
                error: {
                  code: 429,
                  message: 'Resource has been exhausted',
                  status: 'RESOURCE_EXHAUSTED'
                }
              },
              { status: 429 }
            );
          }
        )
      );

      const gemini = new GoogleGenAI({ apiKey: 'test-key' });
      const wrappedGemini = tracedGemini(gemini);

      await expect(
        mlflow.withSpan(
          async () => {
            return await wrappedGemini.models.generateContent({
              model: 'gemini-2.0-flash-001',
              contents: 'This should fail'
            });
          },
          {
            name: 'error-test-trace',
            spanType: mlflow.SpanType.CHAIN
          }
        )
      ).rejects.toThrow();

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('ERROR');

      const llmSpan = trace.data.spans.find((s) => s.spanType === mlflow.SpanType.LLM);
      expect(llmSpan).toBeDefined();
      expect(llmSpan!.status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
      expect(llmSpan!.inputs).toEqual({
        model: 'gemini-2.0-flash-001',
        contents: 'This should fail'
      });
      expect(llmSpan!.outputs).toBeUndefined();
      expect(llmSpan!.startTime).toBeDefined();
      expect(llmSpan!.endTime).toBeDefined();
    });

    it('should trace Gemini request wrapped in a parent span', async () => {
      const gemini = new GoogleGenAI({ apiKey: 'test-key' });
      const wrappedGemini = tracedGemini(gemini);

      const result = await mlflow.withSpan(
        async (_span) => {
          const response = await wrappedGemini.models.generateContent({
            model: 'gemini-2.0-flash-001',
            contents: 'Hello from parent span'
          });
          return response;
        },
        {
          name: 'predict',
          spanType: mlflow.SpanType.CHAIN,
          inputs: 'Hello from parent span'
        }
      );

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(2);

      const parentSpan = trace.data.spans[0];
      expect(parentSpan.name).toBe('predict');
      expect(parentSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(parentSpan.spanType).toBe(mlflow.SpanType.CHAIN);
      expect(parentSpan.inputs).toEqual('Hello from parent span');
      expect(parentSpan.outputs).toEqual(result);
      expect(parentSpan.startTime).toBeDefined();
      expect(parentSpan.endTime).toBeDefined();

      const childSpan = trace.data.spans[1];
      expect(childSpan.name).toBe('generateContent');
      expect(childSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
      expect(childSpan.spanType).toBe(mlflow.SpanType.LLM);
      expect(childSpan.inputs).toEqual({
        model: 'gemini-2.0-flash-001',
        contents: 'Hello from parent span'
      });
      expect(childSpan.outputs).toBeDefined();
      expect(childSpan.startTime).toBeDefined();
      expect(childSpan.endTime).toBeDefined();
    });

    it('should not trace methods outside SUPPORTED_METHODS', () => {
      const gemini = new GoogleGenAI({ apiKey: 'test-key' });
      const wrappedGemini = tracedGemini(gemini);

      expect(wrappedGemini.models).toBeDefined();
    });
  });
});
