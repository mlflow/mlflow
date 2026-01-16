/**
 * Tests for MLflow Anthropic integration with MSW mock server
 */

import * as mlflow from 'mlflow-tracing';
import Anthropic from '@anthropic-ai/sdk';
import { tracedAnthropic } from '../src';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { anthropicMockHandlers } from './mockAnthropicServer';
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

    const anthropic = new Anthropic({ apiKey: 'test-key' });
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
});
