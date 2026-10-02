import * as mlflow from '@mlflow/core';
import { createAuthProvider } from '@mlflow/core/src/auth';
import {
  APIPromise,
  BadRequestError,
  choice,
  noul,
  TypeSafeClient,
  TypeSafeError,
  type Fetch,
} from '@typesafe-ai/sdk';
import { tracedTypeSafe } from '../src';

const TEST_TRACKING_URI = 'http://localhost:5000';
const TYPESAFE_BASE_URL = 'https://typesafe.test';

const SYSTEM_ONE_RESULT = {
  model: 'jev-resolved',
  answers: {
    relevant: { type: 'noul' as const, noul: 0.98 },
    tone: {
      type: 'choice' as const,
      choice: 'friendly',
      confidence: 0.9,
      probabilities: { friendly: 0.9, hostile: 0.1 },
    },
  },
  usage: { input_tokens: 12, output_tokens: 3 },
};

type RecordedRequest = {
  url: string;
  init: RequestInit | undefined;
  body: unknown;
};

function jsonResponse(
  body: unknown,
  init: { status?: number; headers?: Record<string, string> } = {},
): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { 'content-type': 'application/json', ...init.headers },
  });
}

function mockFetch(respond: (request: RecordedRequest) => Response | Promise<Response>): {
  fetch: Fetch;
  requests: RecordedRequest[];
} {
  const requests: RecordedRequest[] = [];
  const fetch: Fetch = async (input, init) => {
    const rawBody = typeof init?.body === 'string' ? init.body : undefined;
    const request = {
      url: input,
      init,
      body: rawBody === undefined ? undefined : (JSON.parse(rawBody) as unknown),
    };
    requests.push(request);
    return await respond(request);
  };
  return { fetch, requests };
}

describe('tracedTypeSafe', () => {
  let experimentId: string;
  let mlflowClient: mlflow.MlflowClient;

  beforeAll(async () => {
    const authProvider = createAuthProvider({ trackingUri: TEST_TRACKING_URI });
    mlflowClient = new mlflow.MlflowClient({
      trackingUri: TEST_TRACKING_URI,
      authProvider,
    });
    experimentId = await mlflowClient.createExperiment(
      `typesafe-typescript-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    );
    mlflow.init({ trackingUri: TEST_TRACKING_URI, experimentId });
  });

  afterAll(async () => {
    await mlflow.flushTraces();
    await mlflowClient.deleteExperiment(experimentId);
  });

  async function getLastTrace(): Promise<mlflow.Trace> {
    let lastError: unknown;
    for (let attempt = 0; attempt < 40; attempt++) {
      await new Promise((resolve) => setTimeout(resolve, 10));
      await mlflow.flushTraces();
      const traceId = mlflow.getLastActiveTraceId();
      if (!traceId) {
        continue;
      }
      try {
        return await mlflowClient.getTrace(traceId);
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError ?? new Error('Timed out waiting for the TypeSafe trace');
  }

  function makeClient(fetch: Fetch, defaultModel = 'jev-default'): TypeSafeClient {
    return tracedTypeSafe(
      new TypeSafeClient({
        apiKey: 'typesafe-test-key',
        baseURL: TYPESAFE_BASE_URL,
        defaultModel,
        retry: { maxRetries: 0 },
        fetch,
      }),
    );
  }

  it('captures the effective request, structured response, usage, model, and request ID', async () => {
    const { fetch, requests } = mockFetch(() =>
      jsonResponse(SYSTEM_ONE_RESULT, {
        headers: { 'x-typesafe-request-id': 'req_success' },
      }),
    );
    const client = makeClient(fetch, 'jev-requested');
    const request = {
      state: { document: 'Hello' },
      questions: {
        relevant: noul('Is this relevant?'),
        tone: choice('What is the tone?', { friendly: null, hostile: null }),
      },
      feature: 'test',
    };

    const result = await client.systemOne(request, {
      headers: { 'x-secret': 'do-not-trace' },
      timeout: 1_000,
    });

    expect(result).toEqual(SYSTEM_ONE_RESULT);
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      url: `${TYPESAFE_BASE_URL}/v1/systemone`,
      body: {
        ...request,
        questions: {
          relevant: { type: 'noul', instructions: 'Is this relevant?' },
          tone: {
            type: 'choice',
            instructions: 'What is the tone?',
            criteria: { friendly: null, hostile: null },
          },
        },
        model: 'jev-requested',
      },
    });

    const trace = await getLastTrace();
    expect(trace.info.state).toBe('OK');
    expect(trace.info.tokenUsage).toEqual({
      input_tokens: 12,
      output_tokens: 3,
      total_tokens: 15,
    });
    expect(trace.data.spans).toHaveLength(1);
    const span = trace.data.spans[0];
    expect(span.name).toBe('typesafe.system_one');
    expect(span.spanType).toBe(mlflow.SpanType.LLM);
    expect(span.logLevel).toBe(mlflow.SpanLogLevel.INFO);
    expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
    expect(span.inputs).toEqual(requests[0].body);
    expect(span.outputs).toEqual(SYSTEM_ONE_RESULT);
    expect(span.attributes['mlflow.llm.model']).toBe('jev-resolved');
    expect(span.attributes['mlflow.llm.provider']).toBe('typesafe');
    expect(span.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT]).toBe('typesafe');
    expect(span.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE]).toEqual({
      input_tokens: 12,
      output_tokens: 3,
      total_tokens: 15,
    });
    expect(span.attributes['typesafe.request_id']).toBe('req_success');
    expect(JSON.stringify(span.inputs)).not.toContain('do-not-trace');
  });

  it('returns the original APIPromise with response helpers and inferred result types', async () => {
    const { fetch } = mockFetch(() =>
      jsonResponse(SYSTEM_ONE_RESULT, {
        headers: { 'x-typesafe-request-id': 'req_promise' },
      }),
    );
    const client = makeClient(fetch);

    const promise = client.systemOne({
      state: 'hello',
      questions: { relevant: noul('Relevant?') },
    });
    expect(promise).toBeInstanceOf(APIPromise);

    const mapped = promise.map((result) => result.answers.relevant.noul);
    expect(mapped).toBeInstanceOf(APIPromise);
    const { data, requestId, response } = await mapped.withResponse();
    expect(data).toBe(0.98);
    expect(requestId).toBe('req_promise');
    expect(response.status).toBe(200);

    const trace = await getLastTrace();
    expect(trace.data.spans[0].outputs).toEqual(SYSTEM_ONE_RESULT);
  });

  it('clones the response without consuming the raw body', async () => {
    const { fetch } = mockFetch(() => jsonResponse(SYSTEM_ONE_RESULT));
    const client = makeClient(fetch);

    const response = await client
      .systemOne({ state: 'hello', questions: { relevant: noul('Relevant?') } })
      .asResponse();

    expect(response.bodyUsed).toBe(false);
    expect(await response.json()).toEqual(SYSTEM_ONE_RESULT);
    const trace = await getLastTrace();
    expect(trace.data.spans[0].outputs).toEqual(SYSTEM_ONE_RESULT);
  });

  it('records API errors and reraises the original TypeSafe error', async () => {
    const { fetch } = mockFetch(() =>
      jsonResponse(
        { message: 'invalid request' },
        { status: 400, headers: { 'x-typesafe-request-id': 'req_error' } },
      ),
    );
    const client = makeClient(fetch);

    const error = await client
      .systemOne({ state: 'hello', questions: { relevant: noul('Relevant?') } })
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(BadRequestError);
    expect(error).toMatchObject({ requestId: 'req_error' });
    const trace = await getLastTrace();
    expect(trace.info.state).toBe('ERROR');
    const span = trace.data.spans[0];
    expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
    expect(span.status.description).toContain('BadRequestError');
    expect(span.status.description).toContain('invalid request');
    expect(span.logLevel).toBe(mlflow.SpanLogLevel.ERROR);
    expect(span.outputs).toBeUndefined();
    expect(span.attributes['typesafe.request_id']).toBe('req_error');
  });

  it('records synchronous SDK validation errors without issuing a request', async () => {
    const { fetch, requests } = mockFetch(() => jsonResponse(SYSTEM_ONE_RESULT));
    const client = makeClient(fetch);

    expect(() => client.systemOne({ state: 'hello', questions: {} })).toThrow(TypeSafeError);
    expect(requests).toHaveLength(0);

    const trace = await getLastTrace();
    expect(trace.info.state).toBe('ERROR');
    const span = trace.data.spans[0];
    expect(span.status.statusCode).toBe(mlflow.SpanStatusCode.ERROR);
    expect(span.status.description).toContain('TypeSafeError');
    expect(span.logLevel).toBe(mlflow.SpanLogLevel.ERROR);
  });

  it('preserves parent-child relationships and does not trace other SDK methods', async () => {
    const { fetch } = mockFetch((request) => {
      if (request.url.endsWith('/v1/models')) {
        return jsonResponse({ models: [] });
      }
      return jsonResponse(SYSTEM_ONE_RESULT);
    });
    const client = makeClient(fetch);
    const beforeModelsList = mlflow.getLastActiveTraceId();
    await client.models.list();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(mlflow.getLastActiveTraceId()).toBe(beforeModelsList);

    await mlflow.withSpan(
      async () => {
        await client.systemOne({
          state: 'hello',
          questions: { relevant: noul('Relevant?') },
        });
      },
      { name: 'parent', spanType: mlflow.SpanType.CHAIN },
    );

    const trace = await getLastTrace();
    expect(trace.data.spans).toHaveLength(2);
    const parent = trace.data.spans.find((span) => span.name === 'parent');
    const child = trace.data.spans.find((span) => span.name === 'typesafe.system_one');
    expect(parent).toBeDefined();
    expect(child?.parentId).toBe(parent?.spanId);
  });
});
