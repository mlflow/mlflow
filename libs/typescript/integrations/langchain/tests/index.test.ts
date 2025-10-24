import * as mlflow from 'mlflow-tracing';
import { MlflowClient } from 'mlflow-tracing';
import { MlflowCallback } from '../src';
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from '@langchain/core/runnables';
import { ChatOpenAI } from "@langchain/openai";

import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { openAIMockHandlers } from '../../openai/tests/mockOpenAIServer';

const TEST_TRACKING_URI = 'http://localhost:5000';


describe('MlflowCallback integration', () => {
  let experimentId: string;
  let client: MlflowClient;
  let server: ReturnType<typeof setupServer>;

  beforeAll(async () => {
    // Setup MSW mock server
    server = setupServer(...openAIMockHandlers);
    server.listen();

    client = new MlflowClient({ trackingUri: TEST_TRACKING_URI, host: TEST_TRACKING_URI });
    const experimentName = `langchain-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    experimentId = await client.createExperiment(experimentName);
    mlflow.init({
      trackingUri: TEST_TRACKING_URI,
      experimentId
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

  it('records spans for runnable sequence with LLM', async () => {
    const handler = new MlflowCallback();

    const model = new ChatOpenAI({apiKey: 'test-key'});
    const promptTemplate = PromptTemplate.fromTemplate('Hello, {name}!');
    const chain = RunnableSequence.from([promptTemplate, model]);

    const result = await chain.invoke({ name: 'hello' }, { callbacks: [handler] });
    expect(result.content).toEqual('Test response content');

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('OK');
    expect(trace.data.spans.length).toBe(3);

    const chainSpan = trace.data.spans[0];
    expect(chainSpan.spanType).toBe(mlflow.SpanType.CHAIN);
    expect(chainSpan.inputs).toEqual({ name: 'hello' });
    expect(chainSpan.outputs).toBeDefined();
    expect(chainSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
    expect(chainSpan.parentId).toBeNull();

    const promptTemplateSpan = trace.data.spans[1];
    expect(promptTemplateSpan.spanType).toBe(mlflow.SpanType.CHAIN);
    expect(promptTemplateSpan.inputs).toEqual({ name: 'hello' });
    expect(promptTemplateSpan.outputs).toBeDefined();
    expect(promptTemplateSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
    expect(promptTemplateSpan.parentId).toBe(chainSpan.spanId);

    const llmSpan = trace.data.spans[2];
    expect(llmSpan.spanType).toBe(mlflow.SpanType.CHAT_MODEL);
    expect(llmSpan.inputs).toBeDefined();
    expect(llmSpan.outputs).toBeDefined();
    expect(llmSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
    expect(llmSpan.parentId).toBe(chainSpan.spanId);

    // Token count should be recorded
    expect(trace.info.tokenUsage).toBeDefined();
    expect(trace.info.tokenUsage?.input_tokens).toBe(100);
    expect(trace.info.tokenUsage?.output_tokens).toBe(200);
    expect(trace.info.tokenUsage?.total_tokens).toBe(300);

    expect(llmSpan.attributes[mlflow.SpanAttributeKey.MESSAGE_FORMAT]).toBe('langchain');
    expect(llmSpan.attributes[mlflow.SpanAttributeKey.TOKEN_USAGE]).toEqual({
      input_tokens: 100,
      output_tokens: 200,
      total_tokens: 300
    })
  });

  it('records spans for streaming invocation', async () => {
    const handler = new MlflowCallback();

    const model = new ChatOpenAI({apiKey: 'test-key'});
    const promptTemplate = PromptTemplate.fromTemplate('Hello, {name}!');
    const chain = RunnableSequence.from([promptTemplate, model]);

    const stream = await chain.stream({ name: 'hello' }, { callbacks: [handler] });
    for await (const chunk of stream) {
      console.log('streaming chunk', chunk);
    }

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('OK');
    expect(trace.data.spans.length).toBe(3);

    const chainSpan = trace.data.spans[0];
    expect(chainSpan.spanType).toBe(mlflow.SpanType.CHAIN);
    expect(chainSpan.inputs).toEqual({ name: 'hello' });
    expect(chainSpan.outputs).toBeDefined();
    expect(chainSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
    expect(chainSpan.parentId).toBeNull();

    const promptTemplateSpan = trace.data.spans[1];
    expect(promptTemplateSpan.spanType).toBe(mlflow.SpanType.CHAIN);
    expect(promptTemplateSpan.inputs).toEqual({ name: 'hello' });
    expect(promptTemplateSpan.outputs).toBeDefined();
    expect(promptTemplateSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
    expect(promptTemplateSpan.parentId).toBe(chainSpan.spanId);

    const llmSpan = trace.data.spans[2];
    expect(llmSpan.spanType).toBe(mlflow.SpanType.CHAT_MODEL);
    expect(llmSpan.inputs).toBeDefined();
    expect(llmSpan.outputs).toBeDefined();
    expect(llmSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
    expect(llmSpan.parentId).toBe(chainSpan.spanId);
  });

  it('records spans with a manual parent span', async () => {
    const handler = new MlflowCallback();

    const model = new ChatOpenAI({apiKey: 'test-key'});
    const promptTemplate = PromptTemplate.fromTemplate('Hello, {name}!');
    const chain = RunnableSequence.from([promptTemplate, model]);

    const result = await mlflow.withSpan(
      async (span) => {
        return await chain.invoke({ name: 'hello' }, { callbacks: [handler] });
      },
      {
        name: 'manual_parent_span',
        spanType: mlflow.SpanType.AGENT,
        inputs: { name: 'hello' },
        attributes: { manual_parent_span: true }
      },
    )
    expect(result.content).toEqual('Test response content');

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('OK');
    expect(trace.data.spans.length).toBe(4);

    const manualParentSpan = trace.data.spans[0];
    expect(manualParentSpan.name).toBe('manual_parent_span');
    expect(manualParentSpan.status.statusCode).toBe(mlflow.SpanStatusCode.OK);
    expect(manualParentSpan.spanType).toBe(mlflow.SpanType.AGENT);
    const chainSpan = trace.data.spans[1];
    expect(chainSpan.spanType).toBe(mlflow.SpanType.CHAIN);
    expect(chainSpan.parentId).toBe(manualParentSpan.spanId);
    const promptTemplateSpan = trace.data.spans[2];
    expect(promptTemplateSpan.spanType).toBe(mlflow.SpanType.CHAIN);
    expect(promptTemplateSpan.parentId).toBe(chainSpan.spanId);
    const llmSpan = trace.data.spans[3];
    expect(llmSpan.spanType).toBe(mlflow.SpanType.CHAT_MODEL);
    expect(llmSpan.parentId).toBe(chainSpan.spanId);
  });

  // TODO: add tests for streaming, tool agent, retriever,

  it('marks spans as error when runnable fails', async () => {
    // Configure MSW to return rate limit error
    server.use(
      http.post('https://api.openai.com/v1/chat/completions', () => {
        return HttpResponse.json(
          {
            error: {
              type: 'requests',
              message: 'Bad request'
            }
          },
          { status: 400 }
        );
      })
    );

    const handler = new MlflowCallback();

    const model = new ChatOpenAI({apiKey: 'test-key'});
    const promptTemplate = PromptTemplate.fromTemplate('Hello, {name}!');
    const chain = RunnableSequence.from([promptTemplate, model]);

    await expect(
      chain.invoke({ name: 'hello' }, { callbacks: [handler] })
    ).rejects.toThrow('Bad request');

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('ERROR');
    expect(trace.data.spans.length).toBe(3);
  });
});
