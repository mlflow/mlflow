import { ROOT_CONTEXT, SpanStatusCode } from '@opentelemetry/api';
import type { SpanExporter, ReadableSpan, Span } from '@opentelemetry/sdk-trace-base';
import {
  MlflowClient,
  SpanAttributeKey,
  TraceInfo,
  TraceMetadataKey,
  TraceState,
  createTraceLocationFromUcTablePrefix,
} from '@mlflow/core';
import { MLflowSpanProcessor, type V4TraceInfoOptions } from '../src/processor';

const TRACE_A = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
const TRACE_B = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';
const ROOT_ID = '0000000000000001';
const CHILD_ID = '0000000000000002';
const SECOND_CHILD_ID = '0000000000000003';
const WRAPPER_ID = '0000000000000004';
const GRANDCHILD_ID = '0000000000000005';

interface MakeSpanOptions {
  traceId?: string;
  spanId?: string;
  parentSpanId?: string;
  attributes?: Record<string, unknown>;
  startTime?: [number, number];
  endTime?: [number, number];
  statusCode?: SpanStatusCode;
  useParentSpanContext?: boolean;
}

function makeSpan({
  traceId = TRACE_A,
  spanId = ROOT_ID,
  parentSpanId,
  attributes = {},
  startTime = [1_700_000_000, 100_000_000],
  endTime = [1_700_000_001, 600_000_000],
  statusCode = SpanStatusCode.UNSET,
  useParentSpanContext = false,
}: MakeSpanOptions = {}): ReadableSpan {
  return {
    attributes,
    name: `span-${spanId}`,
    spanContext: () => ({ traceId, spanId, traceFlags: 1 }),
    duration: [1, 500_000_000],
    startTime,
    endTime,
    ended: true,
    status: { code: statusCode },
    kind: 0,
    resource: { attributes: {} },
    instrumentationLibrary: { name: 'test' },
    events: [],
    links: [],
    ...(useParentSpanContext
      ? {
          parentSpanContext: parentSpanId
            ? { traceId, spanId: parentSpanId, traceFlags: 1 }
            : undefined,
        }
      : { parentSpanId }),
  } as unknown as ReadableSpan;
}

function getAttr(span: ReadableSpan, key: string): unknown {
  return (span.attributes as Record<string, unknown>)[key];
}

function createMockExporter(): SpanExporter & {
  exportMock: jest.Mock;
  shutdownMock: jest.Mock;
} {
  const exportMock = jest.fn((_spans: ReadableSpan[], cb: (result: { code: number }) => void) => {
    cb({ code: 0 });
  });
  const shutdownMock = jest.fn(() => Promise.resolve());

  return {
    export: exportMock,
    shutdown: shutdownMock,
    exportMock,
    shutdownMock,
  };
}

function createMockClient(
  implementation: (
    location: string,
    traceId: string,
    traceInfo: TraceInfo,
  ) => Promise<TraceInfo> = (_location, _traceId, traceInfo) => Promise.resolve(traceInfo),
): MlflowClient & { createTraceInfoV4: jest.Mock } {
  return {
    createTraceInfoV4: jest.fn(implementation),
  } as unknown as MlflowClient & { createTraceInfoV4: jest.Mock };
}

function traceInfoOptions(
  client: MlflowClient,
  overrides: Partial<V4TraceInfoOptions> = {},
): V4TraceInfoOptions {
  return {
    client,
    traceLocation: createTraceLocationFromUcTablePrefix('catalog', 'schema', 'prefix'),
    ...overrides,
  };
}

function parseMetadata(traceInfo: TraceInfo, key: string): Record<string, number> | undefined {
  const value = traceInfo.traceMetadata[key];
  return value ? (JSON.parse(value) as Record<string, number>) : undefined;
}

function tokenUsage(input: number, output: number): string {
  return JSON.stringify({
    input_tokens: input,
    output_tokens: output,
    total_tokens: input + output,
  });
}

function cost(input: number, output: number, includeTotal = true): string {
  return JSON.stringify({
    input_cost: input,
    output_cost: output,
    ...(includeTotal ? { total_cost: input + output } : {}),
  });
}

describe('MLflowSpanProcessor', () => {
  it('keeps the one-argument constructor behavior unchanged', async () => {
    const exporter = createMockExporter();
    const processor = new MLflowSpanProcessor(exporter);
    const span = makeSpan({
      attributes: { 'ai.operationId': 'ai.generateText', 'ai.model.id': 'gpt-5' },
    });

    processor.onEnd(span);
    await processor.forceFlush();

    expect(getAttr(span, 'mlflow.spanType')).toBe('LLM');
    expect(getAttr(span, 'mlflow.llm.model')).toBe('gpt-5');
    expect(getAttr(span, 'mlflow.llm.cost')).toBeUndefined();
    expect(exporter.exportMock).toHaveBeenCalledWith([span], expect.any(Function));
    await processor.shutdown();
  });

  it('translates before both accumulation and batched export', async () => {
    const exporter = createMockExporter();
    const client = createMockClient();
    const processor = new MLflowSpanProcessor(exporter, {
      traceInfo: traceInfoOptions(client),
    });
    const root = makeSpan({
      attributes: {
        'ai.operationId': 'ai.generateText',
        'ai.usage.inputTokens': 7,
        'ai.usage.outputTokens': 3,
      },
    });

    processor.onEnd(root);
    await processor.forceFlush();

    const written = client.createTraceInfoV4.mock.calls[0][2] as TraceInfo;
    expect(parseMetadata(written, TraceMetadataKey.TOKEN_USAGE)).toEqual({
      input_tokens: 7,
      output_tokens: 3,
      total_tokens: 10,
    });
    const exported = exporter.exportMock.mock.calls[0][0][0] as ReadableSpan;
    expect(getAttr(exported, 'mlflow.chat.tokenUsage')).toBe(tokenUsage(7, 3));
    await processor.shutdown();
  });

  it('writes TraceInfo exactly once when the root completes', async () => {
    const client = createMockClient();
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client),
    });
    const child = makeSpan({ spanId: CHILD_ID, parentSpanId: ROOT_ID });
    const root = makeSpan();

    processor.onEnd(child);
    processor.onEnd(root);
    processor.onEnd(root);
    await processor.forceFlush();

    expect(client.createTraceInfoV4).toHaveBeenCalledTimes(1);
    await processor.shutdown();
  });

  it('aggregates token and cost metadata from a single child', async () => {
    const client = createMockClient();
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client),
    });
    processor.onEnd(
      makeSpan({
        spanId: CHILD_ID,
        parentSpanId: ROOT_ID,
        attributes: {
          'mlflow.chat.tokenUsage': JSON.stringify({
            input_tokens: 10,
            output_tokens: 4,
            total_tokens: 14,
            cache_read_input_tokens: 2,
            cache_creation_input_tokens: 1,
          }),
          'mlflow.llm.cost': cost(0.01, 0.02, false),
        },
      }),
    );
    processor.onEnd(makeSpan());
    await processor.forceFlush();

    const traceInfo = client.createTraceInfoV4.mock.calls[0][2] as TraceInfo;
    expect(parseMetadata(traceInfo, TraceMetadataKey.TOKEN_USAGE)).toEqual({
      input_tokens: 10,
      output_tokens: 4,
      total_tokens: 14,
      cache_read_input_tokens: 2,
      cache_creation_input_tokens: 1,
    });
    expect(parseMetadata(traceInfo, TraceMetadataKey.COST)).toEqual({
      input_cost: 0.01,
      output_cost: 0.02,
      total_cost: 0.03,
    });
    await processor.shutdown();
  });

  it('computes UC span and trace cost from actual Vercel model usage', async () => {
    const client = createMockClient();
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client),
    });
    const child = makeSpan({
      spanId: CHILD_ID,
      parentSpanId: ROOT_ID,
      attributes: {
        'ai.operationId': 'ai.generateText.doGenerate',
        'ai.model.id': 'gpt-5-mini',
        'ai.model.provider': 'openai.responses',
        'ai.usage.inputTokens': 100,
        'ai.usage.outputTokens': 20,
      },
    });

    processor.onEnd(child);
    processor.onEnd(makeSpan());
    await processor.forceFlush();

    const spanCost = JSON.parse(getAttr(child, SpanAttributeKey.LLM_COST) as string) as Record<
      string,
      number
    >;
    expect(spanCost.input_cost).toBeCloseTo(0.000025, 12);
    expect(spanCost.output_cost).toBeCloseTo(0.00004, 12);
    expect(spanCost.total_cost).toBeCloseTo(0.000065, 12);
    const traceInfo = client.createTraceInfoV4.mock.calls[0][2] as TraceInfo;
    const traceCost = parseMetadata(traceInfo, TraceMetadataKey.COST)!;
    expect(traceCost.input_cost).toBeCloseTo(0.000025, 12);
    expect(traceCost.output_cost).toBeCloseTo(0.00004, 12);
    expect(traceCost.total_cost).toBeCloseTo(0.000065, 12);
    await processor.shutdown();
  });

  it('preserves an explicit span cost instead of applying catalog pricing', async () => {
    const client = createMockClient();
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client),
    });
    const explicitCost = cost(1, 2);
    const child = makeSpan({
      spanId: CHILD_ID,
      parentSpanId: ROOT_ID,
      attributes: {
        'ai.operationId': 'ai.generateText.doGenerate',
        'ai.model.id': 'gpt-5-mini',
        'ai.model.provider': 'openai.responses',
        'ai.usage.inputTokens': 100,
        'ai.usage.outputTokens': 20,
        [SpanAttributeKey.LLM_COST]: explicitCost,
      },
    });

    processor.onEnd(child);
    processor.onEnd(makeSpan());
    await processor.forceFlush();

    expect(getAttr(child, SpanAttributeKey.LLM_COST)).toBe(explicitCost);
    const traceInfo = client.createTraceInfoV4.mock.calls[0][2] as TraceInfo;
    expect(parseMetadata(traceInfo, TraceMetadataKey.COST)).toEqual({
      input_cost: 1,
      output_cost: 2,
      total_cost: 3,
    });
    await processor.shutdown();
  });

  it('sums two independent model-call children', async () => {
    const client = createMockClient();
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client),
    });
    processor.onEnd(
      makeSpan({
        spanId: CHILD_ID,
        parentSpanId: ROOT_ID,
        attributes: {
          'mlflow.chat.tokenUsage': tokenUsage(10, 5),
          'mlflow.llm.cost': cost(0.1, 0.2),
        },
      }),
    );
    processor.onEnd(
      makeSpan({
        spanId: SECOND_CHILD_ID,
        parentSpanId: ROOT_ID,
        attributes: {
          'mlflow.chat.tokenUsage': tokenUsage(4, 6),
          'mlflow.llm.cost': cost(0.04, 0.06),
        },
      }),
    );
    processor.onEnd(makeSpan());
    await processor.forceFlush();

    const traceInfo = client.createTraceInfoV4.mock.calls[0][2] as TraceInfo;
    expect(parseMetadata(traceInfo, TraceMetadataKey.TOKEN_USAGE)).toEqual({
      input_tokens: 14,
      output_tokens: 11,
      total_tokens: 25,
    });
    expect(parseMetadata(traceInfo, TraceMetadataKey.COST)).toEqual({
      input_cost: 0.14,
      output_cost: 0.26,
      total_cost: 0.4,
    });
    await processor.shutdown();
  });

  it('does not double-count a metric-bearing parent and child', async () => {
    const client = createMockClient();
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client),
    });
    processor.onEnd(
      makeSpan({
        spanId: CHILD_ID,
        parentSpanId: ROOT_ID,
        attributes: {
          'mlflow.chat.tokenUsage': tokenUsage(3, 2),
          'mlflow.llm.cost': cost(0.03, 0.02),
        },
      }),
    );
    processor.onEnd(
      makeSpan({
        attributes: {
          'mlflow.chat.tokenUsage': tokenUsage(30, 20),
          'mlflow.llm.cost': cost(0.3, 0.2),
        },
      }),
    );
    await processor.forceFlush();

    const traceInfo = client.createTraceInfoV4.mock.calls[0][2] as TraceInfo;
    expect(parseMetadata(traceInfo, TraceMetadataKey.TOKEN_USAGE)?.total_tokens).toBe(5);
    expect(parseMetadata(traceInfo, TraceMetadataKey.COST)?.total_cost).toBe(0.05);
    await processor.shutdown();
  });

  it('excludes repeated ancestors through a non-metric wrapper', async () => {
    const client = createMockClient();
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client),
    });
    processor.onEnd(
      makeSpan({
        spanId: GRANDCHILD_ID,
        parentSpanId: WRAPPER_ID,
        useParentSpanContext: true,
        attributes: { 'mlflow.chat.tokenUsage': tokenUsage(2, 1) },
      }),
    );
    processor.onEnd(
      makeSpan({
        spanId: WRAPPER_ID,
        parentSpanId: ROOT_ID,
        useParentSpanContext: true,
      }),
    );
    processor.onEnd(makeSpan({ attributes: { 'mlflow.chat.tokenUsage': tokenUsage(20, 10) } }));
    await processor.forceFlush();

    const traceInfo = client.createTraceInfoV4.mock.calls[0][2] as TraceInfo;
    expect(parseMetadata(traceInfo, TraceMetadataKey.TOKEN_USAGE)).toEqual({
      input_tokens: 2,
      output_tokens: 1,
      total_tokens: 3,
    });
    await processor.shutdown();
  });

  it('ignores malformed JSON and non-numeric metric values', async () => {
    const client = createMockClient();
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client),
    });
    processor.onEnd(
      makeSpan({
        spanId: CHILD_ID,
        parentSpanId: ROOT_ID,
        attributes: {
          'mlflow.chat.tokenUsage': '{bad-json',
          'mlflow.llm.cost': JSON.stringify({ total_cost: 'expensive' }),
        },
      }),
    );
    processor.onEnd(makeSpan());
    await processor.forceFlush();

    const traceInfo = client.createTraceInfoV4.mock.calls[0][2] as TraceInfo;
    expect(traceInfo.traceMetadata[TraceMetadataKey.TOKEN_USAGE]).toBeUndefined();
    expect(traceInfo.traceMetadata[TraceMetadataKey.COST]).toBeUndefined();
    await processor.shutdown();
  });

  it('lets explicit trace metadata override synthesized values', async () => {
    const client = createMockClient();
    const explicitUsage = tokenUsage(100, 50);
    const explicitCost = cost(1, 2);
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client, {
        traceMetadata: {
          [TraceMetadataKey.TOKEN_USAGE]: explicitUsage,
          [TraceMetadataKey.COST]: explicitCost,
          custom: 'value',
        },
      }),
    });
    processor.onEnd(
      makeSpan({
        attributes: {
          'mlflow.chat.tokenUsage': tokenUsage(1, 1),
          'mlflow.llm.cost': cost(0.1, 0.1),
        },
      }),
    );
    await processor.forceFlush();

    const traceInfo = client.createTraceInfoV4.mock.calls[0][2] as TraceInfo;
    expect(traceInfo.traceMetadata[TraceMetadataKey.TOKEN_USAGE]).toBe(explicitUsage);
    expect(traceInfo.traceMetadata[TraceMetadataKey.COST]).toBe(explicitCost);
    expect(traceInfo.traceMetadata.custom).toBe('value');
    await processor.shutdown();
  });

  it('isolates concurrent trace IDs', async () => {
    const client = createMockClient();
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client),
    });
    processor.onEnd(
      makeSpan({
        traceId: TRACE_A,
        spanId: CHILD_ID,
        parentSpanId: ROOT_ID,
        attributes: {
          'mlflow.chat.tokenUsage': tokenUsage(1, 2),
        },
      }),
    );
    processor.onEnd(
      makeSpan({
        traceId: TRACE_B,
        spanId: CHILD_ID,
        parentSpanId: ROOT_ID,
        attributes: {
          'mlflow.chat.tokenUsage': tokenUsage(10, 20),
        },
      }),
    );
    processor.onEnd(makeSpan({ traceId: TRACE_B }));
    processor.onEnd(makeSpan({ traceId: TRACE_A }));
    await processor.forceFlush();

    const byOtelId = new Map(
      client.createTraceInfoV4.mock.calls.map((call) => [call[1], call[2] as TraceInfo]),
    );
    expect(parseMetadata(byOtelId.get(TRACE_A)!, TraceMetadataKey.TOKEN_USAGE)?.total_tokens).toBe(
      3,
    );
    expect(parseMetadata(byOtelId.get(TRACE_B)!, TraceMetadataKey.TOKEN_USAGE)?.total_tokens).toBe(
      30,
    );
    await processor.shutdown();
  });

  it('exports spans and reports a TraceInfo write failure', async () => {
    const exporter = createMockExporter();
    const failure = new Error('write failed');
    const client = createMockClient(() => Promise.reject(failure));
    const onTraceInfoError = jest.fn();
    const processor = new MLflowSpanProcessor(exporter, {
      traceInfo: traceInfoOptions(client, { onTraceInfoError }),
    });

    processor.onEnd(makeSpan());
    await processor.forceFlush();

    expect(exporter.exportMock).toHaveBeenCalledTimes(1);
    expect(onTraceInfoError).toHaveBeenCalledWith(failure, TRACE_A);
    await processor.shutdown();
  });

  it('forceFlush waits for pending TraceInfo writes', async () => {
    let resolveWrite: ((traceInfo: TraceInfo) => void) | undefined;
    const client = createMockClient(
      (_location, _traceId, traceInfo) =>
        new Promise((resolve) => {
          resolveWrite = () => resolve(traceInfo);
        }),
    );
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client),
    });
    processor.onEnd(makeSpan());

    let flushed = false;
    const flush = processor.forceFlush().then(() => {
      flushed = true;
    });
    await Promise.resolve();
    expect(flushed).toBe(false);

    resolveWrite!({} as TraceInfo);
    await flush;
    expect(flushed).toBe(true);
    await processor.shutdown();
  });

  it('tracks active children when the root ends first', async () => {
    const client = createMockClient();
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client),
    });
    const root = makeSpan();
    const child = makeSpan({
      spanId: CHILD_ID,
      parentSpanId: ROOT_ID,
      attributes: { 'mlflow.chat.tokenUsage': tokenUsage(5, 2) },
    });
    processor.onStart(root as unknown as Span, ROOT_CONTEXT);
    processor.onStart(child as unknown as Span, ROOT_CONTEXT);

    processor.onEnd(root);
    await Promise.resolve();
    expect(client.createTraceInfoV4).not.toHaveBeenCalled();

    processor.onEnd(child);
    await processor.forceFlush();
    const traceInfo = client.createTraceInfoV4.mock.calls[0][2] as TraceInfo;
    expect(parseMetadata(traceInfo, TraceMetadataKey.TOKEN_USAGE)?.total_tokens).toBe(7);
    await processor.shutdown();
  });

  it('finalizes rooted traces and reports rootless traces at shutdown', async () => {
    const client = createMockClient();
    const onTraceInfoError = jest.fn();
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client, { onTraceInfoError }),
    });
    const root = makeSpan({ traceId: TRACE_A });
    const activeChild = makeSpan({ traceId: TRACE_A, spanId: CHILD_ID, parentSpanId: ROOT_ID });
    processor.onStart(root as unknown as Span, ROOT_CONTEXT);
    processor.onStart(activeChild as unknown as Span, ROOT_CONTEXT);
    processor.onEnd(root);
    processor.onEnd(makeSpan({ traceId: TRACE_B, spanId: CHILD_ID, parentSpanId: ROOT_ID }));

    await processor.shutdown();

    expect(client.createTraceInfoV4).toHaveBeenCalledTimes(1);
    expect(client.createTraceInfoV4.mock.calls[0][1]).toBe(TRACE_A);
    expect(onTraceInfoError).toHaveBeenCalledWith(expect.any(Error), TRACE_B);
  });

  it('opportunistically evicts abandoned traces without creating a timer', async () => {
    const client = createMockClient();
    const onTraceInfoError = jest.fn();
    const setIntervalSpy = jest.spyOn(global, 'setInterval');
    const now = jest.spyOn(Date, 'now');
    now.mockReturnValueOnce(1_000).mockReturnValue(1_011);
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client, { maxTraceAgeMs: 10, onTraceInfoError }),
    });

    processor.onEnd(makeSpan({ traceId: TRACE_A, spanId: CHILD_ID, parentSpanId: ROOT_ID }));
    processor.onEnd(makeSpan({ traceId: TRACE_B, spanId: CHILD_ID, parentSpanId: ROOT_ID }));

    expect(onTraceInfoError).toHaveBeenCalledWith(expect.any(Error), TRACE_A);
    expect(setIntervalSpy).not.toHaveBeenCalled();
    now.mockRestore();
    setIntervalSpy.mockRestore();
    await processor.shutdown();
  });

  it('bounds the number of retained active traces', async () => {
    const client = createMockClient();
    const onTraceInfoError = jest.fn();
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client, { maxActiveTraces: 1, onTraceInfoError }),
    });

    processor.onEnd(makeSpan({ traceId: TRACE_A, spanId: CHILD_ID, parentSpanId: ROOT_ID }));
    processor.onEnd(makeSpan({ traceId: TRACE_B, spanId: CHILD_ID, parentSpanId: ROOT_ID }));

    expect(onTraceInfoError).toHaveBeenCalledWith(expect.any(Error), TRACE_A);
    await processor.shutdown();
  });

  it('serializes the UC endpoint arguments and canonical TraceInfo fields', async () => {
    const client = createMockClient();
    const processor = new MLflowSpanProcessor(createMockExporter(), {
      traceInfo: traceInfoOptions(client),
    });
    processor.onEnd(
      makeSpan({
        startTime: [1_700_000_000, 100_000_000],
        endTime: [1_700_000_002, 600_000_000],
        statusCode: SpanStatusCode.ERROR,
        attributes: {
          'mlflow.spanInputs': '{"prompt":"hello"}',
          'mlflow.spanOutputs': '{"answer":"goodbye"}',
        },
      }),
    );
    await processor.forceFlush();

    expect(client.createTraceInfoV4).toHaveBeenCalledWith(
      'catalog.schema.prefix',
      TRACE_A,
      expect.any(TraceInfo),
    );
    const traceInfo = client.createTraceInfoV4.mock.calls[0][2] as TraceInfo;
    expect(traceInfo.traceId).toBe(`trace:/catalog.schema.prefix/${TRACE_A}`);
    expect(traceInfo.requestTime).toBe(1_700_000_000_100);
    expect(traceInfo.executionDuration).toBe(2_500);
    expect(traceInfo.state).toBe(TraceState.ERROR);
    expect(traceInfo.requestPreview).toBe('{"prompt":"hello"}');
    expect(traceInfo.responsePreview).toBe('{"answer":"goodbye"}');
    expect(traceInfo.toJson()).toMatchObject({
      trace_id: `trace:/catalog.schema.prefix/${TRACE_A}`,
      request_time: '2023-11-14T22:13:20.100Z',
      execution_duration: '2.5s',
      state: TraceState.ERROR,
      trace_location: {
        uc_table_prefix: {
          catalog_name: 'catalog',
          schema_name: 'schema',
          table_prefix: 'prefix',
        },
      },
    });
    await processor.shutdown();
  });
});
