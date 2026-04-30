import { trace as otelTrace } from '@opentelemetry/api';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-node';

import { SpanAttributeKey, SpanLogLevel, SpanType, toSpanLogLevel } from '../../src/core/constants';
import { defaultLogLevelForSpanType } from '../../src/core/log_level';
import { createMlflowSpan, type LiveSpan } from '../../src/core/entities/span';

const provider = new BasicTracerProvider();
otelTrace.setGlobalTracerProvider(provider);
const tracer = otelTrace.getTracer('mlflow-log-level-test', '1.0.0');

const newLiveSpan = (): LiveSpan => {
  const otelSpan = tracer.startSpan('test');
  return createMlflowSpan(otelSpan, 'tr-test', SpanType.LLM) as LiveSpan;
};

describe('toSpanLogLevel', () => {
  it.each<[SpanLogLevel | number | string, SpanLogLevel]>([
    [SpanLogLevel.DEBUG, SpanLogLevel.DEBUG],
    [SpanLogLevel.CRITICAL, SpanLogLevel.CRITICAL],
    [10, SpanLogLevel.DEBUG],
    [20, SpanLogLevel.INFO],
    [30, SpanLogLevel.WARNING],
    [40, SpanLogLevel.ERROR],
    [50, SpanLogLevel.CRITICAL],
    ['DEBUG', SpanLogLevel.DEBUG],
    ['info', SpanLogLevel.INFO],
    ['Warning', SpanLogLevel.WARNING],
    ['WARN', SpanLogLevel.WARNING], // alias
    ['warn', SpanLogLevel.WARNING],
    ['  ERROR  ', SpanLogLevel.ERROR], // whitespace tolerated
  ])('accepts %p and yields %p', (input, expected) => {
    expect(toSpanLogLevel(input)).toBe(expected);
  });

  it.each(['NOPE', 'TRACE', 'FATAL', 'INFOO', ''])('rejects invalid string %p', (value) => {
    expect(() => toSpanLogLevel(value)).toThrow(/Invalid SpanLogLevel/);
  });

  it.each([0, 7, 100, -1])('rejects invalid int %p', (value) => {
    expect(() => toSpanLogLevel(value)).toThrow(/Invalid SpanLogLevel/);
  });
});

describe('LiveSpan log-level accessors', () => {
  it('logLevel is null until set', () => {
    const span = newLiveSpan();
    expect(span.logLevel).toBeNull();
    span.end();
  });

  it.each<[SpanLogLevel | number | string, SpanLogLevel]>([
    [SpanLogLevel.WARNING, SpanLogLevel.WARNING],
    [40, SpanLogLevel.ERROR],
    ['info', SpanLogLevel.INFO],
    ['WARN', SpanLogLevel.WARNING],
  ])('setLogLevel(%p) round-trips to %p', (input, expected) => {
    const span = newLiveSpan();
    span.setLogLevel(input);
    expect(span.logLevel).toBe(expected);
    // Stored as the canonical int under the reserved attribute key for portability.
    expect(span.getAttribute(SpanAttributeKey.LOG_LEVEL)).toBe(expected as number);
    span.end();
  });

  it('setLogLevel rejects invalid input', () => {
    const span = newLiveSpan();
    expect(() => span.setLogLevel('NOPE')).toThrow(/Invalid SpanLogLevel/);
    span.end();
  });
});

describe('defaultLogLevelForSpanType', () => {
  it.each<[string, SpanLogLevel]>([
    [SpanType.LLM, SpanLogLevel.INFO],
    [SpanType.CHAT_MODEL, SpanLogLevel.INFO],
    [SpanType.AGENT, SpanLogLevel.INFO],
    [SpanType.TOOL, SpanLogLevel.INFO],
    [SpanType.RETRIEVER, SpanLogLevel.INFO],
    [SpanType.EMBEDDING, SpanLogLevel.INFO],
    [SpanType.MEMORY, SpanLogLevel.INFO],
    [SpanType.UNKNOWN, SpanLogLevel.INFO],
    [SpanType.CHAIN, SpanLogLevel.DEBUG],
    [SpanType.PARSER, SpanLogLevel.DEBUG],
    [SpanType.RERANKER, SpanLogLevel.DEBUG],
    ['MY_CUSTOM_TYPE', SpanLogLevel.INFO], // custom types fall through to INFO
  ])('maps %p to %p', (spanType, expected) => {
    expect(defaultLogLevelForSpanType(spanType)).toBe(expected);
  });

  it('maps undefined/null to INFO', () => {
    expect(defaultLogLevelForSpanType(undefined)).toBe(SpanLogLevel.INFO);
    expect(defaultLogLevelForSpanType(null)).toBe(SpanLogLevel.INFO);
  });
});
