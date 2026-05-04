import { trace as otelTrace } from '@opentelemetry/api';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-node';

import { SpanAttributeKey, SpanLogLevel, SpanType, toSpanLogLevel } from '../../src/core/constants';
import { defaultLogLevelForSpanType } from '../../src/core/log_level';
import { createMlflowSpan, type LiveSpan } from '../../src/core/entities/span';
import { SpanEvent } from '../../src/core/entities/span_event';

const provider = new BasicTracerProvider();
otelTrace.setGlobalTracerProvider(provider);
const tracer = otelTrace.getTracer('mlflow-log-level-test', '1.0.0');

const newLiveSpan = (spanType: SpanType = SpanType.UNKNOWN): LiveSpan => {
  const otelSpan = tracer.startSpan('test');
  return createMlflowSpan(otelSpan, 'tr-test', spanType) as LiveSpan;
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
    ['  ERROR  ', SpanLogLevel.ERROR], // whitespace tolerated
  ])('accepts %p and yields %p', (input, expected) => {
    expect(toSpanLogLevel(input)).toBe(expected);
  });

  it.each(['NOPE', 'TRACE', 'FATAL', 'INFOO', '', 'WARN', 'warn'])(
    'rejects invalid string %p',
    (value) => {
      // "WARN" is no longer accepted; only the full names work.
      expect(() => toSpanLogLevel(value)).toThrow(/Invalid SpanLogLevel/);
    },
  );

  it.each([0, 7, 100, -1])('rejects invalid int %p', (value) => {
    expect(() => toSpanLogLevel(value)).toThrow(/Invalid SpanLogLevel/);
  });
});

describe('LiveSpan constructor default', () => {
  it.each<[SpanType, SpanLogLevel]>([
    [SpanType.LLM, SpanLogLevel.INFO],
    [SpanType.CHAT_MODEL, SpanLogLevel.INFO],
    [SpanType.TOOL, SpanLogLevel.INFO],
    [SpanType.RETRIEVER, SpanLogLevel.INFO],
    [SpanType.AGENT, SpanLogLevel.INFO],
    [SpanType.EMBEDDING, SpanLogLevel.INFO],
    [SpanType.CHAIN, SpanLogLevel.DEBUG],
    [SpanType.PARSER, SpanLogLevel.DEBUG],
    [SpanType.RERANKER, SpanLogLevel.DEBUG],
    [SpanType.MEMORY, SpanLogLevel.DEBUG],
    [SpanType.UNKNOWN, SpanLogLevel.DEBUG],
  ])('stamps default %p for span type %p', (spanType, expected) => {
    const span = newLiveSpan(spanType);
    expect(span.logLevel).toBe(expected);
    span.end();
  });
});

describe('LiveSpan log-level accessors', () => {
  it.each<[SpanLogLevel | number | string, SpanLogLevel]>([
    [SpanLogLevel.WARNING, SpanLogLevel.WARNING],
    [40, SpanLogLevel.ERROR],
    ['info', SpanLogLevel.INFO],
    ['CRITICAL', SpanLogLevel.CRITICAL],
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

describe('exception event bumps log level to ERROR', () => {
  it('bumps DEBUG span to ERROR via addEvent', () => {
    const span = newLiveSpan(SpanType.PARSER);
    expect(span.logLevel).toBe(SpanLogLevel.DEBUG);
    span.addEvent(new SpanEvent({ name: 'exception', attributes: { 'exception.message': 'boom' } }));
    expect(span.logLevel).toBe(SpanLogLevel.ERROR);
    span.end();
  });

  it('bumps INFO span to ERROR via addEvent', () => {
    const span = newLiveSpan(SpanType.CHAT_MODEL);
    expect(span.logLevel).toBe(SpanLogLevel.INFO);
    span.addEvent(new SpanEvent({ name: 'exception', attributes: { 'exception.message': 'boom' } }));
    expect(span.logLevel).toBe(SpanLogLevel.ERROR);
    span.end();
  });

  it('preserves user-set CRITICAL', () => {
    const span = newLiveSpan(SpanType.PARSER);
    span.setLogLevel(SpanLogLevel.CRITICAL);
    span.addEvent(new SpanEvent({ name: 'exception', attributes: { 'exception.message': 'boom' } }));
    expect(span.logLevel).toBe(SpanLogLevel.CRITICAL);
    span.end();
  });

  it('bumps via recordException (which bypasses our addEvent wrapper)', () => {
    const span = newLiveSpan(SpanType.PARSER);
    span.recordException(new Error('boom'));
    expect(span.logLevel).toBe(SpanLogLevel.ERROR);
    span.end();
  });

  it('non-exception events do not bump the level', () => {
    const span = newLiveSpan(SpanType.CHAT_MODEL);
    span.addEvent(new SpanEvent({ name: 'my_event', attributes: { k: 'v' } }));
    expect(span.logLevel).toBe(SpanLogLevel.INFO);
    span.end();
  });
});

describe('defaultLogLevelForSpanType', () => {
  it.each<[string, SpanLogLevel]>([
    // INFO set: user-visible semantic operations.
    [SpanType.LLM, SpanLogLevel.INFO],
    [SpanType.CHAT_MODEL, SpanLogLevel.INFO],
    [SpanType.AGENT, SpanLogLevel.INFO],
    [SpanType.TOOL, SpanLogLevel.INFO],
    [SpanType.RETRIEVER, SpanLogLevel.INFO],
    [SpanType.EMBEDDING, SpanLogLevel.INFO],
    // DEBUG set: internal/glue work and unclassified types.
    [SpanType.CHAIN, SpanLogLevel.DEBUG],
    [SpanType.PARSER, SpanLogLevel.DEBUG],
    [SpanType.RERANKER, SpanLogLevel.DEBUG],
    [SpanType.MEMORY, SpanLogLevel.DEBUG],
    [SpanType.UNKNOWN, SpanLogLevel.DEBUG],
    ['MY_CUSTOM_TYPE', SpanLogLevel.DEBUG], // custom types fall through to DEBUG
  ])('maps %p to %p', (spanType, expected) => {
    expect(defaultLogLevelForSpanType(spanType)).toBe(expected);
  });

  it('maps undefined/null to DEBUG', () => {
    expect(defaultLogLevelForSpanType(undefined)).toBe(SpanLogLevel.DEBUG);
    expect(defaultLogLevelForSpanType(null)).toBe(SpanLogLevel.DEBUG);
  });
});
