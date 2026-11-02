import { describe, expect, test } from '@jest/globals';
import type { ModelTraceInfoV3, ModelTraceSpan } from '../../../shared/web-shared/model-trace-explorer';
import {
  hasEvidenceDetail,
  toIssueEvidenceDetail,
  toIssueEvidenceHeader,
  type FetchedTrace,
} from './issueEvidence.utils';

const traceInfo = (overrides: Partial<ModelTraceInfoV3> = {}): ModelTraceInfoV3 => ({
  trace_id: 'tr-1',
  trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'experiment-1' } },
  request_time: '2026-08-26T09:00:00.000Z',
  execution_duration: '1.4s',
  state: 'OK',
  tags: { 'mlflow.traceName': 'answer_question' },
  ...overrides,
});

const v3Span = (overrides: Partial<ModelTraceSpan> = {}): ModelTraceSpan =>
  ({
    trace_id: 'tr-1',
    span_id: 'span-root',
    trace_state: '',
    parent_span_id: null,
    name: 'root',
    start_time_unix_nano: '0',
    end_time_unix_nano: '1',
    status: { code: 'STATUS_CODE_OK' },
    attributes: {},
    ...overrides,
  }) as ModelTraceSpan;

/**
 * A card's two reads, composed. The header comes from a trace-info read and the detail from the
 * trace's spans, so most cases below exercise one function; composing keeps their assertions on one
 * object.
 */
const toIssueEvidence = (trace: FetchedTrace, namedSpanIds?: string[]) => {
  const header = toIssueEvidenceHeader(trace.info);
  const detail = toIssueEvidenceDetail(trace, { namedSpanIds });
  return header && detail ? { ...header, ...detail } : undefined;
};

/** The default `v3Span` id, so a case that wants span-level evidence has to ask for it by name. */
const ROOT = ['span-root'];

describe('toIssueEvidenceHeader', () => {
  test('reads the trace identity, timing and state off the trace info', () => {
    expect(toIssueEvidenceHeader(traceInfo())).toEqual({
      traceId: 'tr-1',
      // No UC location, so the id is the bare one rather than a `trace:/…` identifier.
      addressableTraceId: 'tr-1',
      name: 'answer_question',
      requestTimeMs: Date.parse('2026-08-26T09:00:00.000Z'),
      executionDuration: '1.4s',
      state: 'OK',
    });
  });

  // The whole point of reading info separately: a collapsed card renders its header without the
  // trace's spans ever being fetched.
  test('needs no spans at all', () => {
    expect(toIssueEvidenceHeader(traceInfo())?.name).toBe('answer_question');
  });

  test('omits the trace id when the trace carries no name tag', () => {
    expect(toIssueEvidenceHeader(traceInfo({ tags: {} }))?.name).toBeUndefined();
  });

  test('drops a response whose info is not a v3/v4 trace', () => {
    expect(toIssueEvidenceHeader({ request_id: 'tr-1' })).toBeUndefined();
  });
});

describe('toIssueEvidenceDetail', () => {
  test('drops a response whose info is not a v3/v4 trace', () => {
    expect(toIssueEvidenceDetail({ info: { request_id: 'tr-1' } })).toBeUndefined();
  });

  // The trace the detection engine keeps producing: a terminal guardrail fails in 90ms and the root
  // inherits ERROR. Which of the two is the evidence is the engine's call, not a shape this pane can
  // read off the spans -- so both are named here in turn.
  const inheritedErrorTrace: FetchedTrace = {
    info: traceInfo({ state: 'ERROR' }),
    data: {
      spans: [
        v3Span({
          name: 'customer_support_rag',
          status: { code: 'STATUS_CODE_ERROR' },
          attributes: { 'mlflow.spanInputs': JSON.stringify({ messages: 'the user question' }) },
        }),
        v3Span({
          span_id: 'span-guardrail',
          parent_span_id: 'span-root',
          name: 'output_guardrail',
          status: { code: 'STATUS_CODE_ERROR', message: 'guardrail rejected the response' },
          attributes: { 'mlflow.spanInputs': JSON.stringify({ candidate: 'the drafted answer' }) },
        }),
      ],
    },
  };

  test("reads the named span's inputs and error", () => {
    const evidence = toIssueEvidence(inheritedErrorTrace, ['span-guardrail']);

    expect(evidence?.spanName).toBe('output_guardrail');
    expect(evidence?.spanId).toBe('span-guardrail');
    expect(evidence?.fields).toEqual([{ key: 'candidate', value: 'the drafted answer' }]);
    expect(evidence?.errorText).toBe('guardrail rejected the response');
  });

  // The engine may name the root deliberately -- an issue about the request as a whole -- and this
  // pane has no business overriding that because a descendant also errored.
  test('reads the span the issue named even when a descendant also failed', () => {
    const evidence = toIssueEvidence(inheritedErrorTrace, ['span-root']);

    expect(evidence?.spanName).toBe('customer_support_rag');
    expect(evidence?.spanId).toBe('span-root');
    expect(evidence?.fields).toEqual([{ key: 'messages', value: 'the user question' }]);
  });

  // The engine orders its ids most-specific-first, and a leading id can be gone (sampled out, or
  // the trace re-ingested).
  test('uses the next named span when the first one is not in the trace', () => {
    const evidence = toIssueEvidence(inheritedErrorTrace, ['span-dropped-by-sampling', 'span-root']);

    expect(evidence?.spanName).toBe('customer_support_rag');
  });

  test('honours the order the engine gave, not the order the spans arrived in', () => {
    expect(toIssueEvidence(inheritedErrorTrace, ['span-guardrail', 'span-root'])?.spanName).toBe('output_guardrail');
  });

  // MLflow's own span ids have been written `0x`-prefixed; the explorer strips that, so a named id
  // carrying it still has to resolve.
  test('matches a named span id written with an 0x prefix', () => {
    expect(toIssueEvidence(inheritedErrorTrace, ['0xSPAN-ROOT'])?.spanName).toBe('customer_support_rag');
  });

  // A producer that copies the raw v3 wire value emits base64 where the span tree reads hex. The
  // ids are the same span, so the card must resolve it rather than falling back to the trace.
  test('matches a named span id still in its base64 wire form', () => {
    const trace = {
      info: traceInfo({ state: 'ERROR' }),
      data: {
        spans: [
          v3Span({ span_id: '3dff2d2426338578', name: 'output_guardrail', status: { code: 'STATUS_CODE_ERROR' } }),
        ],
      },
    };

    expect(toIssueEvidence(trace, ['Pf8tJCYzhXg='])?.spanName).toBe('output_guardrail');
    expect(toIssueEvidence(trace, ['Pf8tJCYzhXg='])?.spanId).toBe('3dff2d2426338578');
  });

  test('flattens a v4 span’s otel attribute array so its inputs read the same as v3', () => {
    const evidence = toIssueEvidence(
      {
        info: traceInfo(),
        data: {
          spans: [
            {
              trace_id: 'tr-1',
              span_id: 'span-root',
              parent_span_id: null,
              name: 'root',
              kind: 'SPAN_KIND_INTERNAL',
              start_time_unix_nano: '0',
              end_time_unix_nano: '1',
              attributes: [
                { key: 'mlflow.spanInputs', value: { string_value: JSON.stringify({ question: 'v4 input' }) } },
              ],
              status: { code: 'STATUS_CODE_OK' },
            },
          ] as ModelTraceSpan[],
        },
      },
      ROOT,
    );

    expect(evidence?.fields).toEqual([{ key: 'question', value: 'v4 input' }]);
  });

  test('keeps a non-object span input as a single unnamed field', () => {
    const evidence = toIssueEvidence(
      { info: traceInfo(), data: { spans: [v3Span({ attributes: { 'mlflow.spanInputs': 'just a string' } })] } },
      ROOT,
    );

    expect(evidence?.fields).toEqual([{ key: '', value: 'just a string' }]);
  });

  test('reports a trace-level failure with no message, which OSS trace info does not carry', () => {
    const evidence = toIssueEvidence({ info: traceInfo({ state: 'ERROR' }), data: { spans: [v3Span()] } }, ROOT);

    expect(evidence?.errorScope).toBe('trace');
    expect(evidence?.errorText).toBeUndefined();
  });

  test('reports no error for a trace that succeeded', () => {
    const evidence = toIssueEvidence({ info: traceInfo(), data: { spans: [v3Span()] } }, ROOT);

    expect(evidence?.errorScope).toBe('none');
    expect(evidence?.errorText).toBeUndefined();
  });

  test('flags the failure even when nothing recorded a message for it', () => {
    // The shape the demo generator emits and real OTEL exporters can too: status ERROR with no
    // description and no exception event. The card is tagged Error off the trace state, so it has
    // to be able to say the detail is missing rather than render an unexplained Error card.
    const evidence = toIssueEvidence(
      {
        info: traceInfo({ state: 'ERROR' }),
        data: { spans: [v3Span({ status: { code: 'STATUS_CODE_ERROR' }, events: [] })] },
      },
      ROOT,
    );

    expect(evidence?.errorScope).toBe('span');
    expect(evidence?.errorText).toBeUndefined();
  });

  test('flags a failed span even when the trace itself is not marked ERROR', () => {
    const evidence = toIssueEvidence(
      { info: traceInfo({ state: 'OK' }), data: { spans: [v3Span({ status: { code: 'STATUS_CODE_ERROR' } })] } },
      ROOT,
    );

    expect(evidence?.errorScope).toBe('span');
  });

  test('scopes the failure to the trace when the named span succeeded', () => {
    // An issue may name a span that did not fail -- the evidence for a correctness finding is often a
    // span that returned perfectly well. Reporting that as a span failure would misattribute it.
    const evidence = toIssueEvidence(
      {
        info: traceInfo({ state: 'ERROR' }),
        data: { spans: [v3Span({ name: 'vector_search', status: { code: 'STATUS_CODE_OK' } })] },
      },
      ROOT,
    );

    expect(evidence?.spanName).toBe('vector_search');
    expect(evidence?.errorScope).toBe('trace');
  });

  test('leaves requestTimeMs undefined for an unparseable request time', () => {
    expect(
      toIssueEvidence({ info: traceInfo({ request_time: 'not a date' }), data: { spans: [] } })?.requestTimeMs,
    ).toBeUndefined();
  });
});

describe('hasEvidenceDetail', () => {
  test('is false for a named span with neither fields nor an error', () => {
    const evidence = toIssueEvidence({ info: traceInfo(), data: { spans: [v3Span()] } }, ROOT);

    expect(evidence && hasEvidenceDetail(evidence)).toBe(false);
  });

  test('is true when the named span has inputs', () => {
    const evidence = toIssueEvidence(
      {
        info: traceInfo(),
        data: { spans: [v3Span({ attributes: { 'mlflow.spanInputs': JSON.stringify({ q: 'a' }) } })] },
      },
      ROOT,
    );

    expect(evidence && hasEvidenceDetail(evidence)).toBe(true);
  });

  test('is true for a failure with no fields and no message, so the span attribution stays reachable', () => {
    // Without this the common bare-ERROR trace gets no disclosure control at all, which also hides
    // "From span <name>" -- the one thing that says which span the pane blamed.
    const evidence = toIssueEvidence(
      { info: traceInfo({ state: 'ERROR' }), data: { spans: [v3Span({ status: { code: 'STATUS_CODE_ERROR' } })] } },
      ROOT,
    );

    expect(evidence?.fields).toEqual([]);
    expect(evidence?.errorText).toBeUndefined();
    expect(evidence && hasEvidenceDetail(evidence)).toBe(true);
  });
});

describe('nothing is inferred', () => {
  // A trace that happens to contain a failure unrelated to the issue. Picking a span by looking for
  // one that errored would attach that error as the issue's evidence -- and because `errorScope`
  // would become `span`, the card would assert that span failed *as this issue's evidence*.
  const traceWithUnrelatedError: FetchedTrace = {
    info: traceInfo({ request_preview: 'summarize this', response_preview: 'wrong summary' }),
    data: {
      spans: [
        v3Span({ attributes: { 'mlflow.spanInputs': JSON.stringify({ question: 'the request' }) } }),
        v3Span({
          span_id: 'span-cache-miss',
          parent_span_id: 'span-root',
          name: 'cache_lookup',
          status: { code: 'STATUS_CODE_ERROR' },
          events: [{ name: 'exception', attributes: { 'exception.message': 'cache miss' } }],
        }),
      ],
    },
  };

  // The span that failed is not the span that owns a cost or latency finding, and an error unrelated
  // to the issue is not evidence for it -- so an errored span is never promoted on its own.
  test('never picks a span the issue did not name, even when one errored', () => {
    const detail = toIssueEvidenceDetail(traceWithUnrelatedError);

    expect(detail).toMatchObject({ locus: 'trace', errorScope: 'none' });
    expect(detail?.spanName).toBeUndefined();
    expect(detail?.spanId).toBeUndefined();
    expect(detail?.errorText).toBeUndefined();
    expect(detail?.fields).toEqual([]);
  });

  test('reads the trace as evidence when the issue names no span', () => {
    expect(toIssueEvidenceDetail(traceWithUnrelatedError)).toMatchObject({
      locus: 'trace',
      requestPreview: 'summarize this',
      responsePreview: 'wrong summary',
    });
  });

  test('reads the trace as evidence when none of the named ids resolve', () => {
    const detail = toIssueEvidenceDetail(traceWithUnrelatedError, { namedSpanIds: ['nope-1', 'nope-2'] });

    expect(detail).toMatchObject({ locus: 'trace' });
    expect(detail?.spanName).toBeUndefined();
  });

  test('reads the trace as evidence when its spans came back empty', () => {
    const detail = toIssueEvidenceDetail({ info: traceInfo(), data: { spans: [] } }, { namedSpanIds: ROOT });

    expect(detail).toMatchObject({ locus: 'trace' });
    expect(detail?.spanId).toBeUndefined();
  });

  // With no span there are no fields, and a healthy trace has no error either, so the previews are
  // the only thing keeping the card off its empty state.
  test('counts trace-level previews as evidence worth expanding', () => {
    const detail = toIssueEvidenceDetail(traceWithUnrelatedError);

    expect(detail && hasEvidenceDetail(detail)).toBe(true);
  });

  test('has nothing to show for a healthy trace with no previews', () => {
    const detail = toIssueEvidenceDetail({ info: traceInfo() });

    expect(detail).toMatchObject({ locus: 'trace', errorScope: 'none' });
    expect(detail && hasEvidenceDetail(detail)).toBe(false);
  });

  test('still reports a trace-level failure with no span to pin it on', () => {
    expect(toIssueEvidenceDetail({ info: traceInfo({ state: 'ERROR' }) })).toMatchObject({
      locus: 'trace',
      errorScope: 'trace',
    });
  });
});
