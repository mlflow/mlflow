import { isNil, isPlainObject } from 'lodash';
import {
  convertOtelAttributesToMap,
  decodeSpanId,
  getModelTraceSpanId,
  isV3ModelTraceInfo,
  isV3ModelTraceSpan,
  tryDeserializeAttribute,
} from '../../../shared/web-shared/model-trace-explorer/ModelTraceExplorer.utils';
import { createTraceV4LongIdentifier } from '../../../shared/web-shared/model-trace-explorer/traceV4Identifiers';
import type {
  ModelTrace,
  ModelTraceEvent,
  ModelTraceInfoV3,
  ModelTraceSpan,
  ModelTraceSpanV3,
  ModelTraceState,
} from '../../../shared/web-shared/model-trace-explorer';
import { doesTraceSupportV4API } from '../../../shared/web-shared/genai-traces-table';

const TRACE_NAME_TAG = 'mlflow.traceName';
const SPAN_INPUTS_ATTRIBUTE = 'mlflow.spanInputs';
const SPAN_TYPE_ATTRIBUTE = 'mlflow.spanType';
const EXCEPTION_EVENT_NAME = 'exception';

// Long enough to recognise a prompt or an id, short enough that one field cannot push the error
// off the card. The full value is one click away in the trace itself.
const FIELD_VALUE_MAX_CHARS = 160;

/** Input fields listed on an expanded card before the rest collapse into "+N more". */
export const MAX_EVIDENCE_FIELDS = 4;

export interface IssueEvidenceField {
  /** Empty when the span's input is a bare value rather than a keyed object. */
  key: string;
  value: string;
}

/**
 * One trace's summary row, flattened for a card header. This comes from a trace-info read, which
 * costs ~5 KB where the same trace's spans cost ~15 MB, so every card can carry its metadata while
 * only an expanded card pays for detail.
 */
export interface IssueEvidenceHeader {
  /** The bare id, as the artifact carries it — what the caller orders and keys cards by. */
  traceId: string;
  /** The id anything fetching or addressing this trace needs. See `getAddressableTraceId`. */
  addressableTraceId: string;
  name?: string;
  /** Epoch millis, for a relative age. Undefined when the trace carries no usable request time. */
  requestTimeMs?: number;
  /** Pre-formatted by the API, e.g. `1.4s`. */
  executionDuration?: string;
  state?: ModelTraceState;
}

export interface SpanIdentity {
  id: string;
  name: string;
  type?: string;
}

/**
 * The level a card's claim is made at. Shared vocabulary rather than two sets of literals, because
 * `locus` and `errorScope` below are compared against each other -- a card whose evidence is a span
 * but whose failure is only trace-level renders differently from one where both are the span.
 */
export const EVIDENCE_SPAN_SCOPE = 'span';
export const EVIDENCE_TRACE_SCOPE = 'trace';
/** `errorScope` only: no failure to report, which is not a level and so never a `locus`. */
export const EVIDENCE_NO_ERROR = 'none';

export type IssueEvidenceLocus = typeof EVIDENCE_SPAN_SCOPE | typeof EVIDENCE_TRACE_SCOPE;
/** Every locus, plus the absence of a failure -- so the two can never drift apart. */
export type IssueEvidenceErrorScope = IssueEvidenceLocus | typeof EVIDENCE_NO_ERROR;

/**
 * What only a trace's spans can answer, so it arrives with the fetch a card's expansion triggers.
 */
export interface IssueEvidenceDetail {
  /**
   * Whether this card's evidence is one span's or the whole trace's. `trace` is not a fallback for a
   * failed lookup — it is the right answer whenever no span owns the issue, and it costs no span
   * read at all.
   */
  locus: IssueEvidenceLocus;
  /** Trace-level evidence: what was asked and what came back, as the API previews them. */
  requestPreview?: string;
  responsePreview?: string;
  /** Span the fields and the error were read off, named so the card can attribute them. */
  spanName?: string;
  /** That same span's id, so the link to the trace can open on it. */
  spanId?: string;
  fields: IssueEvidenceField[];
  /**
   * Where the failure is, independently of whether anything recorded *why*; the card owes an Error
   * row whenever this is not `none`. `span` and `trace` are kept apart because the span shown is
   * not always the one that failed — an issue may name a *successful* span as its evidence, and
   * "this span failed" would misattribute it.
   */
  errorScope: IssueEvidenceErrorScope;
  /** Undefined when the failure exists but nothing recorded a message for it. */
  errorText?: string;
}

/**
 * A trace as either read path hands it over: `useGetTracesById` resolves `{ info, data.spans }`,
 * and the batched UC read's `{ trace_info, spans }` is normalized to the same shape by the caller.
 */
export interface FetchedTrace {
  info: ModelTrace['info'] | undefined;
  data?: { spans?: ModelTraceSpan[] };
}

const truncate = (value: string) =>
  value.length > FIELD_VALUE_MAX_CHARS ? `${value.slice(0, FIELD_VALUE_MAX_CHARS)}…` : value;

const stringifyFieldValue = (value: unknown): string => {
  if (typeof value === 'string') {
    return truncate(value);
  }
  try {
    return truncate(JSON.stringify(value) ?? String(value));
  } catch {
    // Circular or otherwise unserialisable input: show the coerced value rather than dropping
    // the field, so the reader still sees that the span had one.
    return String(value);
  }
};

const isExceptionEvent = (event: ModelTraceEvent) => event.name === EXCEPTION_EVENT_NAME;

// lodash types `isPlainObject` as returning `boolean`, so it narrows nothing on its own. A
// deserialized span attribute should be treated as fields only when it really is a plain object,
// which `isPlainObject` checks and a bare `typeof value === 'object'` would not.
const isRecord = (value: unknown): value is Record<string, unknown> => isPlainObject(value);

// `isV3ModelTraceSpan` admits v4 spans too, and `convertOtelAttributesToMap` flattens v4's
// key/value attribute array (and its events') into the map v3 already uses — so after both, every
// span reads as v3 regardless of which fork produced it.
const normalizeSpans = (spans: ModelTraceSpan[]): ModelTraceSpanV3[] =>
  spans
    .filter(isV3ModelTraceSpan)
    .map(convertOtelAttributesToMap)
    // Re-guard rather than assert: `convertOtelAttributesToMap` is typed as returning the union, so
    // narrowing it back by hand would be a claim about its behaviour that nothing checks.
    .filter(isV3ModelTraceSpan);

export const toSpanIdentities = (trace: FetchedTrace | undefined): SpanIdentity[] => {
  const spans = normalizeSpans(trace?.data?.spans ?? []);
  return spans.map((span) => {
    const rawType = span.attributes?.[SPAN_TYPE_ATTRIBUTE];
    const type = typeof rawType === 'string' ? tryDeserializeAttribute(rawType) : rawType;
    return {
      id: getModelTraceSpanId(span),
      name: span.name,
      ...(typeof type === 'string' ? { type } : {}),
    };
  });
};

/**
 * Reduces an id to the bare lowercase hex the explorer keys its span tree on, via the explorer's
 * own decoder so any wire form MLflow has written (base64, `0x`-prefixed) resolves. `0x` is
 * stripped after decoding because `decodeSpanId` returns early for a short id whose base64 decode
 * fails, skipping its own prefix handling. Exported for the collapsed-card deep link, which must
 * emit the form the explorer matches without fetching spans.
 */
export const toExplorerSpanId = (spanId: string) => decodeSpanId(spanId.trim(), true).replace(/^0x/i, '').toLowerCase();

/**
 * The first span the issue names that the fetched trace actually carries. The engine orders its
 * ids most-specific-first, so walking them in order both honours that judgement and degrades to
 * the next named span — a better fallback than jumping straight to inference when the leading id
 * has been sampled out or the trace re-ingested.
 */
const findNamedSpan = (spans: ModelTraceSpanV3[], spanIds: string[]): ModelTraceSpanV3 | undefined => {
  const byNormalizedId = new Map(spans.map((span) => [toExplorerSpanId(getModelTraceSpanId(span)), span]));
  for (const spanId of spanIds) {
    const wanted = toExplorerSpanId(spanId);
    const span = wanted ? byNormalizedId.get(wanted) : undefined;
    if (span) {
      return span;
    }
  }
  return undefined;
};

const getSpanFields = (span: ModelTraceSpanV3 | undefined): IssueEvidenceField[] => {
  const rawInputs = span?.attributes?.[SPAN_INPUTS_ATTRIBUTE];
  if (isNil(rawInputs)) {
    return [];
  }
  const inputs = typeof rawInputs === 'string' ? tryDeserializeAttribute(rawInputs) : rawInputs;
  if (isRecord(inputs)) {
    return Object.entries(inputs).map(([key, value]) => ({
      key,
      value: stringifyFieldValue(value),
    }));
  }
  const value = stringifyFieldValue(inputs);
  return value === '' ? [] : [{ key: '', value }];
};

const getSpanErrorText = (span: ModelTraceSpanV3 | undefined): string | undefined => {
  const exception = span?.events?.find(isExceptionEvent);
  const type = exception?.attributes?.['exception.type'];
  const message = exception?.attributes?.['exception.message'];
  if (typeof message === 'string' && message !== '') {
    return typeof type === 'string' && type !== '' ? `${type}: ${message}` : message;
  }
  const statusMessage = span?.status?.message;
  return typeof statusMessage === 'string' && statusMessage !== '' ? statusMessage : undefined;
};

/**
 * The id anything resolving this trace needs — a per-trace read, or the `selectedEvaluationId` param
 * the drawer reopens from. UC-resident traces get the long `trace:/<location>/<id>` identifier, the
 * rest their bare id.
 *
 * Deliberately un-gated on the v4 rollout flag: the flag says which API a page prefers, not what is
 * resolvable, and a bare OTEL id in the URL reopens nothing on a cold load. Gating on
 * `doesTraceSupportV4API` alone also matches `useActiveEvaluation`'s writer, so a row and a card
 * cannot give one trace two URL forms.
 */
export const getAddressableTraceId = (info: ModelTraceInfoV3): string =>
  doesTraceSupportV4API(info) ? createTraceV4LongIdentifier(info) : info.trace_id;

/**
 * Flattens a trace's summary row into a card header, or `undefined` for a response whose info is
 * not a v3/v4 trace.
 */
export const toIssueEvidenceHeader = (info: ModelTrace['info'] | undefined): IssueEvidenceHeader | undefined => {
  if (!isV3ModelTraceInfo(info)) {
    return undefined;
  }
  const requestTimeMs = info.request_time ? new Date(info.request_time).getTime() : NaN;

  return {
    traceId: info.trace_id,
    addressableTraceId: getAddressableTraceId(info),
    name: info.tags?.[TRACE_NAME_TAG] || undefined,
    requestTimeMs: Number.isNaN(requestTimeMs) ? undefined : requestTimeMs,
    executionDuration: info.execution_duration,
    state: info.state,
  };
};

/**
 * Reads the evidence for one trace — one span's inputs and error where the issue names a span, the
 * trace's own request and response where it does not.
 *
 * `namedSpanIds` (derived from the issue's evidence entries, most specific first) is the *only* source of a
 * span; nothing is inferred. Picking the span that errored would read an error-shaped issue into
 * every category, and the span that failed is not the one that owns a cost or latency finding.
 */
export const toIssueEvidenceDetail = (
  trace: FetchedTrace,
  { namedSpanIds }: { namedSpanIds?: string[] } = {},
): IssueEvidenceDetail | undefined => {
  const { info } = trace;
  if (!isV3ModelTraceInfo(info)) {
    return undefined;
  }
  const spans = normalizeSpans(trace.data?.spans ?? []);
  const offendingSpan = findNamedSpan(spans, namedSpanIds ?? []);
  const traceErrored = info.state === 'ERROR';

  if (!offendingSpan) {
    return {
      locus: EVIDENCE_TRACE_SCOPE,
      requestPreview: info.request_preview || undefined,
      responsePreview: info.response_preview || undefined,
      fields: [],
      // No span to pin anything on, so the trace's own state is the only claim available. OSS
      // trace info carries no error message, so the card says a failure exists without one.
      errorScope: traceErrored ? EVIDENCE_TRACE_SCOPE : EVIDENCE_NO_ERROR,
    };
  }

  const spanErrored = offendingSpan.status?.code === 'STATUS_CODE_ERROR';
  // Empty-string is treated as absent -- the API sends one for "nothing recorded".
  const errorText = getSpanErrorText(offendingSpan) || undefined;

  return {
    locus: EVIDENCE_SPAN_SCOPE,
    spanName: offendingSpan.name,
    spanId: getModelTraceSpanId(offendingSpan) || undefined,
    fields: getSpanFields(offendingSpan),
    // The span's own status wins: it is the specific claim. A trace-level mark with a span that
    // succeeded is still a failure worth surfacing, just not one to pin on this span.
    errorScope: spanErrored ? EVIDENCE_SPAN_SCOPE : traceErrored ? EVIDENCE_TRACE_SCOPE : EVIDENCE_NO_ERROR,
    errorText,
  };
};

/**
 * Whether a loaded detail has anything to show. A failure counts even with no message attached:
 * the body still names the span it came from and says the detail is missing, which is what tells
 * the reader the gap is in the trace rather than in this pane.
 */
export const hasEvidenceDetail = (detail: IssueEvidenceDetail): boolean =>
  detail.fields.length > 0 ||
  detail.errorScope !== EVIDENCE_NO_ERROR ||
  Boolean(detail.requestPreview) ||
  Boolean(detail.responsePreview);
