/**
 * MLflow Span Link Entities
 *
 * This module provides TypeScript implementations of MLflow span link entities,
 * compatible with OpenTelemetry for correlating spans across different traces.
 */

/**
 * Parameters for creating a SpanLink instance.
 */
export interface SpanLinkParams {
  /** The trace ID of the linked span (e.g., "tr-abc123" or hex string) */
  traceId: string;
  /** The span ID within that trace (16-character hex string) */
  spanId: string;
  /** Optional attributes describing the link relationship */
  attributes?: Record<string, any>;
}

/**
 * JSON representation of a span link, matching the Python `Link.to_dict()` format.
 */
export interface SerializedSpanLink {
  trace_id: string;
  span_id: string;
  attributes?: Record<string, any> | null;
}

/**
 * Represents an OpenTelemetry Span Link that connects spans across traces.
 *
 * Span Links allow correlating spans that don't have a parent-child relationship,
 * such as spans from different traces in multi-agent systems or distributed workflows.
 *
 * @example
 * ```typescript
 * const link = new SpanLink({
 *   traceId: 'tr-0123456789abcdef0123456789abcdef',
 *   spanId: '0123456789abcdef',
 *   attributes: { 'link.type': 'follows_from' },
 * });
 * ```
 */
export class SpanLink {
  readonly traceId: string;
  readonly spanId: string;
  readonly attributes: Record<string, any>;

  constructor(params: SpanLinkParams) {
    this.traceId = params.traceId;
    this.spanId = params.spanId;
    this.attributes = params.attributes ?? {};
  }

  toJson(): SerializedSpanLink {
    return {
      trace_id: this.traceId,
      span_id: this.spanId,
      attributes: Object.keys(this.attributes).length > 0 ? this.attributes : null,
    };
  }

  static fromJson(json: SerializedSpanLink): SpanLink {
    return new SpanLink({
      traceId: json.trace_id,
      spanId: json.span_id,
      attributes: json.attributes ?? undefined,
    });
  }
}
