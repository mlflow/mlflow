import {
  HrTime,
  SpanKind as OTelSpanKind,
  SpanStatusCode as OTelSpanStatusCode,
} from '@opentelemetry/api';
import type { ReadableSpan as OTelReadableSpan } from '@opentelemetry/sdk-trace-base';

/**
 * OTLP/HTTP+JSON serialization for OTel spans, matching the
 * OpenTelemetry Protocol HTTP JSON encoding accepted by Databricks at
 * `/api/2.0/otel/v1/traces`.
 *
 * See https://opentelemetry.io/docs/specs/otlp/#otlphttp-request for the spec.
 * Implemented manually to avoid pulling in `@opentelemetry/exporter-trace-otlp-http`
 * as a new npm dependency.
 */

interface OtlpKeyValue {
  key: string;
  value: OtlpAnyValue;
}

interface OtlpAnyValue {
  stringValue?: string;
  boolValue?: boolean;
  intValue?: string;
  doubleValue?: number;
  arrayValue?: { values: OtlpAnyValue[] };
  kvlistValue?: { values: OtlpKeyValue[] };
}

interface OtlpSpan {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  kind: number;
  startTimeUnixNano: string;
  endTimeUnixNano: string;
  attributes: OtlpKeyValue[];
  events: {
    timeUnixNano: string;
    name: string;
    attributes: OtlpKeyValue[];
  }[];
  status: {
    code: number;
    message?: string;
  };
}

interface OtlpScopeSpans {
  scope: { name: string; version?: string; attributes?: OtlpKeyValue[] };
  spans: OtlpSpan[];
}

interface OtlpResourceSpans {
  resource: { attributes: OtlpKeyValue[] };
  scopeSpans: OtlpScopeSpans[];
}

export interface OtlpExportTraceServiceRequest {
  resourceSpans: OtlpResourceSpans[];
}

function hrTimeToNanoString(hrTime: HrTime): string {
  return (BigInt(hrTime[0]) * 1_000_000_000n + BigInt(hrTime[1])).toString();
}

function spanKindToOtlp(kind: OTelSpanKind | undefined): number {
  // OTLP SpanKind enum matches OTel's enum values 0..5.
  return typeof kind === 'number' ? kind : 0;
}

function statusCodeToOtlp(code: OTelSpanStatusCode | undefined): number {
  switch (code) {
    case OTelSpanStatusCode.OK:
      return 1;
    case OTelSpanStatusCode.ERROR:
      return 2;
    default:
      return 0;
  }
}

function toAnyValue(value: unknown): OtlpAnyValue {
  if (value == null) {
    return { stringValue: '' };
  }
  if (typeof value === 'string') {
    return { stringValue: value };
  }
  if (typeof value === 'boolean') {
    return { boolValue: value };
  }
  if (typeof value === 'number') {
    return Number.isInteger(value)
      ? { intValue: String(value) }
      : { doubleValue: value };
  }
  if (typeof value === 'bigint') {
    return { intValue: value.toString() };
  }
  if (Array.isArray(value)) {
    return { arrayValue: { values: value.map(toAnyValue) } };
  }
  // Fall back to a stringified representation for nested objects to keep
  // wire compatibility with the Databricks OTLP endpoint (which expects
  // scalar attributes).
  return { stringValue: JSON.stringify(value) };
}

function attributesToOtlp(attributes: Record<string, unknown> | undefined): OtlpKeyValue[] {
  if (!attributes) {
    return [];
  }
  return Object.entries(attributes).map(([key, value]) => ({ key, value: toAnyValue(value) }));
}

interface OtlpResourceLike {
  attributes?: Record<string, unknown>;
}

/**
 * Convert a list of OTel ReadableSpans into an OTLP ExportTraceServiceRequest
 * suitable for POSTing as JSON to `/api/2.0/otel/v1/traces`.
 *
 * Spans are grouped by their (resource, scope) pair to match the OTLP shape.
 */
export function spansToOtlpRequest(spans: OTelReadableSpan[]): OtlpExportTraceServiceRequest {
  if (spans.length === 0) {
    return { resourceSpans: [] };
  }

  const grouped = new Map<
    OtlpResourceLike,
    Map<{ name: string; version?: string }, OtlpSpan[]>
  >();

  for (const span of spans) {
    const resource = (span.resource as OtlpResourceLike | undefined) ?? { attributes: {} };
    // The SDK exposes the instrumentation scope under different fields across
    // OTel versions. Accept either and fall back to the SDK name.
    const scopeRaw =
      (span as { instrumentationScope?: { name: string; version?: string } })
        .instrumentationScope ??
      (span as { instrumentationLibrary?: { name: string; version?: string } })
        .instrumentationLibrary;
    const scope = scopeRaw ?? { name: 'mlflow-tracing' };

    let scopes = grouped.get(resource);
    if (!scopes) {
      scopes = new Map();
      grouped.set(resource, scopes);
    }
    let spansForScope = scopes.get(scope);
    if (!spansForScope) {
      spansForScope = [];
      scopes.set(scope, spansForScope);
    }

    const otelSpan: OtlpSpan = {
      traceId: span.spanContext().traceId,
      spanId: span.spanContext().spanId,
      parentSpanId: span.parentSpanContext?.spanId,
      name: span.name,
      kind: spanKindToOtlp(span.kind),
      startTimeUnixNano: hrTimeToNanoString(span.startTime),
      endTimeUnixNano: hrTimeToNanoString(span.endTime ?? span.startTime),
      attributes: attributesToOtlp(span.attributes as Record<string, unknown> | undefined),
      events: (span.events ?? []).map((event) => ({
        timeUnixNano: hrTimeToNanoString(event.time),
        name: event.name,
        attributes: attributesToOtlp(event.attributes as Record<string, unknown> | undefined),
      })),
      status: {
        code: statusCodeToOtlp(span.status?.code),
        message: span.status?.message,
      },
    };

    spansForScope.push(otelSpan);
  }

  const resourceSpans: OtlpResourceSpans[] = [];
  for (const [resource, scopes] of grouped) {
    const scopeSpans: OtlpScopeSpans[] = [];
    for (const [scope, spansForScope] of scopes) {
      scopeSpans.push({
        scope: { name: scope.name, version: scope.version },
        spans: spansForScope,
      });
    }
    resourceSpans.push({
      resource: { attributes: attributesToOtlp(resource.attributes) },
      scopeSpans,
    });
  }
  return { resourceSpans };
}
