import { ReadableSpan as OTelReadableSpan } from '@opentelemetry/sdk-trace-base';
import { ProtobufTraceSerializer } from '@opentelemetry/otlp-transformer';

function decodeMlflowSpanAttributes(
  attributes: OTelReadableSpan['attributes'],
): Record<string, unknown> {
  const decoded: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(attributes)) {
    if (typeof value === 'string' && value.startsWith('"')) {
      try {
        decoded[key] = JSON.parse(value);
      } catch {
        decoded[key] = value;
      }
    } else {
      decoded[key] = value;
    }
  }
  return decoded;
}

/**
 * Return a view of `span` whose `attributes` are decoded (see
 * {@link decodeMlflowSpanAttributes}) while every other property/method is
 * delegated to the original span untouched.
 */
function spanWithDecodedAttributes(span: OTelReadableSpan): OTelReadableSpan {
  const decoded = decodeMlflowSpanAttributes(span.attributes);
  return new Proxy(span, {
    get(target, prop) {
      if (prop === 'attributes') {
        return decoded;
      }
      const value: unknown = Reflect.get(target, prop);
      return typeof value === 'function' ? (value.bind(target) as unknown) : value;
    },
  });
}

/**
 * Serialize live OTel spans into an OTLP `ExportTraceServiceRequest`
 * protobuf, ready for POSTing to the tracking store's span-ingestion
 * endpoint. Returns `undefined` when nothing can be serialized.
 */
export function serializeSpansToOtlpBytes(spans: OTelReadableSpan[]): Uint8Array | undefined {
  const serializable = spans.filter((s) => s?.resource != null && s?.instrumentationScope != null);
  if (serializable.length === 0) {
    return undefined;
  }
  try {
    const bytes = ProtobufTraceSerializer.serializeRequest(
      serializable.map(spanWithDecodedAttributes),
    );
    if (!bytes || bytes.length === 0) {
      return undefined;
    }
    return bytes;
  } catch (err) {
    console.warn(
      '[mlflow] Failed to serialize spans to OTLP protobuf; ' +
        'spans will not appear in DB-backed span metrics.',
      err,
    );
    return undefined;
  }
}

/**
 * {@link serializeSpansToOtlpBytes} wrapped as base64, for storing on a
 * WAL record.
 */
export function serializeSpansToOtlpBase64(spans: OTelReadableSpan[]): string | undefined {
  const bytes = serializeSpansToOtlpBytes(spans);
  return bytes ? Buffer.from(bytes).toString('base64') : undefined;
}
