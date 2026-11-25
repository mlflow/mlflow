import { ReadableSpan as OTelReadableSpan } from '@opentelemetry/sdk-trace-base';
import { ProtobufTraceSerializer } from '@opentelemetry/otlp-transformer';
import { isDatabricksUri } from '../auth';
import { MlflowHttpError } from '../clients/utils';
import { MAX_OTLP_REQUEST_BYTES } from '../core/constants';
import { withoutRemoteParent } from '../core/utils';

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
 * Whether the tracking store at `trackingUri` accepts OTLP spans through the
 * OSS `/v1/traces` endpoint. Databricks tracking servers only support logging
 * spans to a Unity Catalog table, so span ingestion must be skipped for
 * `databricks` / `databricks://...` URIs (mirrors the Python exporter).
 */
export function supportsOtlpSpanIngestion(trackingUri: string): boolean {
  return !isDatabricksUri(trackingUri);
}

/**
 * Whether `err` means the tracking server does not implement OTLP span
 * ingestion at all, as opposed to rejecting this one request. Callers latch on
 * it so they stop POSTing (and warning) for every trace.
 *
 * - 405 / 501: the route exists but the backing store cannot log spans (e.g. a
 *   FileStore-backed server).
 * - 404 without an MLflow `error_code`: the server has no `/v1/traces` route.
 *   The route itself also maps MLflow errors to their HTTP status, so a 404
 *   that carries an `error_code` (e.g. `RESOURCE_DOES_NOT_EXIST`) is about this
 *   request only and must not disable ingestion.
 */
export function isSpanIngestionUnsupportedError(err: unknown): err is MlflowHttpError {
  if (!(err instanceof MlflowHttpError)) {
    return false;
  }
  if (err.status === 405 || err.status === 501) {
    return true;
  }
  return err.status === 404 && err.errorCode === undefined;
}

/**
 * Serialize live OTel spans into an OTLP `ExportTraceServiceRequest`
 * protobuf, ready for POSTing to the tracking store's span-ingestion
 * endpoint. Returns `undefined` when the destination does not support OTLP
 * span ingestion or nothing can be serialized.
 */
export function serializeSpansToOtlpBytes(
  spans: OTelReadableSpan[],
  trackingUri: string,
): Uint8Array | undefined {
  if (!supportsOtlpSpanIngestion(trackingUri)) {
    return undefined;
  }
  const serializable = spans.filter((s) => s?.resource != null && s?.instrumentationScope != null);
  if (serializable.length === 0) {
    return undefined;
  }
  try {
    const bytes = ProtobufTraceSerializer.serializeRequest(
      serializable.map((s) => spanWithDecodedAttributes(withoutRemoteParent(s))),
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
export function serializeSpansToOtlpBase64(
  spans: OTelReadableSpan[],
  trackingUri: string,
): string | undefined {
  const bytes = serializeSpansToOtlpBytes(spans, trackingUri);
  return bytes ? Buffer.from(bytes).toString('base64') : undefined;
}

/**
 * Best-effort send of serialized spans to the tracking store. Skips when the
 * destination does not support OTLP span ingestion, when there is no payload,
 * or when the payload exceeds {@link MAX_OTLP_REQUEST_BYTES}. Returns `true`
 * when `send` was invoked.
 */
export async function sendOtlpSpansIfSupported(
  trackingUri: string,
  experimentId: string,
  bytes: Uint8Array | undefined,
  send: (experimentId: string, bytes: Uint8Array) => Promise<void>,
  context: string,
  warn: (message: string) => void = console.warn,
): Promise<boolean> {
  if (!bytes || !supportsOtlpSpanIngestion(trackingUri)) {
    return false;
  }
  if (bytes.length > MAX_OTLP_REQUEST_BYTES) {
    warn(
      `[mlflow] OTLP span payload for ${context} is ${bytes.length} bytes, exceeding the ` +
        `${MAX_OTLP_REQUEST_BYTES}-byte request limit; skipping span ingestion ` +
        '(spans remain in the trace artifact only).',
    );
    return false;
  }
  await send(experimentId, bytes);
  return true;
}
