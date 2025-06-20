import type { HrTime } from '@opentelemetry/api';
import { LiveSpan } from '../entities/span';

/**
 * OpenTelemetry Typescript SDK uses a unique timestamp format `HrTime` to represent
 * timestamps. This function converts a timestamp in nanoseconds to an `HrTime`
 * Supports both number and BigInt for large timestamps
 * Ref: https://github.com/open-telemetry/opentelemetry-js/blob/a9fc600f2bd7dbf9345ec14e4421f1cc034f1f9c/api/src/common/Time.ts#L17-L30C13
 * @param nanoseconds The timestamp in nanoseconds (number or BigInt)
 * @returns The timestamp in `HrTime` format
 */
export function convertNanoSecondsToHrTime(nanoseconds: number | bigint): HrTime {
  // Handle both number and BigInt inputs
  if (typeof nanoseconds === 'bigint') {
    // Use BigInt arithmetic to maintain precision
    const seconds = Number(nanoseconds / 1_000_000_000n);
    const nanos = Number(nanoseconds % 1_000_000_000n);
    return [seconds, nanos] as HrTime;
  }
  // For regular numbers, use standard arithmetic
  return [Math.floor(nanoseconds / 1e9), nanoseconds % 1e9] as HrTime;
}

/**
 * Convert HrTime to nanoseconds as BigInt
 * @param hrTime HrTime tuple [seconds, nanoseconds]
 * @returns BigInt nanoseconds
 */
export function convertHrTimeToNanoSeconds(hrTime: HrTime): bigint {
  return BigInt(hrTime[0]) * 1_000_000_000n + BigInt(hrTime[1]);
}

/**
 * Convert HrTime to milliseconds
 * @param hrTime HrTime tuple [seconds, nanoseconds]
 * @returns Milliseconds
 */
export function convertHrTimeToMs(hrTime: HrTime): number {
  return hrTime[0] * 1e3 + hrTime[1] / 1e6;
}

/**
 * Convert a hex span ID to base64 format for JSON serialization
 * Following Python implementation: _encode_span_id_to_byte
 * @param spanId Hex string span ID (16 chars)
 * @returns Base64 encoded span ID
 */
export function encodeSpanIdToBase64(spanId: string): string {
  // Convert hex string to bytes (8 bytes for span ID)
  const bytes = new Uint8Array(8);

  // Parse hex string (add padding to 16 chars)
  const hexStr = spanId.padStart(16, '0');
  for (let i = 0; i < 8; i++) {
    bytes[i] = parseInt(hexStr.substring(i * 2, i * 2 + 2), 16);
  }

  // Convert to base64
  return Buffer.from(bytes).toString('base64');
}

/**
 * Convert a hex span ID to base64 format for JSON serialization
 * Following Python implementation: _encode_trace_id_to_byte
 * @param spanId Hex string span ID (32 chars)
 * @returns Base64 encoded span ID
 */
export function encodeTraceIdToBase64(traceId: string): string {
  // Convert hex string to bytes (16 bytes for trace ID)
  const bytes = new Uint8Array(16);

  // Parse hex string (add padding to 32 chars)
  const hexStr = traceId.padStart(32, '0');
  for (let i = 0; i < 16; i++) {
    bytes[i] = parseInt(hexStr.substring(i * 2, i * 2 + 2), 16);
  }

  // Convert to base64
  return Buffer.from(bytes).toString('base64');
}

/**
 * Convert a base64 span ID back to hex format
 * Following Python implementation: _decode_id_from_byte
 * @param base64SpanId Base64 encoded span ID
 * @returns Hex string span ID
 */
export function decodeIdFromBase64(base64SpanId: string): string {
  // Decode from base64
  const bytes = Buffer.from(base64SpanId, 'base64');

  // Convert to hex string
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}


/**
 * Deduplicate span names in the trace data by appending an index number to the span name.
 *
 * This is only applied when there are multiple spans with the same name. The span names
 * are modified in place to avoid unnecessary copying.
 *
 * Examples:
 *   ["red", "red"] -> ["red_1", "red_2"]
 *   ["red", "red", "blue"] -> ["red_1", "red_2", "blue"]
 *
 * @param spans A list of spans to deduplicate
 */
export function deduplicateSpanNamesInPlace(spans: LiveSpan[]): void {
  // Count occurrences of each span name
  const spanNameCounter = new Map<string, number>();

  for (const span of spans) {
    const name = span.name;
    spanNameCounter.set(name, (spanNameCounter.get(name) || 0) + 1);
  }

  // Apply renaming only for duplicated spans
  const spanNameIndexes = new Map<string, number>();
  for (const [name, count] of spanNameCounter.entries()) {
    if (count > 1) {
      spanNameIndexes.set(name, 1);
    }
  }

  // Add index to the duplicated span names
  for (const span of spans) {
    const name = span.name;
    const currentIndex = spanNameIndexes.get(name);

    if (currentIndex !== undefined) {
      // Modify the span name in place by accessing the internal OpenTelemetry span
      // The 'name' field is readonly in OTel but we need to jail-break it here
      // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      (span._span as any).name = `${name}_${currentIndex}`;
      spanNameIndexes.set(name, currentIndex + 1);
    }
  }
}
