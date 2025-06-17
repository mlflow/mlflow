import type { HrTime } from '@opentelemetry/api';

/**
 * OpenTelemetry Typescript SDK uses a unique timestamp format `HrTime` to represent
 * timestamps. This function converts a timestamp in nanoseconds to an `HrTime`
 * Supports both number and BigInt for large timestamps
 * Ref: https://github.com/open-telemetry/opentelemetry-js/blob/a9fc600f2bd7dbf9345ec14e4421f1cc034f1f9c/api/src/common/Time.ts#L17-L30C13
 * @param nanoseconds The timestamp in nanoseconds (number or BigInt)
 * @returns The timestamp in `HrTime` format
 */
export function convertNanoSecondsToHrTime(nanoseconds: number | bigint): HrTime {
  // Convert BigInt to number safely for HrTime (OpenTelemetry uses number arrays)
  const nanos = typeof nanoseconds === 'bigint' ? Number(nanoseconds) : nanoseconds;
  return [Math.floor(nanos / 1e9), nanos % 1e9] as HrTime;
}

export function convertHrTimeToNanoSeconds(hrTime: HrTime): number {
  return hrTime[0] * 1e9 + hrTime[1];
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

  // Parse hex string (remove any padding to 16 chars)
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

  // Parse hex string (remove any padding to 32 chars)
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
