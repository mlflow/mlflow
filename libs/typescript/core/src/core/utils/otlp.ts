/**
 * OTLP (OpenTelemetry Protocol) utilities for parsing proto format data.
 *
 * This module provides utilities for decoding OpenTelemetry protobuf format
 * data returned by MLflow APIs.
 */

/**
 * Proto-format attribute value as returned by OpenTelemetry APIs.
 * Mirrors the OTel protobuf AnyValue message structure.
 */
export interface ProtoAnyValue {
  string_value?: string;
  int_value?: number;
  double_value?: number;
  bool_value?: boolean;
  bytes_value?: string; // base64 encoded
  kvlist_value?: { values: ProtoKeyValue[] };
  array_value?: { values: ProtoAnyValue[] };
}

/**
 * Proto-format key-value pair.
 */
export interface ProtoKeyValue {
  key: string;
  value: ProtoAnyValue;
}

/**
 * Proto-format span as returned by the GetTrace API.
 */
export interface ProtoSpan {
  name: string;
  trace_id: string;
  span_id: string;
  parent_span_id?: string;
  start_time_unix_nano: number;
  end_time_unix_nano?: number | null;
  status?: {
    code: string;
    message?: string;
  };
  attributes?: ProtoKeyValue[];
  events?: ProtoSpanEvent[];
}

/**
 * Proto-format span event.
 */
export interface ProtoSpanEvent {
  name: string;
  time_unix_nano: number;
  attributes?: ProtoKeyValue[];
}

/**
 * Decode an OpenTelemetry protobuf AnyValue to a plain JavaScript value.
 *
 *
 * @param value - The proto AnyValue to decode
 * @returns The decoded JavaScript value
 */
export function decodeProtoAnyValue(value: ProtoAnyValue | undefined | null): unknown {
  if (!value) {
    return null;
  }

  // Handle simple types
  if (value.string_value !== undefined) {
    return value.string_value;
  }
  if (value.int_value !== undefined) {
    return value.int_value;
  }
  if (value.double_value !== undefined) {
    return value.double_value;
  }
  if (value.bool_value !== undefined) {
    return value.bool_value;
  }
  if (value.bytes_value !== undefined) {
    return value.bytes_value;
  }

  // Handle complex types with recursion
  if (value.kvlist_value?.values) {
    const result: Record<string, unknown> = {};
    for (const kv of value.kvlist_value.values) {
      result[kv.key] = decodeProtoAnyValue(kv.value);
    }
    return result;
  }

  if (value.array_value?.values) {
    return value.array_value.values.map(decodeProtoAnyValue);
  }

  return null;
}

/**
 * Decode proto-format attributes array to a flat Record.
 *
 * @param attributes - Array of proto key-value pairs
 * @returns Flat attributes object
 */
export function decodeProtoAttributes(
  attributes: ProtoKeyValue[] | undefined
): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  if (!attributes) {
    return result;
  }
  for (const attr of attributes) {
    result[attr.key] = decodeProtoAnyValue(attr.value);
  }
  return result;
}
