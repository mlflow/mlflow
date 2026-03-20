import type { HrTime } from '@opentelemetry/api';
import { Span } from '../entities/span';
import { SpanAttributeKey } from '../constants';
import { TokenUsage } from '../entities/trace_info';

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
  return Math.floor(hrTime[0] * 1e3 + hrTime[1] / 1e6);
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
 * Map function arguments to an object with parameter names as keys
 * @param func The function to extract parameter names from
 * @param args The arguments passed to the function
 * @returns Object mapping parameter names to argument values
 */
export function mapArgsToObject(func: Function, args: any[]): Record<string, any> {
  const paramNames = getParameterNames(func);

  // If we can't extract parameter names, return the args as an array
  if (paramNames.length === 0) {
    return args.length === 0 ? {} : { args };
  }

  const result: Record<string, any> = {};

  paramNames.forEach((name, index) => {
    if (index < args.length) {
      result[name] = args[index];
    }
  });

  return result;
}

/**
 * Extract parameter names from a function using string parsing
 * @param func The function to extract parameter names from
 * @returns Array of parameter names
 */
function getParameterNames(func: Function): string[] {
  const funcStr = func.toString();

  // Handle arrow functions and regular functions
  let paramMatch: RegExpMatchArray | null;

  // Try arrow function pattern: (a, b) => or a =>
  const arrowMatch = funcStr.match(/^[^(]*\(?([^)=]*)\)?\s*=>/);
  if (arrowMatch) {
    const params = arrowMatch[1].trim();
    if (!params) {
      return [];
    }

    // Handle single parameter without parentheses
    if (!params.includes(',') && !funcStr.includes('(')) {
      return [params.trim()];
    }

    paramMatch = ['', params];
  } else {
    // Try regular function pattern: function name(a, b) or (a, b)
    paramMatch = funcStr.match(/(?:function\s*[^(]*)??\(([^)]*)\)/);
  }

  if (!paramMatch?.[1]) {
    return [];
  }

  // Split parameters while handling nested brackets/braces
  const params = [];
  let current = '';
  let depth = 0;

  const paramStr = paramMatch[1];
  for (let i = 0; i < paramStr.length; i++) {
    const char = paramStr[i];

    if (char === '{' || char === '[') {
      depth++;
    } else if (char === '}' || char === ']') {
      depth--;
    } else if (char === ',' && depth === 0) {
      params.push(current.trim());
      current = '';
      continue;
    }
    current += char;
  }
  if (current.trim()) {
    params.push(current.trim());
  }

  return params
    .map((param) => {
      let name = param.trim();

      // Skip destructured parameters
      if (name.includes('{') || name.includes('[')) {
        return null;
      }

      name = name.split('=')[0].trim(); // Remove default values: a = 5
      name = name.split(':')[0].trim(); // Remove type annotations: a: number
      if (name.startsWith('...')) {
        return null; // Ignore rest operator: ...args
      }
      return name;
    })
    .filter((name): name is string => name != null && name !== '');
}

/**
 * Aggregate token usage information from all spans in a trace.
 *
 * This function iterates through all spans and extracts token usage from the
 * SpanAttributeKey.TOKEN_USAGE attribute. It avoids double-counting token usage
 * when both parent and child spans have usage data (e.g., LangChain ChatOpenAI + OpenAI tracing).
 *
 * @param spans - Array of spans to aggregate usage from
 * @returns Aggregated token usage or null if no usage data exists
 */
export function aggregateUsageFromSpans(spans: Span[]): TokenUsage | null {
  const totalUsage: TokenUsage = {
    input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
  };
  let hasUsageData = false;

  // Build parent-children map for DFS traversal (mirrors Python SDK)
  const spanById = new Map<string, Span>();
  const childrenMap = new Map<string, Span[]>();
  const roots: Span[] = [];

  for (const span of spans) {
    spanById.set(span.spanId, span);
    if (span.parentId && spanById.has(span.parentId)) {
      const children = childrenMap.get(span.parentId) || [];
      children.push(span);
      childrenMap.set(span.parentId, children);
    } else {
      roots.push(span);
    }
  }

  function dfs(span: Span, ancestorHasData: boolean): void {
    const tokenUsageAttr = span.attributes[SpanAttributeKey.TOKEN_USAGE];
    const spanHasData = tokenUsageAttr != null;

    if (spanHasData && !ancestorHasData) {
      const tokenUsage = tokenUsageAttr as TokenUsage;
      totalUsage.input_tokens += tokenUsage.input_tokens || 0;
      totalUsage.output_tokens += tokenUsage.output_tokens || 0;
      totalUsage.total_tokens += tokenUsage.total_tokens || 0;
      // Optional cache keys — only include when present
      if (tokenUsage.cache_read_input_tokens != null) {
        totalUsage.cache_read_input_tokens =
          (totalUsage.cache_read_input_tokens || 0) + tokenUsage.cache_read_input_tokens;
      }
      if (tokenUsage.cache_creation_input_tokens != null) {
        totalUsage.cache_creation_input_tokens =
          (totalUsage.cache_creation_input_tokens || 0) + tokenUsage.cache_creation_input_tokens;
      }
      hasUsageData = true;
    }

    for (const child of childrenMap.get(span.spanId) || []) {
      dfs(child, ancestorHasData || spanHasData);
    }
  }

  for (const root of roots) {
    dfs(root, false);
  }

  return hasUsageData ? totalUsage : null;
}
