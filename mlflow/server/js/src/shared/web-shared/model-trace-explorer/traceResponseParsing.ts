export type TraceResponseParseOptions = { compactPayloads?: boolean };
export type TraceResponseRequest = { id: number; body: ArrayBuffer } & TraceResponseParseOptions;
export type TraceResponseMessage =
  | { type: 'ready' }
  | { type: 'span-chunk'; id: number; index: number; spans: unknown[] }
  | { type: 'parsed'; id: number; value: unknown }
  | { type: 'failed'; id: number; name: string; message: string };

export const parseTraceResponseBody = (body: ArrayBuffer): unknown => JSON.parse(new TextDecoder().decode(body));

type SpanContainer = { spans: unknown[] };
const hasSpans = (value: unknown): value is SpanContainer =>
  value !== null && typeof value === 'object' && 'spans' in value && Array.isArray(value.spans);

// Only endpoint envelopes own span lists; nested user payloads named `spans` are left alone.
export const getTraceResponseSpanContainers = (value: unknown): SpanContainer[] => {
  if (hasSpans(value)) return [value];
  if (value === null || typeof value !== 'object') return [];
  if ('trace' in value && hasSpans(value.trace)) return [value.trace];
  return 'traces' in value && Array.isArray(value.traces) ? value.traces.filter(hasSpans) : [];
};

export const countTraceSpanAttributeChars = (span: unknown): number => {
  if (span === null || typeof span !== 'object' || !('attributes' in span)) return 0;
  const { attributes } = span;
  if (!attributes || typeof attributes !== 'object') return 0;
  const pending: unknown[] = Array.isArray(attributes)
    ? attributes.map((attribute) => attribute?.value)
    : Object.values(attributes);
  let chars = 0;
  // OSS OTLP payloads can be native nested objects, not just JSON string attributes.
  while (pending.length) {
    const value = pending.pop();
    if (typeof value === 'string') chars += value.length;
    else if (value && typeof value === 'object') {
      for (const child of Object.values(value)) pending.push(child);
    }
  }
  return chars;
};
