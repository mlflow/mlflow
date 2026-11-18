export type TraceResponseRequest = { id: number; body: ArrayBuffer };
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
  if (Array.isArray(attributes)) {
    return attributes.reduce((total, attribute) => {
      const value: unknown = attribute?.value?.string_value;
      return total + (typeof value === 'string' ? value.length : 0);
    }, 0);
  }
  return Object.values(attributes).reduce<number>(
    (total, value) => total + (typeof value === 'string' ? value.length : 0),
    0,
  );
};
