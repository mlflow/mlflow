const JSON_PAYLOAD_KEYS = new Set([
  'mlflow.spanInputs',
  'mlflow.spanOutputs',
  'mlflow.chat.messages',
  'mlflow.chat.tools',
]);
const SLICE_CHARS = 256 * 1024;

// Escapes change JSON spelling, never its parsed value or numeric lexemes.
export function* compactJsonPayloadInSteps(text: string): Generator<void, string> {
  if (!/[\u0100-\uffff]/.test(text)) return text;
  try {
    JSON.parse(text);
  } catch {
    return text;
  }
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const chunks: string[] = [];
  let length = 0;
  for (let start = 0; start < text.length; start += SLICE_CHARS) {
    yield;
    const escaped = text
      .slice(start, start + SLICE_CHARS)
      .replace(/[\u0080-\uffff]/g, (unit) => '\\u' + unit.charCodeAt(0).toString(16).padStart(4, '0'));
    length += escaped.length;
    if (length > text.length * 1.5) return text;
    // Decode ASCII bytes so the result is flat one-byte text, without slices retaining the original.
    chunks.push(decoder.decode(encoder.encode(escaped)));
  }
  return chunks.join('');
}

export function* compactTraceSpanPayloadsInSteps(span: unknown): Generator<void, unknown> {
  if (span === null || typeof span !== 'object' || !('attributes' in span)) return span;
  const { attributes } = span;
  if (!attributes || typeof attributes !== 'object') return span;
  if (Array.isArray(attributes)) {
    let result: unknown[] | undefined;
    for (let index = 0; index < attributes.length; index++) {
      const attribute = attributes[index];
      const value: unknown = attribute?.value?.string_value;
      if (typeof value !== 'string' || !JSON_PAYLOAD_KEYS.has(attribute.key)) continue;
      const compacted = yield* compactJsonPayloadInSteps(value);
      if (compacted === value) continue;
      result ??= attributes.slice();
      result[index] = { ...attribute, value: { ...attribute.value, string_value: compacted } };
    }
    return result ? { ...span, attributes: result } : span;
  }
  let result: Record<string, unknown> | undefined;
  for (const key of JSON_PAYLOAD_KEYS) {
    const value = (attributes as Record<string, unknown>)[key];
    if (typeof value !== 'string') continue;
    const compacted = yield* compactJsonPayloadInSteps(value);
    if (compacted === value) continue;
    result ??= { ...attributes };
    result[key] = compacted;
  }
  return result ? { ...span, attributes: result } : span;
}

export const compactTraceSpanPayloads = (span: unknown): unknown => {
  const steps = compactTraceSpanPayloadsInSteps(span);
  let step = steps.next();
  while (!step.done) step = steps.next();
  return step.value;
};
