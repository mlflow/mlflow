const JSON_PAYLOAD_KEYS = new Set([
  'mlflow.spanInputs',
  'mlflow.spanOutputs',
  'mlflow.chat.messages',
  'mlflow.chat.tools',
]);

// Escapes change JSON spelling, never its parsed value or numeric lexemes.
export const compactJsonPayload = (text: string): string => {
  if (!/[\u0100-\uffff]/.test(text)) return text;
  try {
    JSON.parse(text);
  } catch {
    return text;
  }
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const escaped = text.replace(/[\u0100-\uffff]/g, (unit) => '\\u' + unit.charCodeAt(0).toString(16).padStart(4, '0'));
  if (escaped.length > text.length * 1.5) return text;
  // Flatten Latin-1-compatible text without retaining the original two-byte representation.
  return decoder.decode(encoder.encode(escaped));
};

export const compactTraceSpanPayloads = (span: unknown): unknown => {
  if (span === null || typeof span !== 'object' || !('attributes' in span)) return span;
  const { attributes } = span;
  if (!attributes || typeof attributes !== 'object') return span;
  if (Array.isArray(attributes)) {
    let result: unknown[] | undefined;
    for (let index = 0; index < attributes.length; index++) {
      const attribute = attributes[index];
      const value: unknown = attribute?.value?.string_value;
      if (typeof value !== 'string' || !JSON_PAYLOAD_KEYS.has(attribute.key)) continue;
      const compacted = compactJsonPayload(value);
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
    const compacted = compactJsonPayload(value);
    if (compacted === value) continue;
    result ??= { ...attributes };
    result[key] = compacted;
  }
  return result ? { ...span, attributes: result } : span;
};
