/** Coerces an A2UI DynamicString prop value to a plain string for rendering. */
export const asString = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));

/** JSON.stringify that never throws — falls back to a JSON string for String(value). */
export const safeJsonStringify = (value: unknown): string => {
  try {
    return JSON.stringify(value ?? null);
  } catch {
    return JSON.stringify(String(value));
  }
};
