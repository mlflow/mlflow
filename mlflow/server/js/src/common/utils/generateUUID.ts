/**
 * Generates a UUID that works in both secure (HTTPS) and insecure (HTTP) contexts.
 * crypto.randomUUID() requires a secure context, so we fall back to crypto.getRandomValues()
 * which is available in all contexts.
 */
export function generateUUID(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    try {
      return crypto.randomUUID();
    } catch {
      // Falls through to fallback
    }
  }

  // Fallback using crypto.getRandomValues() which works in insecure contexts
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);

  // Set version (4) and variant (RFC4122) bits
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;

  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

