import type { SecretInfo } from '../types';

export function parseAuthConfig(secret: SecretInfo | undefined | null): Record<string, string> | null {
  if (!secret) return null;
  return secret.auth_config ?? null;
}

// Secret fields that hold JSON documents (e.g. GCP service account key files) rather than API key
// strings; these may legitimately contain non-ASCII characters.
const JSON_CREDENTIAL_SECRET_FIELDS = new Set(['vertex_credentials']);

const NON_ASCII_PATTERN = /[\u0080-\u{10FFFF}]/u;

export function isApiKeySecretField(fieldName: string): boolean {
  return !JSON_CREDENTIAL_SECRET_FIELDS.has(fieldName);
}

export function containsNonAsciiCharacters(value: string): boolean {
  return NON_ASCII_PATTERN.test(value);
}

/**
 * Returns true if any API key field contains non-ASCII characters. Masked API key inputs keep the
 * OS IME active, so keystrokes typed in a CJK input mode are silently composed into non-Latin text.
 */
export function hasNonAsciiApiKey(secretFields: Record<string, string> | undefined): boolean {
  return Object.entries(secretFields ?? {}).some(
    ([fieldName, value]) => isApiKeySecretField(fieldName) && containsNonAsciiCharacters(value),
  );
}
