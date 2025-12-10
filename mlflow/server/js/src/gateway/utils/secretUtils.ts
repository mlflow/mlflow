import type { SecretInfo } from '../types';

/**
 * Parsed masked value entry: [key, value] tuple.
 * For single-value secrets, key is empty string.
 */
export type MaskedValueEntry = [string, string];

/**
 * Parse auth_config from a secret, handling both pre-parsed objects
 * and JSON strings stored in auth_config_json.
 *
 * @param secret - The secret to parse auth_config from
 * @returns Parsed auth_config object or null
 */
export function parseAuthConfig(secret: SecretInfo | undefined | null): Record<string, unknown> | null {
  if (!secret) return null;

  // Parse auth_config_json if it exists, otherwise use auth_config
  if (secret.auth_config_json) {
    try {
      return JSON.parse(secret.auth_config_json) as Record<string, unknown>;
    } catch {
      // Invalid JSON, ignore
      return null;
    }
  }
  return secret.auth_config ?? null;
}

/**
 * Parse masked_value from a secret into key-value pairs.
 *
 * Handles multiple formats:
 * - Single key: "sk-****xyz" -> [['', 'sk-****xyz']]
 * - Multiple keys: "{aws_access_key_id: AKI****xyz, aws_secret_access_key: wJa****123}"
 *   -> [['aws_access_key_id', 'AKI****xyz'], ['aws_secret_access_key', 'wJa****123']]
 * - Legacy JSON format: '{"a...df"}' -> [['', 'a...df']]
 *
 * @param secret - The secret to parse masked_value from
 * @returns Array of [key, value] tuples, or null if no masked_value
 */
export function parseMaskedValues(secret: SecretInfo | undefined | null): MaskedValueEntry[] | null {
  if (!secret?.masked_value) return null;

  const value = secret.masked_value.trim();

  // Check if it's in the {key: value, key: value} format (multi-key secrets)
  if (value.startsWith('{') && value.endsWith('}')) {
    const inner = value.slice(1, -1); // Remove { and }

    // Check if it contains ": " which indicates key-value pairs
    // vs a single masked value like {"a...df"} which is a bug in how we store secrets
    if (inner.includes(': ')) {
      // Parse the custom format: {key1: value1, key2: value2}
      const pairs: MaskedValueEntry[] = [];

      // Split by comma, but be careful with values that might contain commas
      // The format is "key: value" pairs separated by ", "
      const parts = inner.split(', ');
      for (const part of parts) {
        const colonIndex = part.indexOf(': ');
        if (colonIndex > 0) {
          const key = part.slice(0, colonIndex).trim();
          const val = part.slice(colonIndex + 2).trim();
          pairs.push([key, val]);
        }
      }

      if (pairs.length > 0) {
        return pairs;
      }
    }

    // It's a single masked value wrapped in braces (legacy/bug format like {"a...df"})
    // Strip surrounding quotes if present and show the inner content
    const cleaned = inner.replace(/^["']|["']$/g, '');
    return [['', cleaned]];
  }

  // Single value - just show the masked value directly without a key label
  return [['', value]];
}

/**
 * Check if parsed masked values represent a single key (no field name).
 *
 * @param maskedValues - Parsed masked values from parseMaskedValues
 * @returns true if this is a single-value secret without field names
 */
export function isSingleMaskedValue(maskedValues: MaskedValueEntry[] | null): boolean {
  return maskedValues !== null && maskedValues.length === 1 && maskedValues[0][0] === '';
}

/**
 * Format masked value for simple display (strips JSON brackets/quotes).
 * Use this for contexts where you just want the raw masked value as a string.
 *
 * @param maskedValue - Raw masked_value string from secret
 * @returns Cleaned masked value string
 */
export function formatMaskedValueSimple(maskedValue: string | undefined | null): string {
  if (!maskedValue) return '';
  return maskedValue.replace(/^[{"\s]+|[}"\s]+$/g, '');
}
