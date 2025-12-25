import type { SecretInfo } from '../types';

export function parseAuthConfig(secret: SecretInfo | undefined | null): Record<string, string> | null {
  if (!secret) return null;
  return secret.auth_config ?? null;
}

export function isSingleMaskedValue(maskedValue: Record<string, string>): boolean {
  return Object.keys(maskedValue).length === 1;
}
