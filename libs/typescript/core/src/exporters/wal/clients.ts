/**
 * Daemon-side `MlflowClient` cache.
 */

import { createAuthProvider } from '../../auth';
import { MlflowClient } from '../../clients/client';

const cache = new Map<string, MlflowClient>();

/**
 * Return the cached `MlflowClient` for `trackingUri`, constructing it
 * lazily on first access. Subsequent calls with the same URI return the
 * same instance.
 */
export function clientForUri(trackingUri: string): MlflowClient {
  const cached = cache.get(trackingUri);
  if (cached !== undefined) {
    return cached;
  }
  const authProvider = createAuthProvider({ trackingUri });
  const client = new MlflowClient({ trackingUri, authProvider });
  cache.set(trackingUri, client);
  return client;
}

export function clearClientCache(): void {
  cache.clear();
}
