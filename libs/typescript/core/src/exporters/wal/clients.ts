/**
 * Daemon-side `MlflowClient` cache.
 */

import { createAuthProvider } from '../../auth';
import { MlflowClient } from '../../clients/client';

const cache = new Map<string, MlflowClient>();

/**
 * Return the cached `MlflowClient` for `trackingUri`, constructing it
 * lazily on first access. Subsequent calls with the same URI return the
 * same instance unless an explicit {@link clearClientForUri} happened
 * in between (typically driven by the upload loop's auth-error retry).
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

/**
 * Drop the cached client for `trackingUri` only, leaving other URIs'
 * entries intact. Called by the upload loop on 401 / 403 so the next
 * `clientForUri(uri)` call rebuilds the client with freshly-resolved
 * credentials. No-op if no entry exists for `trackingUri`.
 */
export function clearClientForUri(trackingUri: string): void {
  cache.delete(trackingUri);
}

/**
 * Drop all cached clients. Intended for daemon shutdown paths and
 * tests; not part of the steady-state batch loop.
 */
export function clearClientCache(): void {
  cache.clear();
}
