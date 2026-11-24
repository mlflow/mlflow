import {
  clearClientCache,
  clearClientForUri,
  clientForUri,
} from '../../../src/exporters/wal/clients';
import { MlflowClient } from '../../../src/clients/client';

describe('wal/clients', () => {
  beforeEach(() => {
    clearClientCache();
  });

  it('returns the same MlflowClient instance for repeat calls with the same URI', () => {
    const a = clientForUri('http://localhost:5000');
    const b = clientForUri('http://localhost:5000');
    expect(a).toBeInstanceOf(MlflowClient);
    expect(b).toBe(a);
  });

  it('returns distinct MlflowClient instances for different URIs', () => {
    const local = clientForUri('http://localhost:5000');
    const remote = clientForUri('http://other-host:5000');
    expect(local).not.toBe(remote);
    expect(local.getHost()).toBe('http://localhost:5000');
    expect(remote.getHost()).toBe('http://other-host:5000');
  });

  it('rebuilds the client after the cache is cleared', () => {
    const before = clientForUri('http://localhost:5000');
    clearClientCache();
    const after = clientForUri('http://localhost:5000');
    expect(after).not.toBe(before);
    expect(after).toBeInstanceOf(MlflowClient);
  });

  it('isolates state for different tracking URIs after a clear', () => {
    const a1 = clientForUri('http://a:5000');
    const b1 = clientForUri('http://b:5000');
    clearClientCache();
    const a2 = clientForUri('http://a:5000');
    const b2 = clientForUri('http://b:5000');
    expect(a2).not.toBe(a1);
    expect(b2).not.toBe(b1);
    expect(a2).not.toBe(b2);
  });

  describe('clearClientForUri', () => {
    it('evicts only the targeted URI and rebuilds it on next access', () => {
      // Targeted eviction is the upload loop's auth-error retry path:
      // we want the *specific* stale client gone, not every other URI's
      // working cache entry blown away as collateral damage.
      const a = clientForUri('http://a:5000');
      const b = clientForUri('http://b:5000');

      clearClientForUri('http://a:5000');

      const aAfter = clientForUri('http://a:5000');
      const bAfter = clientForUri('http://b:5000');

      expect(aAfter).not.toBe(a);
      expect(bAfter).toBe(b);
    });

    it('is a silent no-op for a URI that was never cached', () => {
      // The upload loop calls clearClientForUri unconditionally before
      // re-resolving via the factory; it must not throw if the cache
      // entry was already evicted (e.g. by a prior auth-retry in a
      // sibling group, or by clearClientCache() during shutdown).
      expect(() => clearClientForUri('http://never-cached:5000')).not.toThrow();
    });
  });
});
