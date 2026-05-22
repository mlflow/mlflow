import { clearClientCache, clientForUri } from '../../../src/exporters/wal/clients';
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
});
