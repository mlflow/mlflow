import { describe, it, expect } from '@jest/globals';
import { deriveClientName } from '../../installInstructions';

describe('deriveClientName', () => {
  it('handles standard reverse-DNS with slash', () => {
    expect(deriveClientName('com.acme/full-mcp-server')).toBe('acme-full-mcp-server');
  });

  it('handles io.github namespace', () => {
    expect(deriveClientName('io.github.user/my-server')).toBe('user-my-server');
  });

  it('handles name without namespace', () => {
    expect(deriveClientName('my-server')).toBe('my-server');
  });

  it('replaces dots and special chars with dashes', () => {
    expect(deriveClientName('com.example/my.special@server')).toBe('example-my-special-server');
  });

  it('collapses consecutive dashes', () => {
    expect(deriveClientName('com.test/a--b')).toBe('test-a-b');
  });

  it('trims leading and trailing dashes', () => {
    expect(deriveClientName('com.test/-server-')).toBe('test-server');
  });

  it('lowercases everything', () => {
    expect(deriveClientName('com.Acme/MyServer')).toBe('acme-myserver');
  });

  it('handles single-segment namespace', () => {
    expect(deriveClientName('acme/server')).toBe('acme-server');
  });

  it('handles empty slug after slash', () => {
    expect(deriveClientName('com.acme/')).toBe('acme');
  });
});
