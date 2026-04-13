import { describe, it, expect } from '@jest/globals';
import { generateCopyName } from './gatewayUtils';

describe('gatewayUtils', () => {
  describe('generateCopyName', () => {
    it('generates a basic copy name', () => {
      expect(generateCopyName('my-endpoint', [])).toBe('my-endpoint-copy-1');
    });

    it('avoids conflicts with existing names', () => {
      expect(generateCopyName('my-endpoint', ['my-endpoint-copy-1'])).toBe('my-endpoint-copy-2');
    });

    it('skips multiple conflicting names', () => {
      const existing = ['my-endpoint-copy-1', 'my-endpoint-copy-2', 'my-endpoint-copy-3'];
      expect(generateCopyName('my-endpoint', existing)).toBe('my-endpoint-copy-4');
    });

    it('does not conflict with unrelated names', () => {
      expect(generateCopyName('my-endpoint', ['other-endpoint', 'another-copy-1'])).toBe('my-endpoint-copy-1');
    });

    it('handles the original name being in the list', () => {
      expect(generateCopyName('my-endpoint', ['my-endpoint'])).toBe('my-endpoint-copy-1');
    });
  });
});
