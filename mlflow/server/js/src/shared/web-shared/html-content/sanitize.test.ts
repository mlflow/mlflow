import { describe, it, expect } from '@jest/globals';

import sanitize from './sanitize';

describe('sanitize', () => {
  describe('target attribute', () => {
    it('should add rel="noopener noreferrer" to links with target="_blank"', () => {
      const input = '<a href="/test" target="_blank">Link</a>';
      const result = sanitize(input);

      expect(result).toEqual('<a target="_blank" href="/test" rel="noopener noreferrer">Link</a>');
    });

    it('should remove target attribute if not "_blank"', () => {
      const input = '<a href="/test" target="_self">Link</a>';
      const result = sanitize(input);

      expect(result).toEqual('<a href="/test">Link</a>');
    });
  });
});
