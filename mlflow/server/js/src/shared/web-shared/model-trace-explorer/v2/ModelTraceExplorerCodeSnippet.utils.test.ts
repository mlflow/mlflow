import { describe, expect, it } from '@jest/globals';

import { getTextPreview, hasMoreLinesThan } from './ModelTraceExplorerCodeSnippet.utils';

describe('bounded snippet previews', () => {
  it.each(['', 'one', '\n', 'a\nb\n', 'café\n日本語\n🙂', 'line\n'.repeat(20000)])(
    'preserves split-based preview and line-count results for case %#',
    (text) => {
      for (const lines of [1, 2, 10, 100]) {
        expect(hasMoreLinesThan(text, lines)).toBe(text.split('\n').length > lines);
        for (const chars of [0, 1, 5, 300, 12000]) {
          expect(getTextPreview(text, lines, chars)).toBe(text.split('\n').slice(0, lines).join('\n').slice(0, chars));
        }
      }
    },
  );
});
