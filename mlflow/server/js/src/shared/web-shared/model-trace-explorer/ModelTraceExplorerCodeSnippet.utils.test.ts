import { describe, expect, it } from '@jest/globals';

import { getTextPreview, hasMoreLinesThan } from './ModelTraceExplorerCodeSnippet.utils';

describe('snippet preview boundaries', () => {
  it.each(['', 'one', 'one\n', '\n\n', 'café\n日本語 🙂\nlast', 'a'.repeat(1000), 'x\n'.repeat(5000)])(
    'preserves existing line and character boundaries for %j',
    (text) => {
      for (const maxLines of [1, 2, 10, 100]) {
        expect(hasMoreLinesThan(text, maxLines)).toBe(text.split('\n').length > maxLines);
        for (const maxChars of [1, 5, 300, 12000]) {
          expect(getTextPreview(text, maxLines, maxChars)).toBe(
            text.split('\n').slice(0, maxLines).join('\n').slice(0, maxChars),
          );
        }
      }
    },
  );
});
