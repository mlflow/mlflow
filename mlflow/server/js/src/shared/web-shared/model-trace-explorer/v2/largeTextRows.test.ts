import { describe, expect, it } from '@jest/globals';

import { getLargeTextRows } from './largeTextRows';

describe('getLargeTextRows', () => {
  it.each([
    '',
    '\n',
    'a\n',
    '\n\n',
    'a\r\nb',
    'café 日本語 🙂',
    'x'.repeat(5000),
    'x'.repeat(1999) + '🙂' + 'y'.repeat(3000),
  ])('preserves every character and blank line for case %#', (text) => {
    const rows = getLargeTextRows(text);
    const reconstructed = rows
      .map((row, index) => `${index && row.lineNumber ? '\n' : ''}${text.slice(row.start, row.end)}`)
      .join('');
    expect(reconstructed).toBe(text);
    for (const row of rows) {
      expect(row.end - row.start).toBeLessThanOrEqual(2000);
      expect(text.slice(row.start, row.end)).not.toMatch(/^[\uDC00-\uDFFF]|[\uD800-\uDBFF]$/u);
    }
    expect(rows.filter((row) => row.lineNumber).map((row) => row.lineNumber)).toEqual(
      text.split('\n').map((_, index) => index + 1),
    );
  });
});
