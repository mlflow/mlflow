import { describe, expect, it } from '@jest/globals';

import { getLargeTextRows, getLargeTextSearchRows } from './largeTextRows';

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
      .map((row, index) => `${index && row.start > rows[index - 1].end ? '\n' : ''}${text.slice(row.start, row.end)}`)
      .join('');
    expect(reconstructed).toBe(text);
    for (const row of rows) {
      expect(row.end - row.start).toBeLessThanOrEqual(2000);
      expect(text.slice(row.start, row.end)).not.toMatch(/^[\uDC00-\uDFFF]|[\uD800-\uDBFF]$/u);
    }
    expect(rows.filter((row, index) => index === 0 || row.start > rows[index - 1].end)).toHaveLength(
      text.split('\n').length,
    );
  });
});

describe('large text search rows', () => {
  it('preserves non-overlapping occurrence indexes across chunk boundaries', () => {
    const result = getLargeTextSearchRows(
      'aaaaa',
      [
        { start: 0, end: 1 },
        { start: 1, end: 3 },
        { start: 3, end: 5 },
      ],
      'aa',
    );
    expect(result.matches).toEqual([
      { offset: 0, index: 0, lastIndex: 0 },
      { offset: 0, index: 0, lastIndex: 1 },
      { offset: 2, index: 1, lastIndex: 1 },
    ]);
  });

  it.each(['café', '[.*]', '🌳', 'x\ny'])(
    'indexes literal case-insensitive %s matches without losing crossing matches',
    (query) => {
      const text = 'z'.repeat(1999) + query.toUpperCase() + 'z'.repeat(3000) + query;
      const rows = getLargeTextRows(text);
      const { matches } = getLargeTextSearchRows(text, rows, query);
      for (const [index, offset] of [1999, 1999 + query.length + 3000].entries()) {
        const intersecting = rows
          .map((row, rowIndex) => ({ row, rowIndex }))
          .filter(({ row }) => row.end > offset && row.start < offset + query.length);
        expect(intersecting.length).toBeGreaterThan(0);
        for (const { rowIndex } of intersecting) {
          expect(matches[rowIndex]).toMatchObject({ offset, index, lastIndex: index });
        }
      }
    },
  );

  it('keeps only one small search record per text row even for dense matches', () => {
    const text = 'x'.repeat(100001);
    const rows = getLargeTextRows(text);
    const { matches } = getLargeTextSearchRows(text, rows, 'x');
    expect(matches).toHaveLength(rows.length);
    expect(matches[0]).toEqual({ offset: 0, index: 0, lastIndex: 1999 });
    expect(matches[matches.length - 1]?.lastIndex).toBe(text.length - 1);
    expect(getLargeTextSearchRows(text, rows, ' ').matches).toEqual([]);
    expect(getLargeTextSearchRows(text, rows, 'absent').matches).toEqual([]);
  });
});
