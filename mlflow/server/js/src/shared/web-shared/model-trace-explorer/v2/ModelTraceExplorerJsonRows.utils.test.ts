import { describe, expect, it } from '@jest/globals';

import {
  DEFAULT_JSON_ROWS_EXPAND_DEPTH,
  getJsonRows,
  getJsonRowsExpandDepth,
} from './ModelTraceExplorerJsonRows.utils';

const expandToDefaultDepth = (_container: WeakKey, depth: number) => depth <= DEFAULT_JSON_ROWS_EXPAND_DEPTH;

const describeRows = (rows: ReturnType<typeof getJsonRows>) =>
  rows.map((row) => {
    switch (row.kind) {
      case 'primitive':
        return `${row.line} ${row.nodeKey ?? '-'}=${String(row.value)}${row.isLast ? '' : ','}`;
      case 'open':
        return `${row.line} ${row.nodeKey ?? '-'}${row.isArray ? '[' : '{'}${row.collapsed ? '...' : ''}`;
      case 'close':
        return `${row.line} ${row.isArray ? ']' : '}'}`;
      default:
        return `${row.line} max-depth`;
    }
  });

describe('getJsonRows', () => {
  it('numbers lines as if every container were expanded', () => {
    const value = { a: 1, list: [true, null], empty: {}, nested: { b: 'x' } };

    expect(describeRows(getJsonRows(value, expandToDefaultDepth))).toEqual([
      '1 -{',
      '2 a=1,',
      '3 list[',
      '4 -=true,',
      '5 -=null',
      '6 ]',
      // an empty container takes two line numbers but draws on one row
      '7 empty{',
      '9 nested{',
      '10 b=x',
      '11 }',
      '12 }',
    ]);
  });

  it('labels a collapsed container with its closing line and skips its contents', () => {
    const value = { nested: { b: 1, c: 2 }, after: 3 };
    const collapseNested = (container: WeakKey, depth: number) => depth === 0 || container !== value.nested;

    expect(describeRows(getJsonRows(value, collapseNested))).toEqual(['1 -{', '5 nested{...', '6 after=3', '7 }']);
  });

  it('collapses containers deeper than the expand depth', () => {
    let value: unknown = 'leaf';
    for (let depth = 0; depth < 8; depth++) {
      value = [value];
    }
    const rows = getJsonRows(value, expandToDefaultDepth);

    const openRows = rows.filter((row) => row.kind === 'open');
    expect(openRows.map((row) => row.kind === 'open' && row.collapsed)).toEqual([
      false,
      false,
      false,
      false,
      false,
      false,
      true,
    ]);
  });

  it('keeps depth-limited line counts correct when the same value appears at another depth', () => {
    const leaf = { nested: { value: 1 } };
    let deep: unknown = leaf;
    for (let depth = 0; depth < 100; depth++) deep = [deep];
    getJsonRows(deep, () => false);
    const rows = getJsonRows({ leaf, after: true }, (_container, depth) => depth === 0);
    expect(describeRows(rows)).toEqual(['1 -{', '6 leaf{...', '7 after=true', '8 }']);
  });

  it('stops at the row limit', () => {
    const value = Object.fromEntries(Array.from({ length: 1000 }, (_, index) => [`key-${index}`, index]));
    const full = getJsonRows(value, expandToDefaultDepth);
    const preview = getJsonRows(value, expandToDefaultDepth, 8);

    expect(full).toHaveLength(1002);
    expect(preview).toEqual(full.slice(0, 8));
  });
});

describe('getJsonRowsExpandDepth', () => {
  it('expands less of larger payloads', () => {
    expect(getJsonRowsExpandDepth(10_000)).toBe(DEFAULT_JSON_ROWS_EXPAND_DEPTH);
    expect(getJsonRowsExpandDepth(500_000)).toBeLessThan(DEFAULT_JSON_ROWS_EXPAND_DEPTH);
    expect(getJsonRowsExpandDepth(100_000)).toBe(getJsonRowsExpandDepth(500_000));
    expect(getJsonRowsExpandDepth(5_000_000)).toBeLessThan(getJsonRowsExpandDepth(500_000));
  });
});
