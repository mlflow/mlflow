import { describe, expect, test } from '@jest/globals';
import type { ColumnState } from '@ag-grid-community/core';

import { columnStateToPrefs, prefsToColumnState } from './agGridColumnPrefsAdapter';

const ALL = ['a', 'b', 'c', 'd'] as const;

describe('agGridColumnPrefsAdapter', () => {
  describe('prefsToColumnState', () => {
    test('maps order and width; undefined width keeps ag-grid default', () => {
      const state = prefsToColumnState(['c', 'a', 'b'], { c: 200 });
      expect(state).toEqual([
        { colId: 'c', width: 200 },
        { colId: 'a', width: undefined },
        { colId: 'b', width: undefined },
      ]);
    });
  });

  describe('columnStateToPrefs', () => {
    test('captures order and sane widths; drops unknown ids and corrupt widths', () => {
      const state: ColumnState[] = [
        { colId: 'c', width: 120 },
        { colId: 'ghost', width: 80 },
        { colId: 'a', width: 90 },
        { colId: 'b', width: 0 },
        { colId: 'd', width: -5 },
      ] as ColumnState[];

      expect(columnStateToPrefs(state, ALL)).toEqual({
        columnOrder: ['c', 'a', 'b', 'd'],
        columnWidths: { c: 120, a: 90 },
      });
    });
  });

  test('round-trips an ordered/sized snapshot', () => {
    const restored = columnStateToPrefs(prefsToColumnState(['b', 'a', 'c', 'd'], { b: 150 }) as ColumnState[], ALL);
    expect(restored.columnOrder).toEqual(['b', 'a', 'c', 'd']);
    expect(restored.columnWidths).toEqual({ b: 150 });
  });
});
