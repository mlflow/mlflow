import { describe, expect, test } from '@jest/globals';
import type { ColumnState } from '@ag-grid-community/core';

import { columnStateToPrefs, prefsToColumnState } from './agGridColumnPrefsAdapter';

const ALL = ['a', 'b', 'c', 'd'] as const;

describe('agGridColumnPrefsAdapter', () => {
  describe('prefsToColumnState', () => {
    test('maps order, width, and hide (from visibleColumns)', () => {
      const state = prefsToColumnState({
        visibleColumns: ['a', 'c'],
        columnOrder: ['c', 'a', 'b'],
        columnWidths: { c: 200 },
      });
      expect(state).toEqual([
        { colId: 'c', hide: false, width: 200 },
        { colId: 'a', hide: false, width: undefined },
        { colId: 'b', hide: true, width: undefined },
      ]);
    });
  });

  describe('columnStateToPrefs', () => {
    test('captures order, visibility (from !hide), and numeric widths; drops unknown ids', () => {
      const state: ColumnState[] = [
        { colId: 'c', hide: false, width: 120 },
        { colId: 'ghost', hide: false, width: 80 },
        { colId: 'a', hide: true, width: 90 },
        { colId: 'b', hide: false },
      ] as ColumnState[];

      expect(columnStateToPrefs(state, ALL)).toEqual({
        columnOrder: ['c', 'a', 'b'],
        visibleColumns: ['c', 'b'],
        columnWidths: { c: 120, a: 90 },
      });
    });
  });

  test('round-trips a visible/ordered/sized snapshot', () => {
    const prefs = { visibleColumns: ['b', 'a'], columnOrder: ['b', 'a', 'c', 'd'], columnWidths: { b: 150 } };
    const restored = columnStateToPrefs(prefsToColumnState(prefs) as ColumnState[], ALL);
    expect(restored.columnOrder).toEqual(['b', 'a', 'c', 'd']);
    expect(new Set(restored.visibleColumns)).toEqual(new Set(['b', 'a']));
    expect(restored.columnWidths).toEqual({ b: 150 });
  });
});
