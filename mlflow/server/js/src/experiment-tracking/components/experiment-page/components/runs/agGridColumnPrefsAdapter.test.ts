import { describe, expect, test } from '@jest/globals';
import type { ColumnState } from '@ag-grid-community/core';

import { columnStateToPrefs, getReorderCorrection, prefsToColumnState } from './agGridColumnPrefsAdapter';

const ALL = ['a', 'b', 'c', 'd'] as const;

// Realistic runs-table shape: "cb" stands in for the structural checkbox column (ag-grid gives it
// an auto colId, not present in the data-column set); RUN_NAME + CREATED are the pinned anchors.
const RUN_NAME = 'attributes.`Run Name`';
const CREATED = 'runDateAndNestInfo';
const ANCHORS = [RUN_NAME, CREATED];
const DATA = [RUN_NAME, CREATED, 'datasets', 'duration', 'attributes.`Source`'];

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

  describe('getReorderCorrection', () => {
    test('returns null for the default valid order (anchors lead the data columns)', () => {
      const order = ['1', RUN_NAME, CREATED, 'datasets', 'duration', 'attributes.`Source`'];
      expect(getReorderCorrection(order, ANCHORS, DATA)).toBeNull();
    });

    test('returns null when only non-anchor data columns are reordered', () => {
      const order = ['1', RUN_NAME, CREATED, 'duration', 'attributes.`Source`', 'datasets'];
      expect(getReorderCorrection(order, ANCHORS, DATA)).toBeNull();
    });

    test('snaps a data column dropped before Run Name back behind the anchors', () => {
      // User dragged Source in front of Run Name.
      const order = ['1', 'attributes.`Source`', RUN_NAME, CREATED, 'datasets', 'duration'];
      expect(getReorderCorrection(order, ANCHORS, DATA)).toEqual([
        '1',
        RUN_NAME,
        CREATED,
        'attributes.`Source`',
        'datasets',
        'duration',
      ]);
    });

    test('corrects a column dropped between the two anchors', () => {
      const order = ['1', RUN_NAME, 'duration', CREATED, 'datasets', 'attributes.`Source`'];
      expect(getReorderCorrection(order, ANCHORS, DATA)).toEqual([
        '1',
        RUN_NAME,
        CREATED,
        'duration',
        'datasets',
        'attributes.`Source`',
      ]);
    });

    test('keeps the structural checkbox column in its leading slot', () => {
      const order = ['1', 'datasets', RUN_NAME, CREATED, 'duration', 'attributes.`Source`'];
      const corrected = getReorderCorrection(order, ANCHORS, DATA);
      expect(corrected?.[0]).toBe('1');
      expect(corrected).toEqual(['1', RUN_NAME, CREATED, 'datasets', 'duration', 'attributes.`Source`']);
    });

    test('returns null when no anchors are present (compact/compare layouts)', () => {
      const order = ['1', 'datasets', 'duration', 'attributes.`Source`'];
      expect(getReorderCorrection(order, ANCHORS, DATA)).toBeNull();
    });

    test('returns null when anchors are configured but absent from the data columns', () => {
      // dataColIds not yet populated (empty) — nothing is treated as a data column, so there is
      // no order to protect and the correction is intentionally skipped.
      const order = ['1', 'attributes.`Source`', RUN_NAME, CREATED];
      expect(getReorderCorrection(order, ANCHORS, [])).toBeNull();
    });
  });
});
