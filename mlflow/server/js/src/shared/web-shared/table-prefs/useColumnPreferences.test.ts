import { afterEach, beforeEach, describe, expect, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';

import { useColumnPreferences } from './useColumnPreferences';

const COLUMNS = ['a', 'b', 'c', 'd'] as const;
const DEFAULTS = ['a', 'b', 'c'] as const;
const STORAGE_KEY = 'mlflow.test.column-prefs';
const VERSION = 1;

// Mirror `useLocalStorage`'s key format: `${key}_v${version}`.
const fullKey = `${STORAGE_KEY}_v${VERSION}`;
const seedRaw = (raw: Record<string, unknown>) => window.localStorage.setItem(fullKey, JSON.stringify(raw));
const readRaw = (): any => {
  const raw = window.localStorage.getItem(fullKey);
  return raw === null ? null : JSON.parse(raw);
};

const render = () =>
  renderHook(() =>
    useColumnPreferences({ storageKey: STORAGE_KEY, version: VERSION, allColumns: COLUMNS, defaultVisible: DEFAULTS }),
  );

describe('useColumnPreferences', () => {
  beforeEach(() => window.localStorage.clear());
  afterEach(() => window.localStorage.clear());

  describe('visibility', () => {
    test('returns defaults when nothing is persisted', () => {
      const { result } = render();
      expect(result.current.preferences.visibleColumns).toEqual(['a', 'b', 'c']);
    });

    test('respects a persisted empty selection (user hid everything)', () => {
      seedRaw({ visibleColumns: [] });
      const { result } = render();
      expect(result.current.preferences.visibleColumns).toEqual([]);
    });

    test('toggling off removes a column; toggling on restores it in canonical order', () => {
      const { result } = render();
      act(() => result.current.toggleColumn('a'));
      expect(result.current.preferences.visibleColumns).toEqual(['b', 'c']);
      act(() => result.current.toggleColumn('a'));
      expect(result.current.preferences.visibleColumns).toEqual(['a', 'b', 'c']);
    });

    test('drops unknown colIds from a persisted selection', () => {
      seedRaw({ visibleColumns: ['a', 'gone', 'c'] });
      const { result } = render();
      expect(result.current.preferences.visibleColumns).toEqual(['a', 'c']);
    });

    test('persists across remounts', () => {
      const first = render();
      act(() => first.result.current.setVisibleColumns(['d']));
      const second = render();
      expect(second.result.current.preferences.visibleColumns).toEqual(['d']);
    });
  });

  describe('column widths', () => {
    test('sanitizes corrupt widths (NaN, <=0, non-number) on read', () => {
      seedRaw({ columnWidths: { a: 120, b: 0, c: -5, d: 'wide' } });
      const { result } = render();
      expect(result.current.preferences.columnWidths).toEqual({ a: 120 });
    });

    test('setColumnWidths accepts a functional updater and sanitizes the result', () => {
      const { result } = render();
      act(() => result.current.setColumnWidths((prev) => ({ ...prev, a: 200, b: -1 })));
      expect(result.current.preferences.columnWidths).toEqual({ a: 200 });
      expect(readRaw().columnWidths).toEqual({ a: 200 });
    });
  });

  describe('column order', () => {
    test('keeps stored order for known ids and appends missing known ids canonically', () => {
      seedRaw({ columnOrder: ['c', 'a'] });
      const { result } = render();
      expect(result.current.preferences.columnOrder).toEqual(['c', 'a', 'b', 'd']);
    });

    test('drops unknown ids from stored order', () => {
      seedRaw({ columnOrder: ['c', 'ghost', 'a'] });
      const { result } = render();
      expect(result.current.preferences.columnOrder).toEqual(['c', 'a', 'b', 'd']);
    });
  });

  describe('isCustomized', () => {
    test('false when nothing persisted, true after a customization, false after reset', () => {
      const { result } = render();
      expect(result.current.isCustomized).toBe(false);
      act(() => result.current.setColumnWidths({ a: 120 }));
      expect(result.current.isCustomized).toBe(true);
      act(() => result.current.resetToDefaults());
      expect(result.current.isCustomized).toBe(false);
    });

    test('true when a stored entry exists on mount', () => {
      seedRaw({ columnOrder: ['b', 'a'] });
      const { result } = render();
      expect(result.current.isCustomized).toBe(true);
    });
  });

  describe('reset / serialize / hydrate', () => {
    test('resetToDefaults clears visibility, order, and widths atomically', () => {
      const { result } = render();
      act(() => {
        result.current.setVisibleColumns(['d']);
        result.current.setColumnOrder(['d', 'a']);
        result.current.setColumnWidths({ a: 99 });
      });
      act(() => result.current.resetToDefaults());
      expect(result.current.preferences.visibleColumns).toEqual(['a', 'b', 'c']);
      expect(result.current.preferences.columnOrder).toEqual(['a', 'b', 'c', 'd']);
      expect(result.current.preferences.columnWidths).toEqual({});
    });

    test('serialize returns the sanitized snapshot; hydrate round-trips it', () => {
      const source = render();
      act(() => {
        source.result.current.setVisibleColumns(['b', 'd']);
        source.result.current.setColumnWidths({ b: 150 });
      });
      const snapshot = source.result.current.serialize();

      window.localStorage.clear();
      const target = render();
      act(() => target.result.current.hydrate(snapshot));
      expect(target.result.current.preferences.visibleColumns).toEqual(['b', 'd']);
      expect(target.result.current.preferences.columnWidths).toEqual({ b: 150 });
    });
  });
});
