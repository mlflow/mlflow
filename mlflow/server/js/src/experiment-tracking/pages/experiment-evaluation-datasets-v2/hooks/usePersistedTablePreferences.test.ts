import { afterEach, beforeEach, describe, expect, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import type { ColumnSizingState } from '@tanstack/react-table';
import { usePersistedTablePreferences } from './usePersistedTablePreferences';

const COLUMNS = ['inputs', 'expectations', 'last_updated', 'source', 'tags'] as const;
const DEFAULTS = ['inputs', 'expectations', 'last_updated', 'source', 'tags'] as const;

// Mirror `useLocalStorage`'s key format: `${key}_v${version}`. Knowing the
// shape lets us pre-seed corrupt entries directly and read back the raw
// persisted value to verify what landed in storage after a setter call.
const STORAGE_VERSION = 1;
const buildStorageKey = (experimentId: string, datasetId: string) =>
  `mlflow.eval-datasets.table-prefs.${experimentId}.${datasetId}_v${STORAGE_VERSION}`;

const seedRaw = (experimentId: string, datasetId: string, raw: Record<string, unknown>) => {
  window.localStorage.setItem(buildStorageKey(experimentId, datasetId), JSON.stringify(raw));
};

const readRaw = (experimentId: string, datasetId: string): unknown => {
  const raw = window.localStorage.getItem(buildStorageKey(experimentId, datasetId));
  return raw === null ? null : JSON.parse(raw);
};

describe('usePersistedTablePreferences', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  describe('visibility', () => {
    test('returns defaults when nothing is persisted', () => {
      const { result } = renderHook(() =>
        usePersistedTablePreferences({
          experimentId: 'exp-1',
          datasetId: 'ds-1',
          allColumns: COLUMNS,
          defaultVisible: DEFAULTS,
        }),
      );
      expect(result.current.visibleColumns).toEqual(DEFAULTS);
    });

    test('toggling a column removes it; toggling again restores it in canonical order', () => {
      const { result } = renderHook(() =>
        usePersistedTablePreferences({
          experimentId: 'exp-1',
          datasetId: 'ds-1',
          allColumns: COLUMNS,
          defaultVisible: DEFAULTS,
        }),
      );

      act(() => {
        result.current.toggleColumn('source');
      });
      expect(result.current.visibleColumns).toEqual(['inputs', 'expectations', 'last_updated', 'tags']);

      act(() => {
        result.current.toggleColumn('source');
      });
      expect(result.current.visibleColumns).toEqual(['inputs', 'expectations', 'last_updated', 'source', 'tags']);
    });

    test('persists visibility across remounts for the same dataset', () => {
      const params = {
        experimentId: 'exp-1',
        datasetId: 'ds-1',
        allColumns: COLUMNS,
        defaultVisible: DEFAULTS,
      } as const;

      const first = renderHook(() => usePersistedTablePreferences(params));
      act(() => {
        first.result.current.toggleColumn('tags');
      });
      first.unmount();

      const second = renderHook(() => usePersistedTablePreferences(params));
      expect(second.result.current.visibleColumns).toEqual(['inputs', 'expectations', 'last_updated', 'source']);
    });

    test('visibility is isolated per dataset', () => {
      const first = renderHook(() =>
        usePersistedTablePreferences({
          experimentId: 'exp-1',
          datasetId: 'ds-1',
          allColumns: COLUMNS,
          defaultVisible: DEFAULTS,
        }),
      );
      act(() => {
        first.result.current.toggleColumn('source');
      });

      const second = renderHook(() =>
        usePersistedTablePreferences({
          experimentId: 'exp-1',
          datasetId: 'ds-2',
          allColumns: COLUMNS,
          defaultVisible: DEFAULTS,
        }),
      );
      expect(second.result.current.visibleColumns).toEqual(DEFAULTS);
    });

    test('adding a new column to allColumns preserves existing persisted visibility', () => {
      // Lock in the no-bump invariant: when a future change extends the column set, users
      // with an existing customized visibility should keep their selection and the new
      // column should simply be off by default until they toggle it on.
      const initialParams = {
        experimentId: 'exp-1',
        datasetId: 'ds-1',
        allColumns: COLUMNS,
        defaultVisible: DEFAULTS,
      } as const;
      const first = renderHook(() => usePersistedTablePreferences(initialParams));
      act(() => {
        first.result.current.toggleColumn('source');
      });
      first.unmount();

      const extendedColumns = ['inputs', 'expectations', 'last_updated', 'created_by', 'source', 'tags'] as const;
      const second = renderHook(() =>
        usePersistedTablePreferences({
          experimentId: 'exp-1',
          datasetId: 'ds-1',
          allColumns: extendedColumns,
          defaultVisible: DEFAULTS,
        }),
      );
      expect(second.result.current.visibleColumns).toEqual(['inputs', 'expectations', 'last_updated', 'tags']);

      act(() => {
        second.result.current.toggleColumn('created_by');
      });
      expect(second.result.current.visibleColumns).toEqual([
        'inputs',
        'expectations',
        'last_updated',
        'created_by',
        'tags',
      ]);
    });
  });

  describe('column sizing', () => {
    test('starts empty when nothing is persisted', () => {
      const { result } = renderHook(() =>
        usePersistedTablePreferences({
          experimentId: 'exp-1',
          datasetId: 'ds-1',
          allColumns: COLUMNS,
          defaultVisible: DEFAULTS,
        }),
      );
      expect(result.current.columnSizing).toEqual({});
    });

    test('setColumnSizing stores a direct value', () => {
      const { result } = renderHook(() =>
        usePersistedTablePreferences({
          experimentId: 'exp-1',
          datasetId: 'ds-1',
          allColumns: COLUMNS,
          defaultVisible: DEFAULTS,
        }),
      );

      act(() => {
        result.current.setColumnSizing({ inputs: 512 });
      });

      expect(result.current.columnSizing).toEqual({ inputs: 512 });
    });

    test('setColumnSizing accepts a functional updater (the shape TanStack calls it with)', () => {
      const { result } = renderHook(() =>
        usePersistedTablePreferences({
          experimentId: 'exp-1',
          datasetId: 'ds-1',
          allColumns: COLUMNS,
          defaultVisible: DEFAULTS,
        }),
      );

      act(() => {
        result.current.setColumnSizing({ inputs: 512 });
      });
      act(() => {
        result.current.setColumnSizing((prev) => ({ ...prev, expectations: 320 }));
      });

      expect(result.current.columnSizing).toEqual({ inputs: 512, expectations: 320 });
    });

    test('persists widths across remounts for the same dataset', () => {
      const params = {
        experimentId: 'exp-1',
        datasetId: 'ds-1',
        allColumns: COLUMNS,
        defaultVisible: DEFAULTS,
      } as const;

      const first = renderHook(() => usePersistedTablePreferences(params));
      act(() => {
        first.result.current.setColumnSizing({ inputs: 256 });
      });
      first.unmount();

      const second = renderHook(() => usePersistedTablePreferences(params));
      expect(second.result.current.columnSizing).toEqual({ inputs: 256 });
    });

    test('widths are isolated per dataset', () => {
      const first = renderHook(() =>
        usePersistedTablePreferences({
          experimentId: 'exp-1',
          datasetId: 'ds-1',
          allColumns: COLUMNS,
          defaultVisible: DEFAULTS,
        }),
      );
      act(() => {
        first.result.current.setColumnSizing({ inputs: 256 });
      });

      const second = renderHook(() =>
        usePersistedTablePreferences({
          experimentId: 'exp-1',
          datasetId: 'ds-2',
          allColumns: COLUMNS,
          defaultVisible: DEFAULTS,
        }),
      );
      expect(second.result.current.columnSizing).toEqual({});
    });

    test('filters corrupt sizing entries pre-seeded in localStorage on read', () => {
      // Schema-less storage means corrupt entries (negatives, strings, zero, null) can land
      // in localStorage. Guard that the sanitizer strips them before they flow into
      // `flex: 0 0 ${px}px` and collapse a column.
      seedRaw('exp-1', 'ds-1', {
        visibleColumns: [...DEFAULTS],
        columnSizing: {
          inputs: 400,
          source: -10,
          tags: 'foo',
          create_time: 0,
          last_updated: null,
        },
      });

      const { result } = renderHook(() =>
        usePersistedTablePreferences({
          experimentId: 'exp-1',
          datasetId: 'ds-1',
          allColumns: COLUMNS,
          defaultVisible: DEFAULTS,
        }),
      );
      expect(result.current.columnSizing).toEqual({ inputs: 400 });
    });

    test('does not persist corrupt entries when the setter is called with a direct value', () => {
      const { result } = renderHook(() =>
        usePersistedTablePreferences({
          experimentId: 'exp-1',
          datasetId: 'ds-1',
          allColumns: COLUMNS,
          defaultVisible: DEFAULTS,
        }),
      );

      act(() => {
        result.current.setColumnSizing({
          inputs: 400,
          source: -10,
          tags: 'foo' as unknown as number,
        } as ColumnSizingState);
      });

      // Read the raw persisted value rather than result.current.columnSizing: the read
      // path also sanitises, so checking it doesn't prove the write path did.
      expect(readRaw('exp-1', 'ds-1')).toEqual({
        visibleColumns: [...DEFAULTS],
        columnSizing: { inputs: 400 },
      });
    });

    test('strips corrupt prev entries when TanStack invokes the functional updater', () => {
      // Pre-seed corruption that pre-dates this fix. On the user's next drag, TanStack
      // calls `(prev) => ({ ...prev, [id]: nextWidth })`. Without prev-sanitisation,
      // the bad entry would survive the spread and get re-persisted, lingering forever.
      seedRaw('exp-1', 'ds-1', {
        visibleColumns: [...DEFAULTS],
        columnSizing: { inputs: 400, source: -10 },
      });

      const { result } = renderHook(() =>
        usePersistedTablePreferences({
          experimentId: 'exp-1',
          datasetId: 'ds-1',
          allColumns: COLUMNS,
          defaultVisible: DEFAULTS,
        }),
      );

      act(() => {
        result.current.setColumnSizing((prev) => ({ ...prev, expectations: 500 }));
      });

      expect(readRaw('exp-1', 'ds-1')).toEqual({
        visibleColumns: [...DEFAULTS],
        columnSizing: { inputs: 400, expectations: 500 },
      });
    });
  });

  describe('resetToDefaults', () => {
    test('clears both visibility and column widths', () => {
      const { result } = renderHook(() =>
        usePersistedTablePreferences({
          experimentId: 'exp-1',
          datasetId: 'ds-1',
          allColumns: COLUMNS,
          defaultVisible: DEFAULTS,
        }),
      );

      act(() => {
        result.current.toggleColumn('inputs');
        result.current.setColumnSizing({ inputs: 400, tags: 200 });
      });
      expect(result.current.visibleColumns).not.toEqual(DEFAULTS);
      expect(result.current.columnSizing).not.toEqual({});

      act(() => {
        result.current.resetToDefaults();
      });
      expect(result.current.visibleColumns).toEqual(DEFAULTS);
      expect(result.current.columnSizing).toEqual({});
    });
  });
});
