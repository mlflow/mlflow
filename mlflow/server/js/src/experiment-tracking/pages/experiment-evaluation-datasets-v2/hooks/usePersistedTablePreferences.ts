import { useCallback, useMemo } from 'react';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import type { ColumnSizingState, OnChangeFn } from '@tanstack/react-table';

const STORAGE_KEY_PREFIX = 'mlflow.eval-datasets.table-prefs';
// Bump to invalidate all stored entries (no migration path).
const STORAGE_VERSION = 1;

interface StoredPreferences<ColumnId extends string> {
  visibleColumns: ColumnId[];
  columnSizing: ColumnSizingState;
}

interface UsePersistedTablePreferencesParams<ColumnId extends string> {
  experimentId: string;
  datasetId: string;
  /** All known column ids. Order is preserved when toggling. */
  allColumns: readonly ColumnId[];
  /** Columns visible by default when the user hasn't customized. */
  defaultVisible: readonly ColumnId[];
}

interface PersistedTablePreferences<ColumnId extends string> {
  visibleColumns: ColumnId[];
  toggleColumn: (column: ColumnId) => void;
  columnSizing: ColumnSizingState;
  setColumnSizing: OnChangeFn<ColumnSizingState>;
  /** Clears visibility customization AND column widths back to defaults. */
  resetToDefaults: () => void;
}

// Filters non-positive-finite values so corrupt/old entries (NaN, strings,
// null, negatives, 0) can't flow into `flex: 0 0 ${px}px` and collapse a column.
const sanitizeColumnSizing = (raw: unknown): ColumnSizingState => {
  if (raw === null || typeof raw !== 'object') return {};
  const entries = Object.entries(raw as Record<string, unknown>).filter(
    ([, v]) => typeof v === 'number' && Number.isFinite(v) && v > 0,
  );
  return Object.fromEntries(entries) as ColumnSizingState;
};

/**
 * Persists table preferences (visible columns + column widths) in localStorage,
 * scoped per (experiment, dataset). One storage entry holds both so "reset to
 * defaults" clears them atomically and future prefs (sort order, etc.) drop in
 * as new fields.
 *
 * Why localStorage: synchronous read on mount avoids a column-flicker on first
 * paint; IndexedDB would require an async read. Sizes are tiny (<200B per dataset).
 */
export const usePersistedTablePreferences = <ColumnId extends string>({
  experimentId,
  datasetId,
  allColumns,
  defaultVisible,
}: UsePersistedTablePreferencesParams<ColumnId>): PersistedTablePreferences<ColumnId> => {
  const key = `${STORAGE_KEY_PREFIX}.${experimentId}.${datasetId}`;

  const [stored, setStored] = useLocalStorage<StoredPreferences<ColumnId>>({
    key,
    version: STORAGE_VERSION,
    initialValue: { visibleColumns: [...defaultVisible], columnSizing: {} },
  });

  // Iterate `allColumns` rather than `stored.visibleColumns` so the returned
  // order is always the canonical render order and unknown ids are dropped.
  const visibleColumns = useMemo(
    () => allColumns.filter((id) => stored.visibleColumns?.includes(id)),
    [stored.visibleColumns, allColumns],
  );

  const columnSizing = useMemo(() => sanitizeColumnSizing(stored.columnSizing), [stored.columnSizing]);

  const toggleColumn = useCallback(
    (column: ColumnId) => {
      setStored((prev) => {
        const prevVisible = (prev.visibleColumns ?? []).filter((id) => allColumns.includes(id));
        const nextVisible = prevVisible.includes(column)
          ? prevVisible.filter((id) => id !== column)
          : // Preserve canonical column order so toggling on doesn't shuffle the table.
            allColumns.filter((id) => prevVisible.includes(id) || id === column);
        return { ...prev, visibleColumns: nextVisible };
      });
    },
    [setStored, allColumns],
  );

  // TanStack's onColumnSizingChange passes a functional updater on every drag
  // tick. Feeding it a sanitised `prev` ensures corrupt entries already in
  // storage die on the next write instead of being respread via `...prev`.
  const setColumnSizing: OnChangeFn<ColumnSizingState> = useCallback(
    (updater) => {
      setStored((prev) => {
        const cleanPrev = sanitizeColumnSizing(prev.columnSizing);
        const next = typeof updater === 'function' ? updater(cleanPrev) : updater;
        return { ...prev, columnSizing: sanitizeColumnSizing(next) };
      });
    },
    [setStored],
  );

  const resetToDefaults = useCallback(() => {
    setStored({ visibleColumns: [...defaultVisible], columnSizing: {} });
  }, [setStored, defaultVisible]);

  return { visibleColumns, toggleColumn, columnSizing, setColumnSizing, resetToDefaults };
};
