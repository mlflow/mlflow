import { useCallback, useMemo } from 'react';
import { useLocalStorage } from '@databricks/web-shared/hooks';

const COLUMN_STORAGE_KEY_PREFIX = 'mlflow.eval-datasets.columns';

// Bump when the column set or ID scheme changes — old persisted entries will be reset.
const COLUMN_STORAGE_VERSION = 1;

interface UsePersistedTableColumnsParams<ColumnId extends string> {
  experimentId: string;
  /** Per-dataset key; pass undefined to scope only by experiment (used by the list page if ever needed). */
  datasetId?: string;
  /** All known column ids. Order is preserved when toggling. */
  allColumns: readonly ColumnId[];
  /** Columns visible by default when the user hasn't customized. */
  defaultVisible: readonly ColumnId[];
}

interface PersistedTableColumns<ColumnId extends string> {
  visibleColumns: ColumnId[];
  setVisibleColumns: (next: ColumnId[]) => void;
  toggleColumn: (column: ColumnId) => void;
  isVisible: (column: ColumnId) => boolean;
  resetToDefaults: () => void;
}

/**
 * Persists column-visibility selections in localStorage, scoped per (experiment, dataset)
 * — so each dataset can have its own remembered set. Falls back to `defaultVisible` when
 * nothing is stored (or when the version changes).
 *
 * Why localStorage: synchronous read on mount avoids a column-flicker on first paint;
 * IndexedDB would require an async read. Sizes are tiny (<200B per dataset).
 */
export const usePersistedTableColumns = <ColumnId extends string>({
  experimentId,
  datasetId,
  allColumns,
  defaultVisible,
}: UsePersistedTableColumnsParams<ColumnId>): PersistedTableColumns<ColumnId> => {
  const key = datasetId
    ? `${COLUMN_STORAGE_KEY_PREFIX}.${experimentId}.${datasetId}`
    : `${COLUMN_STORAGE_KEY_PREFIX}.${experimentId}`;

  const [stored, setStored] = useLocalStorage<ColumnId[]>({
    key,
    version: COLUMN_STORAGE_VERSION,
    // Spread to a mutable copy — useLocalStorage stores a mutable array.
    initialValue: [...defaultVisible],
    scoped: true,
  });

  // Iterate `allColumns` rather than `stored` so the returned order is always the
  // canonical render order, not whatever order an older `stored` payload happens to
  // have. The records table's phantom new-record row iterates `visibleColumns` for
  // its cells while the data rows render in JSX-canonical order — pinning the
  // visible-column order to `allColumns` keeps the two consistent across migrations
  // (e.g. swapping `source` / `last_updated`) without invalidating user toggles.
  // Also filters out any ids that no longer exist, same defensive behavior as before.
  const visibleColumns = useMemo(() => allColumns.filter((id) => stored.includes(id)), [stored, allColumns]);

  const setVisibleColumns = useCallback(
    (next: ColumnId[]) => {
      setStored(next.filter((id) => allColumns.includes(id)));
    },
    [setStored, allColumns],
  );

  const toggleColumn = useCallback(
    (column: ColumnId) => {
      setStored((prev) => {
        const filtered = prev.filter((id) => allColumns.includes(id));
        if (filtered.includes(column)) {
          return filtered.filter((id) => id !== column);
        }
        // Preserve the canonical column order so toggling on doesn't shuffle the table.
        return allColumns.filter((id) => filtered.includes(id) || id === column);
      });
    },
    [setStored, allColumns],
  );

  const isVisible = useCallback((column: ColumnId) => visibleColumns.includes(column), [visibleColumns]);

  const resetToDefaults = useCallback(() => {
    setStored([...defaultVisible]);
  }, [setStored, defaultVisible]);

  return { visibleColumns, setVisibleColumns, toggleColumn, isVisible, resetToDefaults };
};
