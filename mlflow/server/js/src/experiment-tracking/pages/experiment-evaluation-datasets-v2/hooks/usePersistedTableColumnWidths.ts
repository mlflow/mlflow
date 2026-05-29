import { useCallback, useMemo } from 'react';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import type { ColumnSizingState, OnChangeFn } from '@tanstack/react-table';

const COLUMN_WIDTH_STORAGE_KEY_PREFIX = 'mlflow.eval-datasets.column-widths';

const COLUMN_WIDTH_STORAGE_VERSION = 1;

interface UsePersistedTableColumnWidthsParams {
  experimentId: string;
  datasetId: string;
}

interface PersistedColumnWidths {
  columnSizing: ColumnSizingState; // controlled state for useReactTable
  setColumnSizing: OnChangeFn<ColumnSizingState>; // plugs straight into onColumnSizingChange
}

/**
 * Persists column widths in localStorage, scoped per (experiment, dataset)
 * — so each dataset can have its own remembered set. Falls back to `defaultWidths` when
 * nothing is stored (or when the version changes).
 *
 * Why localStorage: synchronous read on mount avoids a column-flicker on first paint;
 * IndexedDB would require an async read. Sizes are tiny (<200B per dataset).
 */
export const usePersistedColumnWidths = ({
  experimentId,
  datasetId,
}: UsePersistedTableColumnWidthsParams): PersistedColumnWidths => {
  const key = datasetId
    ? `${COLUMN_WIDTH_STORAGE_KEY_PREFIX}.${experimentId}.${datasetId}`
    : `${COLUMN_WIDTH_STORAGE_KEY_PREFIX}.${experimentId}`;

  const [stored, setStored] = useLocalStorage<ColumnSizingState>({
    key,
    version: COLUMN_WIDTH_STORAGE_VERSION,
    initialValue: {},
  });

  return {
    columnSizing: stored,
    setColumnSizing: setStored,
  };
};
