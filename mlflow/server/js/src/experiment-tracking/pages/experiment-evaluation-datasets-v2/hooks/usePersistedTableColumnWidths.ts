import { useMemo } from 'react';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import type { ColumnSizingState, OnChangeFn } from '@tanstack/react-table';

const COLUMN_WIDTH_STORAGE_KEY_PREFIX = 'mlflow.eval-datasets.column-widths';

// Bumping this constant invalidates every stored entry across all users —
// `useLocalStorage` keys are namespaced by version, so old keys become
// unreachable. Bump only when the stored shape is incompatible (e.g., if we
// ever switch from absolute px to proportions). No migration path is offered;
// users get the bucket defaults on first load after a bump.
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
 * Filters out non-positive-finite entries so corrupt/old localStorage values
 * (NaN, strings, null, negatives, 0) can't flow into `flex: 0 0 ${px}px` and
 * collapse a column. Returns the same reference when nothing was filtered, so
 * downstream React hooks that depend on `columnSizing` identity stay stable.
 */
const sanitiseStoredWidths = (stored: unknown): ColumnSizingState => {
  if (stored === null || typeof stored !== 'object') return {};
  const entries = Object.entries(stored as Record<string, unknown>).filter(
    ([, value]) => typeof value === 'number' && Number.isFinite(value) && value > 0,
  );
  return Object.fromEntries(entries) as ColumnSizingState;
};

/**
 * Persists column widths in localStorage, scoped per (experiment, dataset)
 * so each dataset can have its own remembered set. Starts empty (`{}`) and
 * only the columns the user actually drags get written — auto-defaults live
 * on the consumer (`COLUMN_WIDTHS_BY_BREAKPOINT` in `DatasetRecordsTable`).
 *
 * Why localStorage: synchronous read on mount avoids a column-flicker on
 * first paint; IndexedDB would require an async read. Sizes are tiny
 * (<200B per dataset).
 */
export const usePersistedTableColumnWidths = ({
  experimentId,
  datasetId,
}: UsePersistedTableColumnWidthsParams): PersistedColumnWidths => {
  const key = `${COLUMN_WIDTH_STORAGE_KEY_PREFIX}.${experimentId}.${datasetId}`;

  const [stored, setStored] = useLocalStorage<ColumnSizingState>({
    key,
    version: COLUMN_WIDTH_STORAGE_VERSION,
    initialValue: {},
  });

  // Sanitise before exposing — protects every reader from corrupt entries
  // (NaN, strings, negatives) that the schema-less storage layer might admit.
  // Memo by `stored` so a clean run of widths keeps a stable reference.
  const columnSizing = useMemo(() => sanitiseStoredWidths(stored), [stored]);

  return {
    columnSizing,
    setColumnSizing: setStored,
  };
};
