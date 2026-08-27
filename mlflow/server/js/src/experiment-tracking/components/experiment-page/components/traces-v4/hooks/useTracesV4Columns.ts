import { useCallback } from 'react';
import {
  useTraceColumnVisibility,
  type TraceColumnId,
  type TraceColumnVisibility,
} from '@databricks/web-shared/traces-table';
import { TRACE_COLUMN_STORAGE_KEY_PREFIX } from '../utils/constants';

// Bump when the stored schema changes so stale entries reset. v2 → override map (was a flat visible
// list); v3 → new default-visibility set; v4 → Tokens now shown by default. New opt-in columns
// do not require a bump because absent overrides naturally use their hidden default.
const COLUMN_STORAGE_VERSION = 4;

const OPT_IN_COLUMNS: ReadonlySet<TraceColumnId> = new Set([
  'trace_id',
  'trace_name',
  'user',
  'source',
  'run_name',
  'cost',
]);

export interface UseTracesV4ColumnsParams {
  /** True when the current page has at least one session-tagged trace — drives the Session column's default. */
  hasSessionOnPage: boolean;
}

/**
 * MLflow adapter over the shared `useTraceColumnVisibility`: keeps the MLflow storage-key prefix
 * (scoped per experiment), the v4 schema version, and the data-driven Session default (Session shows
 * only when the current page carries sessions; opt-in columns hidden by default). The
 * sticky-override + reset behavior lives entirely in the shared hook.
 */
export const useTracesV4Columns = (
  experimentId: string,
  { hasSessionOnPage }: UseTracesV4ColumnsParams,
): TraceColumnVisibility => {
  const getDefaultVisible = useCallback(
    (id: TraceColumnId) => {
      if (id === 'session') {
        return hasSessionOnPage;
      }
      return !OPT_IN_COLUMNS.has(id);
    },
    [hasSessionOnPage],
  );

  return useTraceColumnVisibility({
    storageKey: `${TRACE_COLUMN_STORAGE_KEY_PREFIX}.${experimentId}`,
    version: COLUMN_STORAGE_VERSION,
    getDefaultVisible,
  });
};
