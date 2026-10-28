import { useTraceColumnSizing, type TraceColumnSizing } from '@databricks/web-shared/traces-table';
import { TRACE_COLUMN_SIZES_STORAGE_KEY_PREFIX } from '../utils/constants';

// Bump when the column set or sizing scheme changes so stale pixel widths reset.
// v3 → narrower Trace ID / Session defaults.
export const COLUMN_SIZES_STORAGE_VERSION = 3;

/**
 * MLflow adapter over the shared `useTraceColumnSizing`: keeps the MLflow storage-key prefix (scoped
 * per experiment) and the v3 schema version. The uncontrolled-sizing settle-persist behavior lives
 * in the shared hook.
 */
export const useTracesV4ColumnSizing = (experimentId: string): TraceColumnSizing =>
  useTraceColumnSizing({
    storageKey: `${TRACE_COLUMN_SIZES_STORAGE_KEY_PREFIX}.${experimentId}`,
    version: COLUMN_SIZES_STORAGE_VERSION,
  });
