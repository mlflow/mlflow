import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import type { TraceComparisonRow } from '@databricks/web-shared/traces-table';

/** The physical traces in one run slot of a matched comparison row. */
export const getComparisonRunTraces = (run: TraceComparisonRow['runs'][number]): ModelTraceInfoV3[] =>
  run.trace ? [run.trace] : [];
