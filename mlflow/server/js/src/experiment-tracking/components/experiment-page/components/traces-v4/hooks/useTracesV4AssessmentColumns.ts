import { useCallback, useMemo } from 'react';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import { useArrayMemo, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import type { GenericColumnOption, TraceTableColumn } from '@databricks/web-shared/traces-table';
import {
  assessmentColumnId,
  assessmentNameFromColumnId,
  computeAssessmentColumns,
  extractTraceIssues,
} from '../utils/assessmentColumns';
import { buildAssessmentColumnDefs } from '../utils/buildAssessmentColumnDefs';
import { buildIssuesColumnDef } from '../utils/buildIssuesColumnDef';
import { TRACE_ASSESSMENT_COLUMN_STORAGE_KEY_PREFIX } from '../utils/constants';

// Bump when the stored schema changes so stale entries reset.
const ASSESSMENT_COLUMN_STORAGE_VERSION = 1;

// Assessment names are dynamic, so their checkbox `componentId` can't be a per-item literal; the
// static-componentId lint rule needs a statically determinable value, not a unique one — so every
// assessment item shares this single literal.
const ASSESSMENT_ITEM_COMPONENT_ID = 'mlflow.traces-v4.column-selector.assessment-item';

export interface TracesV4AssessmentColumns {
  /** Visible assessment column defs, appended to the table via `extraColumns`. */
  columnDefs: TraceTableColumn[];
  /** Every candidate assessment name (visible or not); one filter field is offered per name. */
  candidateNames: string[];
  /** Column-selector options for every candidate assessment (visible or not). */
  selectorOptions: GenericColumnOption[];
  /** Namespaced ids of the currently-visible assessment columns. */
  visibleIds: string[];
  /** Toggle an assessment column's visibility by its namespaced id. */
  toggle: (id: string) => void;
  /** Clear all assessment overrides (return every column to its default visibility). */
  reset: () => void;
}

/**
 * Assessment-column selection for the V4 traces tab: a thin sibling of `useTracesV4Columns`. Persists
 * per-column opt-in/opt-out overrides per experiment (default visible), and derives the visible
 * columns + selector options from the current page's traces via `computeAssessmentColumns`.
 */
export const useTracesV4AssessmentColumns = (
  experimentId: string,
  traces: ModelTraceInfoV3[],
): TracesV4AssessmentColumns => {
  const [overrides, setOverrides] = useLocalStorage<Record<string, boolean>>({
    key: `${TRACE_ASSESSMENT_COLUMN_STORAGE_KEY_PREFIX}.${experimentId}`,
    version: ASSESSMENT_COLUMN_STORAGE_VERSION,
    initialValue: {},
  });

  const selection = useMemo(() => computeAssessmentColumns(traces, overrides), [traces, overrides]);
  // Stabilize the derived name arrays by content so the column defs (and thus the memoized table)
  // don't churn when a recompute yields the same names.
  const candidateNames = useArrayMemo(selection.candidateNames);
  const visibleNames = useArrayMemo(selection.visibleNames);

  // The dedicated Issues column shows only when the current page carries detected issues (data-driven,
  // like the Session column) and renders ahead of the assessment columns, matching the prior tab.
  const hasIssuesOnPage = useMemo(() => traces.some((trace) => extractTraceIssues(trace).length > 0), [traces]);
  const columnDefs = useMemo(
    () => [...(hasIssuesOnPage ? [buildIssuesColumnDef()] : []), ...buildAssessmentColumnDefs(visibleNames)],
    [hasIssuesOnPage, visibleNames],
  );
  const visibleIds = useMemo(() => visibleNames.map(assessmentColumnId), [visibleNames]);
  const selectorOptions = useMemo<GenericColumnOption[]>(
    () =>
      candidateNames.map((name) => ({
        id: assessmentColumnId(name),
        label: name,
        componentId: ASSESSMENT_ITEM_COMPONENT_ID,
      })),
    [candidateNames],
  );

  const toggle = useCallback(
    (id: string) => {
      const name = assessmentNameFromColumnId(id);
      setOverrides((prev) => ({ ...prev, [name]: !(prev[name] ?? true) }));
    },
    [setOverrides],
  );

  const reset = useCallback(() => setOverrides({}), [setOverrides]);

  return { columnDefs, candidateNames, selectorOptions, visibleIds, toggle, reset };
};
