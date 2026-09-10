import { useCallback, useMemo } from 'react';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import { isTraceColumnId, TRACE_COLUMN_IDS, type TraceColumnId } from '@databricks/web-shared/traces-table';
import { isAssessmentColumnId } from '../utils/assessmentColumns';
import { TRACE_COLUMN_ORDER_STORAGE_KEY_PREFIX } from '../utils/constants';

const COLUMN_ORDER_STORAGE_VERSION = 1;

export interface TracesV4ColumnOrder {
  /** Complete mixed order, including hidden standard and known assessment columns. */
  columnOrder: string[];
  reorderColumn: (activeColumn: string, targetColumn: string) => void;
  /** Restores an explicitly saved order; an older view with no order leaves the preference intact. */
  setColumnOrder: (columnOrder: string[] | undefined) => void;
  normalizeColumnOrder: (columnOrder: unknown) => string[];
  reset: () => void;
}

/** Persists one order across fixed standard columns and page-dependent assessment columns. */
export const useTracesV4ColumnOrder = (
  experimentId: string,
  standardColumnOrder: TraceColumnId[],
  assessmentColumnIds: string[],
): TracesV4ColumnOrder => {
  const normalizeColumnOrder = useCallback(
    (storedOrder: unknown): string[] => {
      const candidateOrder = Array.isArray(storedOrder) ? storedOrder : [];
      const seen = new Set<string>();
      const normalized: string[] = [];
      for (const id of [...candidateOrder, ...standardColumnOrder, ...assessmentColumnIds]) {
        if (typeof id === 'string' && (isTraceColumnId(id) || isAssessmentColumnId(id)) && !seen.has(id)) {
          normalized.push(id);
          seen.add(id);
        }
      }
      return normalized;
    },
    [standardColumnOrder, assessmentColumnIds],
  );

  const [storedColumnOrder, setStoredColumnOrder] = useLocalStorage<unknown>({
    key: `${TRACE_COLUMN_ORDER_STORAGE_KEY_PREFIX}.${experimentId}`,
    version: COLUMN_ORDER_STORAGE_VERSION,
    initialValue: [...standardColumnOrder, ...assessmentColumnIds],
  });
  const columnOrder = useMemo(() => normalizeColumnOrder(storedColumnOrder), [normalizeColumnOrder, storedColumnOrder]);

  const reorderColumn = useCallback(
    (activeColumn: string, targetColumn: string) => {
      if (activeColumn === targetColumn) {
        return;
      }
      setStoredColumnOrder((previousOrder: unknown) => {
        const nextOrder = normalizeColumnOrder(previousOrder);
        const activeIndex = nextOrder.indexOf(activeColumn);
        const targetIndex = nextOrder.indexOf(targetColumn);
        if (activeIndex === -1 || targetIndex === -1) {
          return nextOrder;
        }
        nextOrder.splice(activeIndex, 1);
        nextOrder.splice(targetIndex, 0, activeColumn);
        return nextOrder;
      });
    },
    [normalizeColumnOrder, setStoredColumnOrder],
  );

  const setColumnOrder = useCallback(
    (savedColumnOrder: string[] | undefined) => {
      if (savedColumnOrder !== undefined) {
        setStoredColumnOrder(normalizeColumnOrder(savedColumnOrder));
      }
    },
    [normalizeColumnOrder, setStoredColumnOrder],
  );

  const reset = useCallback(
    () => setStoredColumnOrder(normalizeColumnOrder([...TRACE_COLUMN_IDS, ...assessmentColumnIds])),
    [assessmentColumnIds, normalizeColumnOrder, setStoredColumnOrder],
  );

  return {
    columnOrder,
    reorderColumn,
    setColumnOrder,
    normalizeColumnOrder,
    reset,
  };
};
