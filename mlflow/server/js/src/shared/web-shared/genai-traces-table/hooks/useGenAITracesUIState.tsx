import { isNil } from 'lodash';
import { useCallback, useMemo } from 'react';

import { useLocalStorage } from '@databricks/web-shared/hooks';

import {
  EXECUTION_DURATION_COLUMN_ID,
  SOURCE_COLUMN_ID,
  STATE_COLUMN_ID,
  TRACE_NAME_COLUMN_ID,
} from './useTableColumns';
import type { TracesTableColumn } from '../types';
import { TracesTableColumnType } from '../types';

export interface GenAITracesUIState {
  /**
   * A map of column ids to boolean values.
   * If a column id is present in the map, the column is hidden if the value is false, and visible if the value is true.
   */
  columnOverrides: Record<string, boolean>;
}

const DEFAULT_MAX_VISIBLE_COLUMNS = 10;

const LOCAL_STORAGE_KEY = 'genaiTracesUIState-columns';
const LOCAL_STORAGE_VERSION = 1;

const toVisibleColumnsFromHiddenColumns = (hiddenColumns: string[], allColumns: TracesTableColumn[]) => {
  return allColumns.filter((col) => !hiddenColumns.includes(col.id));
};

const toHiddenColumnsFromVisibleColumns = (visibleColumns: TracesTableColumn[], allColumns: TracesTableColumn[]) => {
  return allColumns.filter((col) => !visibleColumns.includes(col)).map((col) => col.id);
};

// This function adjusts the hidden columns to ensure that the number of visible columns is at most DEFAULT_MAX_VISIBLE_COLUMNS
// If over the limit, it removes assessment columns until the limit is met.
const adjustHiddenColumns = (hiddenColumns: string[], allColumns: TracesTableColumn[]): string[] => {
  let visibleColumns = toVisibleColumnsFromHiddenColumns(hiddenColumns, allColumns);
  if (visibleColumns.length > DEFAULT_MAX_VISIBLE_COLUMNS) {
    const assessmentColumns = visibleColumns.filter((col) => col.type === TracesTableColumnType.ASSESSMENT);
    const nonAssessmentColumns = visibleColumns.filter((col) => col.type !== TracesTableColumnType.ASSESSMENT);

    // Calculate how many assessment columns we need to remove
    const columnsToRemove = visibleColumns.length - DEFAULT_MAX_VISIBLE_COLUMNS;
    const assessmentColumnsToKeep = Math.max(0, assessmentColumns.length - columnsToRemove);

    // Keep the first N assessment columns and all non-assessment columns
    visibleColumns = [...nonAssessmentColumns, ...assessmentColumns.slice(0, assessmentColumnsToKeep)];
  }
  return toHiddenColumnsFromVisibleColumns(visibleColumns, allColumns);
};

const getDefaultHiddenColumns = (
  allColumns: TracesTableColumn[],
  defaultSelectedColumns?: (allColumns: TracesTableColumn[]) => TracesTableColumn[],
): string[] => {
  if (defaultSelectedColumns) {
    return adjustHiddenColumns(
      toHiddenColumnsFromVisibleColumns(defaultSelectedColumns(allColumns), allColumns),
      allColumns,
    );
  }

  return adjustHiddenColumns(
    [TRACE_NAME_COLUMN_ID, SOURCE_COLUMN_ID, EXECUTION_DURATION_COLUMN_ID, STATE_COLUMN_ID],
    allColumns,
  );
};

export const useGenAITracesUIStateColumns = (
  experimentId: string,
  allColumns: TracesTableColumn[],
  defaultSelectedColumns?: (allColumns: TracesTableColumn[]) => TracesTableColumn[],
  runUuid?: string,
): {
  hiddenColumns: string[];
  toggleColumns: (cols: TracesTableColumn[]) => void;
} => {
  const [columnState, setColumnState] = useLocalStorage<GenAITracesUIState | undefined>({
    key: `${LOCAL_STORAGE_KEY}-${experimentId}-${runUuid}`,
    version: LOCAL_STORAGE_VERSION,
    initialValue: undefined,
  });

  const defaultHidden = useMemo(() => {
    return getDefaultHiddenColumns(allColumns, defaultSelectedColumns);
  }, [allColumns, defaultSelectedColumns]);

  const hiddenColumns = useMemo(() => {
    const hidden = new Set(defaultHidden);

    if (!columnState?.columnOverrides) {
      return defaultHidden;
    }

    Object.entries(columnState.columnOverrides).forEach(([id, show]) => {
      if (show) {
        hidden.delete(id);
      } else {
        hidden.add(id);
      }
    });

    return [...hidden];
  }, [columnState, defaultHidden]);

  const toggleColumns = useCallback(
    (cols: TracesTableColumn[]) => {
      if (!cols.length) return;

      setColumnState((prev) => {
        const prevOverrides = prev?.columnOverrides ?? {};
        const nextOverrides = { ...prevOverrides };
        let changed = false;

        cols.forEach((col) => {
          /* derive current visibility:  true = hidden, false = visible */
          const currentlyHidden =
            col.id in prevOverrides
              ? !prevOverrides[col.id] // invert, because stored flag is *show*
              : defaultHidden.includes(col.id);

          const newShow = currentlyHidden;
          if (nextOverrides[col.id] !== newShow) {
            nextOverrides[col.id] = newShow;
            changed = true;
          }
        });

        return changed ? { ...prev, columnOverrides: nextOverrides } : prev;
      });
    },
    [setColumnState, defaultHidden],
  );

  return { hiddenColumns, toggleColumns };
};

/**
 * This hook is used to manage the selected columns for the GenAITracesTable.
 * It uses the useGenAITracesUIStateColumns hook to manage the hidden columns, and then filters the allColumns to get the selected columns.
 * It also provides a function to set the selected columns, which will toggle the columns to be visible or hidden.
 * @returns selectedColumns: The selected columns
 * @returns toggleColumns: A function to toggle the columns
 * @returns setSelectedColumns: A function to set the selected columns - can use to do bulk updates to selected columns
 */
export const useSelectedColumns = (
  experimentId: string,
  allColumns: TracesTableColumn[],
  defaultSelectedColumns?: (cols: TracesTableColumn[]) => TracesTableColumn[],
  runUuid?: string,
) => {
  const { hiddenColumns, toggleColumns } = useGenAITracesUIStateColumns(
    experimentId,
    allColumns,
    defaultSelectedColumns,
    runUuid,
  );

  const selectedColumns = useMemo(
    () => allColumns.filter((c) => !hiddenColumns.includes(c.id)),
    [allColumns, hiddenColumns],
  );

  const setSelectedColumns = useCallback(
    (nextSelected: TracesTableColumn[]) => {
      if (isNil(nextSelected)) return;

      const wantSelected = new Set(nextSelected.map((c) => c.id));
      const toToggle: TracesTableColumn[] = [];

      allColumns.forEach((col) => {
        const isSelectedNow = !hiddenColumns.includes(col.id);
        const willBeSelected = wantSelected.has(col.id);

        if (isSelectedNow !== willBeSelected) {
          // only flip what actually changes
          toToggle.push(col);
        }
      });

      if (toToggle.length) {
        toggleColumns(toToggle);
      }
    },
    [allColumns, hiddenColumns, toggleColumns],
  );

  return { selectedColumns, toggleColumns, setSelectedColumns };
};
