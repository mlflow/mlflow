import { isNil } from 'lodash';
import { useCallback, useEffect, useMemo, useRef } from 'react';

import { useLocalStorage } from '../../hooks/useLocalStorage';

import { useColumnsURL } from './useColumnsURL';
import {
  EXECUTION_DURATION_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  LINKED_PROMPTS_COLUMN_ID,
  RUN_NAME_COLUMN_ID,
  SOURCE_COLUMN_ID,
  STATE_COLUMN_ID,
  TAGS_COLUMN_ID,
  TRACE_NAME_COLUMN_ID,
  USER_COLUMN_ID,
} from './useTableColumns';
import { shouldEnableTracesTableStatePersistence } from '../../model-trace-explorer/FeatureUtils';
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

// Low-priority info columns that are hidden before assessment columns when the number of
// visible columns exceeds DEFAULT_MAX_VISIBLE_COLUMNS. Assessment columns are prioritized
// over these columns because they are the primary reason users view the traces table.
// These columns are informational/metadata and less critical for evaluation workflows:
// - tags: supplementary metadata
// - prompt (linked prompts): supplementary context
// - logged_model: version info, less relevant during evaluation
// - run_name: run provenance, not directly relevant to evaluation results
// - user: session metadata, not directly relevant to evaluation results
const LOW_PRIORITY_COLUMN_IDS = [
  TAGS_COLUMN_ID,
  LINKED_PROMPTS_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  RUN_NAME_COLUMN_ID,
  USER_COLUMN_ID,
];

// This function adjusts the hidden columns to ensure that the number of visible columns is at most DEFAULT_MAX_VISIBLE_COLUMNS.
// If over the limit, it removes low-priority info columns first, then assessment columns, then high-priority columns.
const adjustHiddenColumns = (hiddenColumns: string[], allColumns: TracesTableColumn[]): string[] => {
  let visibleColumns = toVisibleColumnsFromHiddenColumns(hiddenColumns, allColumns);
  if (visibleColumns.length > DEFAULT_MAX_VISIBLE_COLUMNS) {
    const assessmentColumns = visibleColumns.filter((col) => col.type === TracesTableColumnType.ASSESSMENT);
    const lowPriorityColumns = visibleColumns.filter(
      (col) => col.type !== TracesTableColumnType.ASSESSMENT && LOW_PRIORITY_COLUMN_IDS.includes(col.id),
    );
    const highPriorityColumns = visibleColumns.filter(
      (col) => col.type !== TracesTableColumnType.ASSESSMENT && !LOW_PRIORITY_COLUMN_IDS.includes(col.id),
    );

    let columnsToRemove = visibleColumns.length - DEFAULT_MAX_VISIBLE_COLUMNS;

    // First remove low-priority info columns to make room for assessment columns
    const lowPriorityToKeep = Math.max(0, lowPriorityColumns.length - columnsToRemove);
    columnsToRemove = Math.max(0, columnsToRemove - lowPriorityColumns.length);

    // If still over limit, remove assessment columns from the tail
    const assessmentColumnsToKeep = Math.max(0, assessmentColumns.length - columnsToRemove);
    columnsToRemove = Math.max(0, columnsToRemove - assessmentColumns.length);

    // If still over limit, trim high-priority columns as a last resort
    const highPriorityToKeep = Math.max(0, highPriorityColumns.length - columnsToRemove);

    visibleColumns = [
      ...highPriorityColumns.slice(0, highPriorityToKeep),
      ...lowPriorityColumns.slice(0, lowPriorityToKeep),
      ...assessmentColumns.slice(0, assessmentColumnsToKeep),
    ];
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
  storageKeyPrefix?: string,
): {
  hiddenColumns: string[];
  toggleColumns: (cols: TracesTableColumn[]) => void;
} => {
  const enableURLPersistence = shouldEnableTracesTableStatePersistence();

  const storageKey = storageKeyPrefix
    ? `${LOCAL_STORAGE_KEY}-${storageKeyPrefix}-${experimentId}-${runUuid}`
    : `${LOCAL_STORAGE_KEY}-${experimentId}-${runUuid}`;

  const [columnState, setColumnState] = useLocalStorage<GenAITracesUIState | undefined>({
    key: storageKey,
    version: LOCAL_STORAGE_VERSION,
    initialValue: undefined,
  });

  const [urlColumnIds, setUrlColumnIds] = useColumnsURL();

  const defaultHidden = useMemo(() => {
    return getDefaultHiddenColumns(allColumns, defaultSelectedColumns);
  }, [allColumns, defaultSelectedColumns]);

  const hiddenColumns = useMemo(() => {
    if (enableURLPersistence && urlColumnIds && urlColumnIds.length > 0) {
      const urlSelectedSet = new Set(urlColumnIds);
      return allColumns.filter((col) => !urlSelectedSet.has(col.id)).map((col) => col.id);
    }

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
  }, [enableURLPersistence, columnState, defaultHidden, urlColumnIds, allColumns]);

  const toggleColumns = useCallback(
    (cols: TracesTableColumn[]) => {
      if (!cols.length) return;

      setColumnState((prev) => {
        const prevOverrides = prev?.columnOverrides ?? {};
        const nextOverrides = { ...prevOverrides };
        let changed = false;

        cols.forEach((col) => {
          const currentlyHidden = hiddenColumns.includes(col.id);
          const newShow = currentlyHidden;

          if (nextOverrides[col.id] !== newShow) {
            nextOverrides[col.id] = newShow;
            changed = true;
          }
        });

        return changed ? { ...prev, columnOverrides: nextOverrides } : prev;
      });

      if (enableURLPersistence) {
        const currentHiddenSet = new Set(hiddenColumns);
        cols.forEach((col) => {
          if (currentHiddenSet.has(col.id)) {
            currentHiddenSet.delete(col.id);
          } else {
            currentHiddenSet.add(col.id);
          }
        });

        const newSelectedIds = allColumns.filter((col) => !currentHiddenSet.has(col.id)).map((col) => col.id);
        // Use replace=true: column visibility is UI config, not navigation state
        setUrlColumnIds(newSelectedIds, true);
      }
    },
    [enableURLPersistence, setColumnState, hiddenColumns, allColumns, setUrlColumnIds],
  );

  // Migration: sync localStorage state to URL on initial mount
  const hasSyncedToURL = useRef(false);
  useEffect(() => {
    if (
      enableURLPersistence &&
      !hasSyncedToURL.current &&
      (!urlColumnIds || urlColumnIds.length === 0) &&
      allColumns.length > 0
    ) {
      hasSyncedToURL.current = true;
      const selectedIds = allColumns.filter((col) => !hiddenColumns.includes(col.id)).map((col) => col.id);
      if (selectedIds.length > 0) {
        setUrlColumnIds(selectedIds, true);
      }
    }
  }, [enableURLPersistence, urlColumnIds, hiddenColumns, allColumns, setUrlColumnIds]);

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
  storageKeyPrefix?: string,
) => {
  const { hiddenColumns, toggleColumns } = useGenAITracesUIStateColumns(
    experimentId,
    allColumns,
    defaultSelectedColumns,
    runUuid,
    storageKeyPrefix,
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
