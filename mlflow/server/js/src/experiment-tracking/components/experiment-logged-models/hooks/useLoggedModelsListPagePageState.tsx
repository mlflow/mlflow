import { first, isEmpty, isEqual } from 'lodash';
import { useCallback, useReducer, useState } from 'react';
import { RUNS_VISIBILITY_MODE } from '../../experiment-page/models/ExperimentPageUIState';
import { isLoggedModelRowHidden } from './useExperimentLoggedModelListPageRowVisibility';
import type { LoggedModelMetricDataset } from '../../../types';
import { ExperimentLoggedModelListPageKnownColumns } from './useExperimentLoggedModelListPageTableColumns';
import { useSafeDeferredValue } from '../../../../common/hooks/useSafeDeferredValue';
import type { LoggedModelsTableGroupByMode } from '../ExperimentLoggedModelListPageTable.utils';

type ActionType =
  | { type: 'SET_ORDER_BY'; orderByColumn: string; orderByAsc: boolean }
  | { type: 'SET_GROUP_BY'; groupBy?: LoggedModelsTableGroupByMode }
  | { type: 'SET_COLUMN_VISIBILITY'; columnVisibility: Record<string, boolean> }
  | { type: 'TOGGLE_DATASET'; dataset: LoggedModelMetricDataset }
  | { type: 'CLEAR_DATASETS' }
  | { type: 'SET_RUN_VISIBILITY'; visibilityMode?: RUNS_VISIBILITY_MODE; rowUuid?: string; rowIndex?: number };

/**
 * Defines current state of the logged models table.
 */
export type LoggedModelsListPageState = {
  orderByColumn?: string;
  orderByAsc: boolean;
  columnVisibility?: Record<string, boolean>;
  rowVisibilityMode: RUNS_VISIBILITY_MODE;
  rowVisibilityMap?: Record<string, boolean>;
  selectedFilterDatasets?: LoggedModelMetricDataset[];
  searchQuery?: string;
  groupBy?: LoggedModelsTableGroupByMode;
};

export const LoggedModelsListPageSortableColumns: string[] = [ExperimentLoggedModelListPageKnownColumns.CreationTime];

/**
 * Provides state management for the logged models table.
 */
export const useLoggedModelsListPageState = () => {
  const [state, dispatch] = useReducer(
    (state: LoggedModelsListPageState, action: ActionType): LoggedModelsListPageState => {
      if (action.type === 'SET_ORDER_BY') {
        return { ...state, orderByColumn: action.orderByColumn, orderByAsc: action.orderByAsc };
      }
      if (action.type === 'SET_GROUP_BY') {
        return { ...state, groupBy: action.groupBy };
      }
      if (action.type === 'SET_COLUMN_VISIBILITY') {
        return { ...state, columnVisibility: action.columnVisibility };
      }
      if (action.type === 'CLEAR_DATASETS') {
        return { ...state, selectedFilterDatasets: [] };
      }
      if (action.type === 'TOGGLE_DATASET') {
        return {
          ...state,
          selectedFilterDatasets: state.selectedFilterDatasets?.some((dataset) => isEqual(dataset, action.dataset))
            ? state.selectedFilterDatasets?.filter((dataset) => !isEqual(dataset, action.dataset))
            : [...(state.selectedFilterDatasets ?? []), action.dataset],
        };
      }
      if (action.type === 'SET_RUN_VISIBILITY') {
        if (action.visibilityMode) {
          return { ...state, rowVisibilityMode: action.visibilityMode, rowVisibilityMap: {} };
        }
        if (action.rowUuid && action.rowIndex !== undefined) {
          const currentHidden = isLoggedModelRowHidden(
            state.rowVisibilityMode,
            action.rowUuid,
            action.rowIndex,
            state.rowVisibilityMap ?? {},
          );
          return { ...state, rowVisibilityMap: { ...state.rowVisibilityMap, [action.rowUuid]: currentHidden } };
        }
      }
      return state;
    },
    {
      orderByColumn: first(LoggedModelsListPageSortableColumns),
      orderByAsc: false,
      columnVisibility: {},
      rowVisibilityMode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
    },
  );

  const setOrderBy = useCallback(
    (orderByColumn: string, orderByAsc: boolean) => dispatch({ type: 'SET_ORDER_BY', orderByColumn, orderByAsc }),
    [],
  );

  const setColumnVisibility = useCallback(
    (columnVisibility: Record<string, boolean>) => dispatch({ type: 'SET_COLUMN_VISIBILITY', columnVisibility }),
    [],
  );

  const setRowVisibilityMode = useCallback(
    (visibilityMode: RUNS_VISIBILITY_MODE) => dispatch({ type: 'SET_RUN_VISIBILITY', visibilityMode }),
    [],
  );

  const toggleRowVisibility = useCallback(
    (rowUuid: string, rowIndex: number) => dispatch({ type: 'SET_RUN_VISIBILITY', rowUuid, rowIndex }),
    [],
  );

  const toggleDataset = useCallback(
    (dataset: LoggedModelMetricDataset) => dispatch({ type: 'TOGGLE_DATASET', dataset }),
    [],
  );

  const setGroupBy = useCallback(
    (groupBy?: LoggedModelsTableGroupByMode) => dispatch({ type: 'SET_GROUP_BY', groupBy }),
    [],
  );

  const clearSelectedDatasets = useCallback(() => dispatch({ type: 'CLEAR_DATASETS' }), []);

  const deferredState = useSafeDeferredValue(state);

  // Search filter state does not go through deferred value
  const [searchQuery, updateSearchQuery] = useState<string>('');

  // To be expanded with other filters in the future
  const isFilteringActive = Boolean(searchQuery || !isEmpty(state.selectedFilterDatasets));

  return {
    state: deferredState,
    isFilteringActive,
    searchQuery,
    setOrderBy,
    setColumnVisibility,
    setRowVisibilityMode,
    toggleRowVisibility,
    updateSearchQuery,
    toggleDataset,
    clearSelectedDatasets,
    setGroupBy,
  };
};
