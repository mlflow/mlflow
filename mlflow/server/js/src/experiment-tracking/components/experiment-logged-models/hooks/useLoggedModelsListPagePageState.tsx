import { first, identity, isFunction } from 'lodash';
import React, { useCallback, useReducer } from 'react';
import { RUNS_VISIBILITY_MODE } from '../../experiment-page/models/ExperimentPageUIState';
import { isLoggedModelRowHidden } from './useExperimentLoggedModelListPageRowVisibility';

type ActionType =
  | { type: 'SET_ORDER_BY'; orderByField: string; orderByAsc: boolean }
  | { type: 'SET_COLUMN_VISIBILITY'; columnVisibility: Record<string, boolean> }
  | { type: 'SET_RUN_VISIBILITY'; visibilityMode?: RUNS_VISIBILITY_MODE; rowUuid?: string; rowIndex?: number }
  | { type: 'SET_ABC'; abc: string };

const useSafeDeferredValue: <T>(value: T) => T =
  'useDeferredValue' in React && isFunction(React.useDeferredValue) ? React.useDeferredValue : identity;

/**
 * Defines current state of the logged models table.
 */
export type LoggedModelsListPageState = {
  orderByField?: string;
  orderByAsc: boolean;
  columnVisibility?: Record<string, boolean>;
  rowVisibilityMode: RUNS_VISIBILITY_MODE;
  rowVisibilityMap?: Record<string, boolean>;
};

export const LoggedModelsListPageSortableColumns = ['creation_time'];

/**
 * Provides state management for the logged models table.
 */
export const useLoggedModelsListPageState = () => {
  const [state, dispatch] = useReducer(
    (state: LoggedModelsListPageState, action: ActionType): LoggedModelsListPageState => {
      if (action.type === 'SET_ORDER_BY') {
        return { ...state, orderByField: action.orderByField, orderByAsc: action.orderByAsc };
      }
      if (action.type === 'SET_COLUMN_VISIBILITY') {
        return { ...state, columnVisibility: action.columnVisibility };
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
      orderByField: first(LoggedModelsListPageSortableColumns),
      orderByAsc: false,
      columnVisibility: {},
      rowVisibilityMode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
    },
  );

  const setOrderBy = useCallback(
    (orderByField: string, orderByAsc: boolean) => dispatch({ type: 'SET_ORDER_BY', orderByField, orderByAsc }),
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

  const deferredState = useSafeDeferredValue(state);

  return { state: deferredState, setOrderBy, setColumnVisibility, setRowVisibilityMode, toggleRowVisibility };
};
