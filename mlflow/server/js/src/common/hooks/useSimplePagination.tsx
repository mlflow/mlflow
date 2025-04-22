import { useCallback, useReducer } from 'react';

type SimplePaginationState = {
  pageSize: number;
  pageIndex: number;
};

type SimplePaginationAction =
  | {
      type: 'SET_CURRENT_PAGE_INDEX';
      pageIndex: number;
    }
  | {
      type: 'SET_PAGE_SIZE';
      pageSize?: number;
    };

const paginationReducer = (state: SimplePaginationState, action: SimplePaginationAction): SimplePaginationState => {
  if (action.type === 'SET_PAGE_SIZE') {
    if (!action.pageSize || state.pageSize === action.pageSize) {
      return state;
    }

    return {
      ...state,
      pageSize: action.pageSize,
      pageIndex: 1,
    };
  }
  if (action.type === 'SET_CURRENT_PAGE_INDEX') {
    return {
      ...state,
      pageIndex: action.pageIndex,
    };
  }
  return state;
};

/**
 * Simple pagination hook that manages the current page index and page size.
 */
export const useSimplePagination = (initialPageSize = 1) => {
  const [{ pageIndex, pageSize }, dispatch] = useReducer(paginationReducer, {
    pageSize: initialPageSize,
    pageIndex: 1,
  });

  const setPageSize = useCallback((pageSize?: number) => dispatch({ type: 'SET_PAGE_SIZE', pageSize }), []);
  const setCurrentPageIndex = useCallback(
    (pageIndex: number) => dispatch({ type: 'SET_CURRENT_PAGE_INDEX', pageIndex }),
    [],
  );

  return {
    pageIndex,
    pageSize,
    setPageSize,
    setCurrentPageIndex,
  };
};
