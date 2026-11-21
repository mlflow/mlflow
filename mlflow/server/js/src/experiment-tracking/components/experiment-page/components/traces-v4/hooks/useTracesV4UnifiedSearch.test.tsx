import { describe, expect, jest, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { FilterOp, type TraceFilterModel } from '@databricks/web-shared/traces-table';
import { useTracesV4UnifiedSearch } from './useTracesV4UnifiedSearch';
import type { UseDebouncedSearchInputResult } from './useDebouncedSearchInput';

const stateClause = { field: 'status', operator: FilterOp.EQUALS, value: 'ERROR' };
const urlTagClause = { field: 'tag', key: 'env', operator: FilterOp.EQUALS, value: 'prod' };

const renderUnifiedSearch = () => {
  const setFilterState = jest.fn();
  const { result } = renderHook(() =>
    useTracesV4UnifiedSearch({
      filterModel: [stateClause],
      setFilterState,
      tagFilters: [{ key: 'env', value: 'prod' }],
      searchInput: {} as UseDebouncedSearchInputResult,
    }),
  );
  return { result, setFilterState };
};

describe('useTracesV4UnifiedSearch', () => {
  test('shows the URL tag filters as pills after the filter model', () => {
    const { result } = renderUnifiedSearch();
    expect(result.current.searchFilterModel).toEqual([stateClause, urlTagClause]);
  });

  test('keeps the URL tag filters in the URL when another pill is removed', () => {
    const { result, setFilterState } = renderUnifiedSearch();

    act(() => result.current.setSearchFilters([urlTagClause]));

    expect(setFilterState).toHaveBeenCalledWith([], [{ key: 'env', value: 'prod' }]);
  });

  test('keeps the URL tag filters in the URL when another pill is edited', () => {
    const { result, setFilterState } = renderUnifiedSearch();
    const editedClause = { ...stateClause, value: 'OK' };

    act(() => result.current.setSearchFilters([editedClause, urlTagClause]));

    expect(setFilterState).toHaveBeenCalledWith([editedClause], [{ key: 'env', value: 'prod' }]);
  });

  test('removes a URL tag filter from the URL when its own pill is removed', () => {
    const { result, setFilterState } = renderUnifiedSearch();

    act(() => result.current.setSearchFilters([stateClause]));

    expect(setFilterState).toHaveBeenCalledWith([stateClause], []);
  });

  test('moves an edited URL tag pill into the filter model', () => {
    const { result, setFilterState } = renderUnifiedSearch();
    const editedTag = { ...urlTagClause, value: 'staging' };

    act(() => result.current.setSearchFilters([stateClause, editedTag]));

    expect(setFilterState).toHaveBeenCalledWith([stateClause, editedTag], []);
  });
});
