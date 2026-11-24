import { describe, expect, jest, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { FilterOp, type TraceFilterModel } from '@databricks/web-shared/traces-table';
import { useTracesV4UnifiedSearch } from './useTracesV4UnifiedSearch';
import type { UseDebouncedSearchInputResult } from './useDebouncedSearchInput';

const stateClause = { field: 'status', operator: FilterOp.EQUALS, value: 'ERROR' };
const urlTagClause = { field: 'tag', key: 'env', operator: FilterOp.EQUALS, value: 'prod' };

const renderUnifiedSearch = () => {
  const setFilterModel = jest.fn<(next: TraceFilterModel) => void>();
  const setTagFilters = jest.fn<(filters: { key: string; value: string }[]) => void>();
  const { result } = renderHook(() =>
    useTracesV4UnifiedSearch({
      filterModel: [stateClause],
      setFilterModel,
      tagFilters: [{ key: 'env', value: 'prod' }],
      setTagFilters,
      searchInput: {} as UseDebouncedSearchInputResult,
    }),
  );
  return { result, setFilterModel, setTagFilters };
};

describe('useTracesV4UnifiedSearch', () => {
  test('shows the URL tag filters as pills after the filter model', () => {
    const { result } = renderUnifiedSearch();
    expect(result.current.searchFilterModel).toEqual([stateClause, urlTagClause]);
  });

  test('keeps the URL tag filters in the URL when another pill is removed', () => {
    const { result, setFilterModel, setTagFilters } = renderUnifiedSearch();

    act(() => result.current.setSearchFilters([urlTagClause]));

    expect(setFilterModel).toHaveBeenCalledWith([]);
    expect(setTagFilters).not.toHaveBeenCalled();
  });

  test('keeps the URL tag filters in the URL when another pill is edited', () => {
    const { result, setFilterModel, setTagFilters } = renderUnifiedSearch();
    const editedClause = { ...stateClause, value: 'OK' };

    act(() => result.current.setSearchFilters([editedClause, urlTagClause]));

    expect(setFilterModel).toHaveBeenCalledWith([editedClause]);
    expect(setTagFilters).not.toHaveBeenCalled();
  });

  test('removes a URL tag filter from the URL when its own pill is removed', () => {
    const { result, setFilterModel, setTagFilters } = renderUnifiedSearch();

    act(() => result.current.setSearchFilters([stateClause]));

    expect(setFilterModel).toHaveBeenCalledWith([stateClause]);
    expect(setTagFilters).toHaveBeenCalledWith([]);
  });

  test('moves an edited URL tag pill into the filter model', () => {
    const { result, setFilterModel, setTagFilters } = renderUnifiedSearch();
    const editedTag = { ...urlTagClause, value: 'staging' };

    act(() => result.current.setSearchFilters([stateClause, editedTag]));

    expect(setFilterModel).toHaveBeenCalledWith([stateClause, editedTag]);
    expect(setTagFilters).toHaveBeenCalledWith([]);
  });
});
