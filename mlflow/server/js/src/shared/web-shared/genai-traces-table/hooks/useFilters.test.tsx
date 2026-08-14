import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';

import { useFilters } from './useFilters';
import { useSearchParams } from '../utils/RoutingUtils';
import { FilterOperator } from '../types';

jest.mock('../utils/RoutingUtils', () => ({
  useSearchParams: jest.fn(),
}));

let mockLocalStorageValue: any = undefined;
const mockSetLocalStorage = jest.fn((val) => {
  mockLocalStorageValue = val;
});

jest.mock('../../hooks/useLocalStorage', () => ({
  useLocalStorage: () => [mockLocalStorageValue, mockSetLocalStorage],
}));

describe('useFilters - persistence', () => {
  let mockSearchParams: URLSearchParams;
  const mockSetSearchParams = jest.fn((setter: any) => {
    mockSearchParams = setter(mockSearchParams);
  });

  beforeEach(() => {
    mockSearchParams = new URLSearchParams();
    mockLocalStorageValue = undefined;
    jest.clearAllMocks();
    jest.mocked(useSearchParams).mockReturnValue([mockSearchParams, mockSetSearchParams]);
  });

  describe('persist=true', () => {
    it('saves filters to localStorage when setFilters is called', () => {
      const { result } = renderHook(() => useFilters({ persist: true, persistKey: 'test-key' }));

      const testFilter = { column: 'status', operator: FilterOperator.EQUALS, value: 'OK' };

      act(() => {
        result.current[1]([testFilter]);
      });

      expect(mockSetLocalStorage).toHaveBeenCalledWith([testFilter]);
    });
  });

  describe('persist=false (default)', () => {
    it('does NOT save to localStorage when setFilters is called', () => {
      const { result } = renderHook(() => useFilters());

      act(() => {
        result.current[1]([{ column: 'status', operator: FilterOperator.EQUALS, value: 'OK' }]);
      });

      expect(mockSetLocalStorage).not.toHaveBeenCalled();
    });
  });

  describe('loadPersistedValues=true', () => {
    it('rehydrates from localStorage when URL params are empty', () => {
      const persistedFilters = [{ column: 'status', operator: 'eq' as const, value: 'ERROR' }];
      mockLocalStorageValue = persistedFilters;

      renderHook(() => useFilters({ loadPersistedValues: true, persistKey: 'test-key' }));

      // setFilters should be called with persisted value (replace=true)
      expect(mockSetSearchParams).toHaveBeenCalled();
    });

    it('does NOT rehydrate when URL params already exist', () => {
      mockSearchParams = new URLSearchParams('filter=status::eq::OK');
      jest.mocked(useSearchParams).mockReturnValue([mockSearchParams, mockSetSearchParams]);

      const persistedFilters = [{ column: 'status', operator: 'eq' as const, value: 'ERROR' }];
      mockLocalStorageValue = persistedFilters;

      renderHook(() => useFilters({ loadPersistedValues: true, persistKey: 'test-key' }));

      // Should NOT call setSearchParams to replace URL with localStorage value
      expect(mockSetSearchParams).not.toHaveBeenCalled();
    });
  });
});
