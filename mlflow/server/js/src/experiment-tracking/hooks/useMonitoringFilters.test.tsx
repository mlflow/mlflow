import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';

import { useMonitoringFilters, type MonitoringFilters } from './useMonitoringFilters';
import { useParams, useSearchParams } from '../../common/utils/RoutingUtils';

jest.mock('../../common/utils/RoutingUtils', () => ({
  useParams: jest.fn(),
  useSearchParams: jest.fn(),
}));

jest.mock('./useMonitoringConfig', () => ({
  useMonitoringConfig: () => ({ dateNow: new Date('2025-01-01T00:00:00Z') }),
}));

let mockLocalStorageValue: MonitoringFilters | undefined = undefined;
const mockPersistLocalStorage = jest.fn((val: MonitoringFilters | undefined) => {
  mockLocalStorageValue = val;
});

jest.mock('@databricks/web-shared/hooks', () => ({
  useLocalStorage: () => [mockLocalStorageValue, mockPersistLocalStorage],
}));

describe('useMonitoringFilters - persistence', () => {
  let mockSearchParams: URLSearchParams;
  const mockSetSearchParams = jest.fn((setter: any) => {
    mockSearchParams = setter(mockSearchParams);
  });

  beforeEach(() => {
    mockSearchParams = new URLSearchParams();
    mockLocalStorageValue = undefined;
    jest.clearAllMocks();
    jest.mocked(useParams).mockReturnValue({ experimentId: 'exp-123' });
    jest.mocked(useSearchParams).mockReturnValue([mockSearchParams, mockSetSearchParams]);
  });

  describe('persist=true', () => {
    it('saves filters to localStorage when setMonitoringFilters is called', () => {
      const { result } = renderHook(() => useMonitoringFilters({ persist: true }));

      const testFilters: MonitoringFilters = { startTimeLabel: 'LAST_24_HOURS' };

      act(() => {
        result.current[1](testFilters);
      });

      expect(mockPersistLocalStorage).toHaveBeenCalledWith(testFilters);
    });
  });

  describe('persist=false (default)', () => {
    it('does NOT save to localStorage when setMonitoringFilters is called', () => {
      const { result } = renderHook(() => useMonitoringFilters());

      act(() => {
        result.current[1]({ startTimeLabel: 'LAST_24_HOURS' });
      });

      expect(mockPersistLocalStorage).not.toHaveBeenCalled();
    });
  });

  describe('loadPersistedValues=true', () => {
    it('rehydrates from localStorage when URL params are empty', () => {
      const persistedFilters: MonitoringFilters = { startTimeLabel: 'LAST_30_DAYS' };
      mockLocalStorageValue = persistedFilters;

      renderHook(() => useMonitoringFilters({ loadPersistedValues: true }));

      expect(mockSetSearchParams).toHaveBeenCalled();
    });

    it('does NOT rehydrate when URL params already exist', () => {
      mockSearchParams = new URLSearchParams('startTimeLabel=LAST_HOUR');
      jest.mocked(useSearchParams).mockReturnValue([mockSearchParams, mockSetSearchParams]);

      const persistedFilters: MonitoringFilters = { startTimeLabel: 'LAST_30_DAYS' };
      mockLocalStorageValue = persistedFilters;

      renderHook(() => useMonitoringFilters({ loadPersistedValues: true }));

      expect(mockSetSearchParams).not.toHaveBeenCalled();
    });
  });
});
