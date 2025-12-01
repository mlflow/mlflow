import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';
import { useSelectedDatasetBySearchParam } from './useSelectedDatasetBySearchParam';
import { useSearchParams } from '../../../../common/utils/RoutingUtils';

jest.mock('../../../../common/utils/RoutingUtils', () => ({
  useSearchParams: jest.fn(),
}));

describe('useSelectedDatasetBySearchParam', () => {
  let mockSearchParams = new URLSearchParams();
  const mockSetSearchParams = jest.fn((setter) => {
    // @ts-expect-error 'setter' is of type 'unknown'
    mockSearchParams = setter(mockSearchParams);
  });

  beforeEach(() => {
    mockSearchParams = new URLSearchParams();
    jest.clearAllMocks();
    jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams(), mockSetSearchParams]);
  });

  test('should return selectedDatasetId from URL', () => {
    jest
      .mocked(useSearchParams)
      .mockReturnValue([new URLSearchParams({ selectedDataset: 'dataset-123' }), mockSetSearchParams]);

    const {
      result: {
        current: [resultDatasetId],
      },
    } = renderHook(() => useSelectedDatasetBySearchParam());

    expect(resultDatasetId).toEqual('dataset-123');
  });

  test('should set selectedDatasetId in URL', () => {
    const {
      result: {
        current: [, setSelectedDatasetId],
      },
    } = renderHook(() => useSelectedDatasetBySearchParam());

    act(() => {
      setSelectedDatasetId('dataset-456');
    });

    expect(mockSetSearchParams).toHaveBeenCalledTimes(1);
    expect(mockSearchParams.get('selectedDataset')).toEqual('dataset-456');
  });

  test('should clear selectedDatasetId when set to undefined', () => {
    mockSearchParams = new URLSearchParams({ selectedDataset: 'dataset-123' });

    const {
      result: {
        current: [, setSelectedDatasetId],
      },
    } = renderHook(() => useSelectedDatasetBySearchParam());

    act(() => {
      setSelectedDatasetId(undefined);
    });

    expect(mockSetSearchParams).toHaveBeenCalledTimes(1);
    expect(mockSearchParams.get('selectedDataset')).toBeNull();
  });
});
