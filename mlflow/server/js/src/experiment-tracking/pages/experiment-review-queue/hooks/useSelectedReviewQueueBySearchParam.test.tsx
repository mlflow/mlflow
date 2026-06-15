import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';

import { useSelectedReviewQueueBySearchParam } from './useSelectedReviewQueueBySearchParam';
import { useSearchParams } from '../../../../common/utils/RoutingUtils';

jest.mock('../../../../common/utils/RoutingUtils', () => ({
  useSearchParams: jest.fn(),
}));

describe('useSelectedReviewQueueBySearchParam', () => {
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

  test('returns the selected queue id from the URL', () => {
    jest
      .mocked(useSearchParams)
      .mockReturnValue([new URLSearchParams({ selectedQueueId: 'rq-123' }), mockSetSearchParams]);

    const {
      result: {
        current: [selectedQueueId],
      },
    } = renderHook(() => useSelectedReviewQueueBySearchParam());

    expect(selectedQueueId).toEqual('rq-123');
  });

  test('sets the selected queue id in the URL', () => {
    const {
      result: {
        current: [, setSelectedQueueId],
      },
    } = renderHook(() => useSelectedReviewQueueBySearchParam());

    act(() => {
      setSelectedQueueId('rq-456');
    });

    expect(mockSetSearchParams).toHaveBeenCalledTimes(1);
    expect(mockSearchParams.get('selectedQueueId')).toEqual('rq-456');
  });

  test('clears the selected queue id when set to undefined', () => {
    mockSearchParams = new URLSearchParams({ selectedQueueId: 'rq-123' });

    const {
      result: {
        current: [, setSelectedQueueId],
      },
    } = renderHook(() => useSelectedReviewQueueBySearchParam());

    act(() => {
      setSelectedQueueId(undefined);
    });

    expect(mockSetSearchParams).toHaveBeenCalledTimes(1);
    expect(mockSearchParams.get('selectedQueueId')).toBeNull();
  });
});
