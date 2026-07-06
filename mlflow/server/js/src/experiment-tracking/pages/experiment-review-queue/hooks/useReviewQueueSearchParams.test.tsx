import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';

import { getReviewQueuePageRoute, useReviewQueueSearchParams } from './useReviewQueueSearchParams';
import { useSearchParams } from '../../../../common/utils/RoutingUtils';

jest.mock('../../../../common/utils/RoutingUtils', () => ({
  ...(jest.requireActual<typeof import('../../../../common/utils/RoutingUtils')>(
    '../../../../common/utils/RoutingUtils',
  ) as any),
  useSearchParams: jest.fn(),
}));

describe('getReviewQueuePageRoute', () => {
  test('builds the queue deep link', () => {
    expect(getReviewQueuePageRoute('exp-1')).toEqual('/experiments/exp-1/review-queue');
    expect(getReviewQueuePageRoute('exp-1', 'rq-1')).toEqual('/experiments/exp-1/review-queue?selectedQueueId=rq-1');
    expect(getReviewQueuePageRoute('exp-1', 'rq-1', { startReview: true })).toEqual(
      '/experiments/exp-1/review-queue?selectedQueueId=rq-1&startReview=true',
    );
    expect(getReviewQueuePageRoute('exp-1', undefined, { startReview: true })).toEqual(
      '/experiments/exp-1/review-queue?startReview=true',
    );
  });
});

describe('useReviewQueueSearchParams', () => {
  let mockSearchParams = new URLSearchParams();
  const mockSetSearchParams = jest.fn((setter) => {
    // @ts-expect-error 'setter' is of type 'unknown'
    mockSearchParams = setter(mockSearchParams);
  });

  const mockParams = (init?: Record<string, string>) => {
    mockSearchParams = new URLSearchParams(init);
    jest.mocked(useSearchParams).mockReturnValue([mockSearchParams, mockSetSearchParams]);
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockParams();
  });

  test('reads selection and start-review intent from the URL', () => {
    mockParams({ selectedQueueId: 'rq-1', selectedItemId: 'tr-1', startReview: 'true' });

    const { result } = renderHook(() => useReviewQueueSearchParams());

    expect(result.current.selectedQueueId).toEqual('rq-1');
    expect(result.current.openItemId).toEqual('tr-1');
    expect(result.current.startReviewRequested).toBe(true);
  });

  test('selecting a queue closes the open item and voids the start-review intent', () => {
    mockParams({ selectedQueueId: 'rq-1', selectedItemId: 'tr-1', startReview: 'true' });

    const { result } = renderHook(() => useReviewQueueSearchParams());
    act(() => result.current.selectQueue('rq-2'));

    expect(mockSearchParams.get('selectedQueueId')).toEqual('rq-2');
    expect(mockSearchParams.get('selectedItemId')).toBeNull();
    expect(mockSearchParams.get('startReview')).toBeNull();
  });

  test('auto-select preserves the start-review intent', () => {
    mockParams({ startReview: 'true' });

    const { result } = renderHook(() => useReviewQueueSearchParams());
    act(() => result.current.selectQueue('rq-1', { preserveStartReview: true }));

    expect(mockSearchParams.get('selectedQueueId')).toEqual('rq-1');
    expect(mockSearchParams.get('startReview')).toEqual('true');
  });

  test('clearing the queue selection drops all selection params', () => {
    mockParams({ selectedQueueId: 'rq-1', selectedItemId: 'tr-1' });

    const { result } = renderHook(() => useReviewQueueSearchParams());
    act(() => result.current.selectQueue(undefined));

    expect([...mockSearchParams.keys()]).toEqual([]);
  });

  test('opening and closing an item only touches the item param', () => {
    mockParams({ selectedQueueId: 'rq-1' });

    const { result } = renderHook(() => useReviewQueueSearchParams());
    act(() => result.current.setOpenItemId('tr-1'));
    expect(mockSearchParams.get('selectedItemId')).toEqual('tr-1');
    expect(mockSearchParams.get('selectedQueueId')).toEqual('rq-1');

    act(() => result.current.setOpenItemId(null));
    expect(mockSearchParams.get('selectedItemId')).toBeNull();
    expect(mockSearchParams.get('selectedQueueId')).toEqual('rq-1');
  });

  test('consuming the start-review intent swaps it for the first to-do item', () => {
    mockParams({ selectedQueueId: 'rq-1', startReview: 'true' });

    const { result } = renderHook(() => useReviewQueueSearchParams());
    act(() => result.current.consumeStartReview('tr-todo'));

    expect(mockSearchParams.get('startReview')).toBeNull();
    expect(mockSearchParams.get('selectedItemId')).toEqual('tr-todo');
  });

  test('consuming the start-review intent with no to-do items stays on the list', () => {
    mockParams({ selectedQueueId: 'rq-1', startReview: 'true' });

    const { result } = renderHook(() => useReviewQueueSearchParams());
    act(() => result.current.consumeStartReview(null));

    expect(mockSearchParams.get('startReview')).toBeNull();
    expect(mockSearchParams.get('selectedItemId')).toBeNull();
  });
});
