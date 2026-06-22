import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { renderHook, waitFor, act } from '@testing-library/react';
import React from 'react';

import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { fetchAPI } from '../../../../common/utils/FetchUtils';
import {
  REVIEW_QUEUES_PAGE_SIZE,
  useGetReviewQueueQuery,
  useInfiniteReviewQueuesQuery,
} from './useListReviewQueuesQuery';

jest.mock('../../../../common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(),
  getAjaxUrl: (path: string) => path,
}));

const mockFetchAPI = jest.mocked(fetchAPI);

const makeQueues = (start: number, count: number) =>
  Array.from({ length: count }, (_, i) => ({ queue_id: `rq-${start + i}`, name: `Queue ${start + i}` }));

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, cacheTime: 0 } } });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
};

const paramsOf = (callIndex: number) => {
  const url = mockFetchAPI.mock.calls[callIndex][0] as string;
  return new URLSearchParams(url.split('?')[1] ?? '');
};

describe('useInfiniteReviewQueuesQuery', () => {
  beforeEach(() => {
    mockFetchAPI.mockReset();
    // Page 1 (no token) -> 20 queues + a continuation token; page 2 -> 5 queues, no token (last page).
    mockFetchAPI.mockImplementation((async (url: string) => {
      if (url.includes('review-queues/get?')) {
        const queueId = new URLSearchParams(url.split('?')[1] ?? '').get('queue_id');
        return { review_queue: { queue_id: queueId, name: `Queue ${queueId}` } };
      }
      const token = new URLSearchParams(url.split('?')[1] ?? '').get('page_token');
      if (!token) {
        return { review_queues: makeQueues(0, REVIEW_QUEUES_PAGE_SIZE), next_page_token: 'tok-2' };
      }
      return { review_queues: makeQueues(REVIEW_QUEUES_PAGE_SIZE, 5), next_page_token: '' };
    }) as typeof fetchAPI);
  });

  it('requests the first page with max_results and no page_token, exposing hasNextPage', async () => {
    const { result } = renderHook(() => useInfiniteReviewQueuesQuery({ experimentId: 'exp-1' }), { wrapper });

    await waitFor(() => expect(result.current.reviewQueues).toHaveLength(REVIEW_QUEUES_PAGE_SIZE));

    const firstParams = paramsOf(0);
    expect(firstParams.get('experiment_id')).toBe('exp-1');
    expect(firstParams.get('max_results')).toBe(String(REVIEW_QUEUES_PAGE_SIZE));
    expect(firstParams.has('page_token')).toBe(false);
    expect(result.current.hasNextPage).toBe(true);
  });

  it('sends order_by clauses to the backend for server-side sorting', async () => {
    const { result } = renderHook(
      () => useInfiniteReviewQueuesQuery({ experimentId: 'exp-1', orderBy: ['name ASC'] }),
      { wrapper },
    );
    await waitFor(() => expect(result.current.reviewQueues).toHaveLength(REVIEW_QUEUES_PAGE_SIZE));
    expect(paramsOf(0).getAll('order_by')).toEqual(['name ASC']);
  });

  it('feeds next_page_token back as page_token and flattens accumulated pages', async () => {
    const { result } = renderHook(() => useInfiniteReviewQueuesQuery({ experimentId: 'exp-1' }), { wrapper });

    await waitFor(() => expect(result.current.reviewQueues).toHaveLength(REVIEW_QUEUES_PAGE_SIZE));

    act(() => {
      result.current.fetchNextPage();
    });

    await waitFor(() => expect(result.current.reviewQueues).toHaveLength(REVIEW_QUEUES_PAGE_SIZE + 5));
    // The second request forwards the first page's continuation token.
    expect(paramsOf(1).get('page_token')).toBe('tok-2');
    // The flattened list preserves order across pages.
    expect(result.current.reviewQueues[0].queue_id).toBe('rq-0');
    expect(result.current.reviewQueues[REVIEW_QUEUES_PAGE_SIZE].queue_id).toBe(`rq-${REVIEW_QUEUES_PAGE_SIZE}`);
  });

  it('stops paging when the server returns an empty next_page_token', async () => {
    const { result } = renderHook(() => useInfiniteReviewQueuesQuery({ experimentId: 'exp-1' }), { wrapper });

    await waitFor(() => expect(result.current.reviewQueues).toHaveLength(REVIEW_QUEUES_PAGE_SIZE));
    act(() => {
      result.current.fetchNextPage();
    });
    await waitFor(() => expect(result.current.reviewQueues).toHaveLength(REVIEW_QUEUES_PAGE_SIZE + 5));

    // The last page's empty token maps to undefined, so no further pages remain.
    expect(result.current.hasNextPage).toBe(false);
  });
});

describe('useGetReviewQueueQuery', () => {
  beforeEach(() => {
    mockFetchAPI.mockReset();
    mockFetchAPI.mockImplementation((async (url: string) => {
      const queueId = new URLSearchParams(url.split('?')[1] ?? '').get('queue_id');
      return { review_queue: { queue_id: queueId, name: `Queue ${queueId}` } };
    }) as typeof fetchAPI);
  });

  it('fetches a single queue by id from the get endpoint', async () => {
    const { result } = renderHook(() => useGetReviewQueueQuery({ queueId: 'rq-42' }), { wrapper });

    await waitFor(() => expect(result.current.reviewQueue?.queue_id).toBe('rq-42'));
    const url = mockFetchAPI.mock.calls[0][0] as string;
    expect(url).toContain('review-queues/get');
    expect(new URLSearchParams(url.split('?')[1]).get('queue_id')).toBe('rq-42');
  });

  it('does not fetch when disabled or no queue id', () => {
    renderHook(() => useGetReviewQueueQuery({ queueId: 'rq-1', enabled: false }), { wrapper });
    renderHook(() => useGetReviewQueueQuery({ queueId: undefined }), { wrapper });
    expect(mockFetchAPI).not.toHaveBeenCalled();
  });
});
