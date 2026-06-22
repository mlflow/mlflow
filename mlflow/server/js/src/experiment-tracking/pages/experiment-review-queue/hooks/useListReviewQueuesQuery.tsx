import { useMemo } from 'react';

import { useInfiniteQuery, useQuery } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { ReviewQueue } from '../types';
import { REVIEW_QUEUES_API_BASE } from './constants';

export const LIST_REVIEW_QUEUES_QUERY_KEY = 'LIST_REVIEW_QUEUES';
export const GET_REVIEW_QUEUE_QUERY_KEY = 'GET_REVIEW_QUEUE';

// Small page so the queues sidebar streams in as the reviewer scrolls instead
// of loading every queue (and its per-queue pending-count fetch) up front.
export const REVIEW_QUEUES_PAGE_SIZE = 20;

interface ListReviewQueuesResponse {
  review_queues?: ReviewQueue[];
  next_page_token?: string;
}

interface GetReviewQueueResponse {
  review_queue?: ReviewQueue;
}

/**
 * Paginated list of an experiment's review queues, newest first
 * (server-side; see `SqlAlchemyStore.list_review_queues`). When `user` is
 * set, only queues that user is assigned to are returned. When `itemId` is
 * set, only queues that already contain that item (a trace id) are returned —
 * use it to see which queues a trace is already a member of.
 */
export const useListReviewQueuesQuery = ({
  experimentId,
  user,
  itemId,
  maxResults,
  pageToken,
  enabled = true,
}: {
  experimentId: string;
  user?: string;
  itemId?: string;
  maxResults?: number;
  pageToken?: string;
  enabled?: boolean;
}) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<ListReviewQueuesResponse, Error>({
    queryKey: [LIST_REVIEW_QUEUES_QUERY_KEY, experimentId, user, itemId, maxResults, pageToken],
    queryFn: async () => {
      const params = new URLSearchParams({ experiment_id: experimentId });
      if (user) {
        params.set('user', user);
      }
      if (itemId) {
        params.set('item_id', itemId);
      }
      // Guard out 0/negative client-side; the handler enforces
      // max_results in [1, SEARCH_MAX_RESULTS_THRESHOLD].
      if (maxResults != null && maxResults > 0) {
        params.set('max_results', String(maxResults));
      }
      if (pageToken) {
        params.set('page_token', pageToken);
      }
      return (await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/list?${params.toString()}`), {
        method: 'GET',
      })) as ListReviewQueuesResponse;
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && Boolean(experimentId),
  });

  return {
    reviewQueues: data?.review_queues ?? [],
    nextPageToken: data?.next_page_token,
    isLoading,
    isFetching,
    refetch,
    error,
  };
};

/**
 * Infinite (page-token) variant of {@link useListReviewQueuesQuery} that powers
 * the sidebar's infinite scroll. Pages accumulate as the caller invokes
 * `fetchNextPage`, and the flattened `reviewQueues` exposes only the pages
 * loaded so far, so the sidebar's filter/sort and per-queue counts (and the
 * page's queue-selection lookup) operate over the loaded window. Mutations
 * invalidate by the shared `LIST_REVIEW_QUEUES_QUERY_KEY` prefix, which matches
 * this key too.
 */
export const useInfiniteReviewQueuesQuery = ({
  experimentId,
  user,
  itemId,
  orderBy,
  enabled = true,
}: {
  experimentId: string;
  user?: string;
  itemId?: string;
  /** Backend `order_by` clauses (e.g. `["name ASC"]`). Part of the query key, so
   *  changing the sort refetches from page 1 in the new (whole-list) order. */
  orderBy?: string[];
  enabled?: boolean;
}) => {
  const { data, fetchNextPage, hasNextPage, isLoading, isFetching, isFetchingNextPage, refetch, error } =
    useInfiniteQuery<ListReviewQueuesResponse, Error>({
      queryKey: [LIST_REVIEW_QUEUES_QUERY_KEY, 'infinite', experimentId, user, itemId, orderBy],
      queryFn: async ({ pageParam = undefined }) => {
        const params = new URLSearchParams({ experiment_id: experimentId });
        if (user) {
          params.set('user', user);
        }
        if (itemId) {
          params.set('item_id', itemId);
        }
        params.set('max_results', String(REVIEW_QUEUES_PAGE_SIZE));
        if (pageParam) {
          params.set('page_token', pageParam as string);
        }
        // Repeated query param: order_by=<clause>&order_by=<clause>.
        (orderBy ?? []).forEach((clause) => params.append('order_by', clause));
        return (await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/list?${params.toString()}`), {
          method: 'GET',
        })) as ListReviewQueuesResponse;
      },
      // The handler returns an empty `next_page_token` (not an absent field) on
      // the last page; map the empty string to `undefined` so react-query stops.
      getNextPageParam: (lastPage) => lastPage.next_page_token || undefined,
      cacheTime: 0,
      refetchOnWindowFocus: false,
      retry: false,
      enabled: enabled && Boolean(experimentId),
    });

  const reviewQueues = useMemo(() => data?.pages.flatMap((page) => page.review_queues ?? []) ?? [], [data]);

  return {
    reviewQueues,
    fetchNextPage,
    hasNextPage,
    isLoading,
    isFetching,
    isFetchingNextPage,
    refetch,
    error,
  };
};

/**
 * Fetch a single review queue by id. Used to resolve a selected/deep-linked
 * queue that isn't on a loaded page of the infinite list, so the right pane can
 * open it with one request instead of paging through the whole list to find it.
 */
export const useGetReviewQueueQuery = ({ queueId, enabled = true }: { queueId?: string; enabled?: boolean }) => {
  const { data, isLoading, error } = useQuery<GetReviewQueueResponse, Error>({
    queryKey: [GET_REVIEW_QUEUE_QUERY_KEY, queueId],
    queryFn: async () => {
      const params = new URLSearchParams({ queue_id: queueId ?? '' });
      return (await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/get?${params.toString()}`), {
        method: 'GET',
      })) as GetReviewQueueResponse;
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && Boolean(queueId),
  });

  return { reviewQueue: data?.review_queue, isLoading, error };
};
