import { useQuery } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { ReviewQueueItem, ReviewStatus } from '../types';
import { REVIEW_QUEUES_API_BASE } from './constants';

export const LIST_REVIEW_QUEUE_TRACES_QUERY_KEY = 'LIST_REVIEW_QUEUE_TRACES';

interface ListReviewQueueTracesResponse {
  items?: ReviewQueueItem[];
  next_page_token?: string;
}

interface ReviewQueueTracesQueryArgs {
  queueId: string;
  status?: ReviewStatus;
  maxResults?: number;
  pageToken?: string;
}

/**
 * Shared react-query config for a queue's attached traces, so a single fetch is
 * reused whether subscribed via `useQuery` (one queue) or `useQueries` (many
 * queues at once, e.g. the sidebar's per-queue pending counts).
 */
export const buildReviewQueueTracesQuery = ({
  queueId,
  status,
  maxResults,
  pageToken,
}: ReviewQueueTracesQueryArgs) => ({
  queryKey: [LIST_REVIEW_QUEUE_TRACES_QUERY_KEY, queueId, status, maxResults, pageToken],
  queryFn: async () => {
    const params = new URLSearchParams({ queue_id: queueId });
    if (status) {
      params.set('status', status);
    }
    if (maxResults != null && maxResults > 0) {
      params.set('max_results', String(maxResults));
    }
    if (pageToken) {
      params.set('page_token', pageToken);
    }
    return (await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/traces/list?${params.toString()}`), {
      method: 'GET',
    })) as ListReviewQueueTracesResponse;
  },
  cacheTime: 0,
  refetchOnWindowFocus: false,
  retry: false,
  enabled: Boolean(queueId),
});

/**
 * Paginated list of a queue's attached traces, newest-attached first,
 * optionally filtered by shared-pool status.
 */
export const useListReviewQueueTracesQuery = ({
  queueId,
  status,
  maxResults,
  pageToken,
  enabled = true,
}: ReviewQueueTracesQueryArgs & { enabled?: boolean }) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<ListReviewQueueTracesResponse, Error>({
    ...buildReviewQueueTracesQuery({ queueId, status, maxResults, pageToken }),
    enabled: enabled && Boolean(queueId),
  });

  return {
    items: data?.items ?? [],
    nextPageToken: data?.next_page_token,
    isLoading,
    isFetching,
    refetch,
    error,
  };
};
