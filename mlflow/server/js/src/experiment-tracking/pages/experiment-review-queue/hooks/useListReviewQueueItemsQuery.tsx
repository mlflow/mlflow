import { useQuery } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { ReviewQueueItem, ReviewStatus } from '../types';
import { REVIEW_QUEUES_API_BASE } from './constants';

export const LIST_REVIEW_QUEUE_ITEMS_QUERY_KEY = 'LIST_REVIEW_QUEUE_ITEMS';

interface ListReviewQueueItemsResponse {
  items?: ReviewQueueItem[];
  next_page_token?: string;
}

/**
 * Paginated list of a queue's attached items, newest-attached first,
 * optionally filtered by shared-pool status.
 */
export const useListReviewQueueItemsQuery = ({
  queueId,
  status,
  maxResults,
  pageToken,
  enabled = true,
}: {
  queueId: string;
  status?: ReviewStatus;
  maxResults?: number;
  pageToken?: string;
  enabled?: boolean;
}) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<ListReviewQueueItemsResponse, Error>({
    queryKey: [LIST_REVIEW_QUEUE_ITEMS_QUERY_KEY, queueId, status, maxResults, pageToken],
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
      return (await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/items/list?${params.toString()}`), {
        method: 'GET',
      })) as ListReviewQueueItemsResponse;
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
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
