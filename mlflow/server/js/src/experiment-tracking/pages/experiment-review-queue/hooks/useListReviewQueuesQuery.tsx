import { useQuery } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { ReviewQueue } from '../types';
import { REVIEW_QUEUES_API_BASE } from './constants';

export const LIST_REVIEW_QUEUES_QUERY_KEY = 'LIST_REVIEW_QUEUES';

interface ListReviewQueuesResponse {
  review_queues?: ReviewQueue[];
  next_page_token?: string;
}

/**
 * Paginated list of an experiment's review queues, newest first
 * (server-side; see `SqlAlchemyStore.list_review_queues`). When `user` is
 * set, only queues that user is assigned to are returned.
 */
export const useListReviewQueuesQuery = ({
  experimentId,
  user,
  maxResults,
  pageToken,
  ensureDefault = false,
  enabled = true,
}: {
  experimentId: string;
  user?: string;
  maxResults?: number;
  pageToken?: string;
  /** No-auth only: ask the server to seed the experiment's default queue. */
  ensureDefault?: boolean;
  enabled?: boolean;
}) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<ListReviewQueuesResponse, Error>({
    queryKey: [LIST_REVIEW_QUEUES_QUERY_KEY, experimentId, user, maxResults, pageToken, ensureDefault],
    queryFn: async () => {
      const params = new URLSearchParams({ experiment_id: experimentId });
      if (user) {
        params.set('user', user);
      }
      // Guard out 0/negative client-side; the handler enforces
      // max_results in [1, SEARCH_MAX_RESULTS_THRESHOLD].
      if (maxResults != null && maxResults > 0) {
        params.set('max_results', String(maxResults));
      }
      if (pageToken) {
        params.set('page_token', pageToken);
      }
      // No-auth only: ask the server to seed the experiment's protected default
      // queue (idempotent) before listing. The caller sets this when auth is
      // unavailable; auth servers leave it off, so no default queue is created.
      if (ensureDefault) {
        params.set('ensure_default', 'true');
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
