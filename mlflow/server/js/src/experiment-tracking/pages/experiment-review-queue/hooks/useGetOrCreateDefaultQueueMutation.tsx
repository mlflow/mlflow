import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { ReviewQueue } from '../types';
import { REVIEW_QUEUES_API_BASE } from './constants';
import { LIST_REVIEW_QUEUES_QUERY_KEY } from './useListReviewQueuesQuery';

export interface GetOrCreateDefaultQueueParams {
  experiment_id: string;
  created_by?: string;
}

interface GetOrCreateDefaultQueueResponse {
  review_queue: ReviewQueue;
}

/**
 * Get-or-create the experiment's single default queue (a CUSTOM queue that
 * inherits all of the experiment's questions, cannot have its questions edited,
 * and cannot be deleted). Atomic and idempotent, so the Review tab can ensure
 * it exists on load. Invalidates the queue list since a first-time call creates
 * a new queue.
 */
export const useGetOrCreateDefaultQueueMutation = () => {
  const queryClient = useQueryClient();

  const { mutateAsync, isLoading, error, reset } = useMutation<
    GetOrCreateDefaultQueueResponse,
    Error,
    GetOrCreateDefaultQueueParams
  >({
    mutationFn: async (params) => {
      return (await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/get-or-create-default`), {
        method: 'POST',
        body: params,
      })) as GetOrCreateDefaultQueueResponse;
    },
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_REVIEW_QUEUES_QUERY_KEY]);
    },
  });

  return {
    getOrCreateDefaultQueueAsync: mutateAsync,
    isResolvingDefaultQueue: isLoading,
    error,
    reset,
  };
};
