import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { ReviewQueue } from '../types';
import { REVIEW_QUEUES_API_BASE } from './constants';
import { LIST_REVIEW_QUEUES_QUERY_KEY } from './useListReviewQueuesQuery';

export interface GetOrCreateUserQueueParams {
  experiment_id: string;
  user: string;
  created_by?: string;
}

interface GetOrCreateUserQueueResponse {
  review_queue: ReviewQueue;
}

/**
 * Get-or-create a reviewer's personal queue for an experiment. Atomic and
 * idempotent on `(experiment_id, name=user)`, so this is the safe way to
 * resolve "my queue" as an add-to-queue target on the fly (e.g. the
 * pre-selected default option in the flag-for-review picker). Invalidates the
 * queue list since a first-time call creates a new queue.
 */
export const useGetOrCreateUserQueueMutation = () => {
  const queryClient = useQueryClient();

  const { mutateAsync, isLoading, error, reset } = useMutation<
    GetOrCreateUserQueueResponse,
    Error,
    GetOrCreateUserQueueParams
  >({
    mutationFn: async (params) => {
      return (await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/get-or-create-user`), {
        method: 'POST',
        body: params,
      })) as GetOrCreateUserQueueResponse;
    },
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_REVIEW_QUEUES_QUERY_KEY]);
    },
  });

  return {
    getOrCreateUserQueueAsync: mutateAsync,
    isResolvingUserQueue: isLoading,
    error,
    reset,
  };
};
