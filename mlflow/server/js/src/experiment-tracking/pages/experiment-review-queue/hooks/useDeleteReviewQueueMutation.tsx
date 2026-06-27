import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import { REVIEW_QUEUES_API_BASE } from './constants';
import { LIST_REVIEW_QUEUES_QUERY_KEY } from './useListReviewQueuesQuery';

export interface DeleteReviewQueueParams {
  queue_id: string;
}

/**
 * Delete a review queue and its user / trace / schema associations. No-op
 * when the queue doesn't exist (idempotent). Invalidates the queue list.
 */
export const useDeleteReviewQueueMutation = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error, reset } = useMutation<unknown, Error, DeleteReviewQueueParams>({
    mutationFn: async (params) => {
      return await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/delete`), {
        method: 'POST',
        body: params,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_REVIEW_QUEUES_QUERY_KEY]);
    },
  });

  return {
    deleteReviewQueue: mutate,
    deleteReviewQueueAsync: mutateAsync,
    isDeletingQueue: isLoading,
    error,
    reset,
  };
};
