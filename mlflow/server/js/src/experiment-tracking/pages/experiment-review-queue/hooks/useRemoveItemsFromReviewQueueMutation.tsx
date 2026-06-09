import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import { REVIEW_QUEUES_API_BASE } from './constants';
import { LIST_REVIEW_QUEUE_ITEMS_QUERY_KEY } from './useListReviewQueueItemsQuery';

export interface RemoveItemsFromReviewQueueParams {
  queue_id: string;
  item_ids: string[];
}

/**
 * Detach traces from a review queue. No-op for traces that aren't attached;
 * the traces and their assessments are untouched (they can be re-flagged).
 * Invalidates the queue's trace list so the Review tab drops the rows.
 */
export const useRemoveItemsFromReviewQueueMutation = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation<unknown, Error, RemoveItemsFromReviewQueueParams>({
    mutationFn: async (params) => {
      return await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/items/remove`), {
        method: 'POST',
        body: params,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_REVIEW_QUEUE_ITEMS_QUERY_KEY]);
    },
  });

  return {
    removeItemsFromReviewQueue: mutate,
    removeItemsFromReviewQueueAsync: mutateAsync,
    isRemovingItems: isLoading,
    error,
  };
};
