import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { ReviewQueueItem } from '../types';
import { REVIEW_QUEUES_API_BASE } from './constants';
import { LIST_REVIEW_QUEUE_ITEMS_QUERY_KEY } from './useListReviewQueueItemsQuery';
import { LIST_REVIEW_QUEUES_QUERY_KEY } from './useListReviewQueuesQuery';

export interface AddItemsToReviewQueueParams {
  queue_id: string;
  item_ids: string[];
}

interface AddItemsToReviewQueueResponse {
  items?: ReviewQueueItem[];
}

/**
 * Attach traces to a review queue. Idempotent per trace (re-attaching keeps
 * the existing status). The server defaults `item_type` to TRACE, so it is
 * omitted here. Invalidates the queue's trace list so the Review tab reflects
 * the new attachments, and the queue list so the per-trace membership view
 * (`itemId`) picks up this new membership.
 */
export const useAddItemsToReviewQueueMutation = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error, reset } = useMutation<
    AddItemsToReviewQueueResponse,
    Error,
    AddItemsToReviewQueueParams
  >({
    mutationFn: async (params) => {
      return (await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/items/add`), {
        method: 'POST',
        body: params,
      })) as AddItemsToReviewQueueResponse;
    },
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_REVIEW_QUEUE_ITEMS_QUERY_KEY]);
      queryClient.invalidateQueries([LIST_REVIEW_QUEUES_QUERY_KEY]);
    },
  });

  return {
    addItemsToReviewQueue: mutate,
    addItemsToReviewQueueAsync: mutateAsync,
    isAddingItems: isLoading,
    error,
    reset,
  };
};
