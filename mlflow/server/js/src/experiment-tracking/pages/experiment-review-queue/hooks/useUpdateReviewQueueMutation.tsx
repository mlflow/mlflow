import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { ReviewQueue } from '../types';
import { REVIEW_QUEUES_API_BASE } from './constants';
import { LIST_REVIEW_QUEUES_QUERY_KEY } from './useListReviewQueuesQuery';

/**
 * Replace a CUSTOM queue's assigned users and/or attached schemas (questions).
 * Each set is sent only when provided. The server freezes `schema_ids` once
 * the queue has traces and rejects USER-queue updates, so callers should gate
 * the question editor on an empty queue. Invalidates the queue list.
 */
export interface UpdateReviewQueueParams {
  queue_id: string;
  users?: string[];
  schema_ids?: string[];
}

interface UpdateReviewQueueResponse {
  review_queue: ReviewQueue;
}

export const useUpdateReviewQueueMutation = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error, reset } = useMutation<
    UpdateReviewQueueResponse,
    Error,
    UpdateReviewQueueParams
  >({
    mutationFn: async (params) => {
      return (await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/update`), {
        method: 'POST',
        body: params,
      })) as UpdateReviewQueueResponse;
    },
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_REVIEW_QUEUES_QUERY_KEY]);
    },
  });

  return {
    updateReviewQueue: mutate,
    updateReviewQueueAsync: mutateAsync,
    isUpdatingQueue: isLoading,
    error,
    reset,
  };
};
