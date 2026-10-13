import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { ReviewQueue, ReviewQueueType } from '../types';
import { REVIEW_QUEUES_API_BASE } from './constants';
import { LIST_REVIEW_QUEUES_QUERY_KEY } from './useListReviewQueuesQuery';

export interface CreateReviewQueueParams {
  experiment_id: string;
  name: string;
  queue_type: ReviewQueueType;
  created_by?: string;
  users?: string[];
  schema_ids?: string[];
}

interface CreateReviewQueueResponse {
  review_queue: ReviewQueue;
}

/**
 * Create a review queue. Collisions on `(experiment_id, name)` return
 * RESOURCE_ALREADY_EXISTS. Invalidates the experiment's queue list.
 */
export const useCreateReviewQueueMutation = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation<
    CreateReviewQueueResponse,
    Error,
    CreateReviewQueueParams
  >({
    mutationFn: async (params) => {
      return (await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/create`), {
        method: 'POST',
        body: params,
      })) as CreateReviewQueueResponse;
    },
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_REVIEW_QUEUES_QUERY_KEY]);
    },
  });

  return {
    createReviewQueue: mutate,
    createReviewQueueAsync: mutateAsync,
    isCreatingQueue: isLoading,
    error,
  };
};
