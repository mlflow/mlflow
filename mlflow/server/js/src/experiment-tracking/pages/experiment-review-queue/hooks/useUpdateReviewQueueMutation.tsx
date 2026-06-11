import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { ReviewQueue } from '../types';
import { REVIEW_QUEUES_API_BASE } from './constants';
import { LIST_REVIEW_QUEUES_QUERY_KEY } from './useListReviewQueuesQuery';

/**
 * Replace a CUSTOM queue's assigned users and/or attached schemas (questions).
 * Pass only the set you want to change; omitting one leaves it untouched (an
 * empty array clears it). The wire protocol gates each set behind an
 * `update_*` flag because a repeated field can't distinguish "absent" from
 * "empty" — the hook derives those flags from which fields are present, so
 * callers just pass `users` / `schema_ids`. The server freezes `schema_ids`
 * once the queue has traces and rejects USER-queue updates. Invalidates the
 * queue list.
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
    mutationFn: async ({ queue_id, users, schema_ids }) => {
      const body = {
        queue_id,
        ...(users !== undefined ? { update_users: true, users } : {}),
        ...(schema_ids !== undefined ? { update_schema_ids: true, schema_ids } : {}),
      };
      return (await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/update`), {
        method: 'POST',
        body,
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
