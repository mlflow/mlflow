import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { ReviewQueueItem } from '../types';
import { REVIEW_QUEUES_API_BASE } from './constants';
import { LIST_REVIEW_QUEUE_TRACES_QUERY_KEY } from './useListReviewQueueTracesQuery';

export interface AddTracesToReviewQueueParams {
  queue_id: string;
  target_ids: string[];
}

interface AddTracesToReviewQueueResponse {
  items?: ReviewQueueItem[];
}

/**
 * Attach traces to a review queue. Idempotent per trace (re-attaching keeps
 * the existing status). The server defaults `target_type` to TRACE, so it is
 * omitted here. Invalidates the queue's trace list so the Review tab reflects
 * the new attachments.
 */
export const useAddTracesToReviewQueueMutation = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error, reset } = useMutation<
    AddTracesToReviewQueueResponse,
    Error,
    AddTracesToReviewQueueParams
  >({
    mutationFn: async (params) => {
      return (await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/traces/add`), {
        method: 'POST',
        body: params,
      })) as AddTracesToReviewQueueResponse;
    },
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_REVIEW_QUEUE_TRACES_QUERY_KEY]);
    },
  });

  return {
    addTracesToReviewQueue: mutate,
    addTracesToReviewQueueAsync: mutateAsync,
    isAddingTraces: isLoading,
    error,
    reset,
  };
};
