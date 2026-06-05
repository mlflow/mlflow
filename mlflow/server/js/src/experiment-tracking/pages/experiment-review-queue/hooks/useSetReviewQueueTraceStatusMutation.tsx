import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { ReviewQueueItem, ReviewStatus } from '../types';
import { REVIEW_QUEUES_API_BASE } from './constants';
import { LIST_REVIEW_QUEUE_TRACES_QUERY_KEY } from './useListReviewQueueTracesQuery';

export interface SetReviewQueueTraceStatusParams {
  queue_id: string;
  target_id: string;
  status: ReviewStatus;
  /** Required for COMPLETE / DECLINED; must be omitted for PENDING (reopen). */
  completed_by?: string;
}

interface SetReviewQueueTraceStatusResponse {
  item: ReviewQueueItem;
}

/**
 * Set the shared-pool status of an attached trace (the reviewer's
 * complete / decline / reopen action). Invalidates the queue's trace list
 * so the table reflects the new status.
 */
export const useSetReviewQueueTraceStatusMutation = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation<
    SetReviewQueueTraceStatusResponse,
    Error,
    SetReviewQueueTraceStatusParams
  >({
    mutationFn: async (params) => {
      return (await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/traces/set-status`), {
        method: 'POST',
        body: params,
      })) as SetReviewQueueTraceStatusResponse;
    },
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_REVIEW_QUEUE_TRACES_QUERY_KEY]);
    },
  });

  return {
    setReviewQueueTraceStatus: mutate,
    setReviewQueueTraceStatusAsync: mutateAsync,
    isSettingStatus: isLoading,
    error,
  };
};
