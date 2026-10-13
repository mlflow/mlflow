import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { ReviewQueueItem, ReviewStatus } from '../types';
import { REVIEW_QUEUES_API_BASE } from './constants';
import { LIST_REVIEW_QUEUE_ITEMS_QUERY_KEY } from './useListReviewQueueItemsQuery';

export interface SetReviewQueueItemStatusParams {
  queue_id: string;
  item_id: string;
  status: ReviewStatus;
  /** Required for COMPLETE / DECLINED; must be omitted for PENDING (reopen). */
  completed_by?: string;
}

interface SetReviewQueueItemStatusResponse {
  item: ReviewQueueItem;
}

/**
 * Set the shared-pool status of an attached item (the reviewer's
 * complete / decline / reopen action). Invalidates the queue's item list
 * so the table reflects the new status.
 */
export const useSetReviewQueueItemStatusMutation = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation<
    SetReviewQueueItemStatusResponse,
    Error,
    SetReviewQueueItemStatusParams
  >({
    mutationFn: async (params) => {
      return (await fetchAPI(getAjaxUrl(`${REVIEW_QUEUES_API_BASE}/items/set-status`), {
        method: 'POST',
        body: params,
      })) as SetReviewQueueItemStatusResponse;
    },
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_REVIEW_QUEUE_ITEMS_QUERY_KEY]);
    },
  });

  return {
    setReviewQueueItemStatus: mutate,
    setReviewQueueItemStatusAsync: mutateAsync,
    isSettingStatus: isLoading,
    error,
  };
};
