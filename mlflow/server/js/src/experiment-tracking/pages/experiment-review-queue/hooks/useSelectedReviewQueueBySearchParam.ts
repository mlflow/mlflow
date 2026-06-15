import { useCallback } from 'react';

import { useSearchParams } from '../../../../common/utils/RoutingUtils';

export const SELECTED_REVIEW_QUEUE_ID_QUERY_PARAM_KEY = 'selectedQueueId';

/**
 * Query-param-driven selected review-queue id, mirroring
 * {@link useSelectedDatasetBySearchParam}. Keeping the selection in the URL makes
 * it shareable and deep-linkable — e.g. the "Added to review" toast links straight
 * to the queue the traces landed in. Writes use `replace` so selecting a queue
 * doesn't stack history entries.
 */
export const useSelectedReviewQueueBySearchParam = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedQueueId = searchParams.get(SELECTED_REVIEW_QUEUE_ID_QUERY_PARAM_KEY) ?? undefined;

  const setSelectedQueueId = useCallback(
    (queueId: string | undefined) => {
      setSearchParams(
        (params) => {
          if (!queueId) {
            params.delete(SELECTED_REVIEW_QUEUE_ID_QUERY_PARAM_KEY);
          } else {
            params.set(SELECTED_REVIEW_QUEUE_ID_QUERY_PARAM_KEY, queueId);
          }
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  return [selectedQueueId, setSelectedQueueId] as const;
};
