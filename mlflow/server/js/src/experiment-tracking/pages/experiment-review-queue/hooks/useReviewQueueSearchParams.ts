import { useCallback } from 'react';

import { generatePath, useSearchParams } from '../../../../common/utils/RoutingUtils';
import { RoutePaths } from '../../../routes';

export const SELECTED_QUEUE_ID_QUERY_PARAM_KEY = 'selectedQueueId';
export const SELECTED_ITEM_ID_QUERY_PARAM_KEY = 'selectedItemId';
export const START_REVIEW_QUERY_PARAM_KEY = 'startReview';

/**
 * Build the in-app route to the Review tab, optionally deep-linking to a
 * queue. With `startReview`, opening the link drops the visitor straight into
 * the focused review of the queue's first to-do trace (see the consuming
 * effect in `ExperimentReviewQueuePage`). With `startReview` and no
 * `queueId`, the page's auto-select resolves the visitor's own USER queue
 * first — a generic "start reviewing your queue" link.
 */
export const getReviewQueuePageRoute = (
  experimentId: string,
  queueId?: string,
  { startReview = false }: { startReview?: boolean } = {},
) => {
  const path = generatePath(RoutePaths.experimentPageTabReviewQueue, { experimentId });
  const params = new URLSearchParams();
  if (queueId) {
    params.set(SELECTED_QUEUE_ID_QUERY_PARAM_KEY, queueId);
  }
  if (startReview) {
    params.set(START_REVIEW_QUERY_PARAM_KEY, 'true');
  }
  const query = params.toString();
  return query ? `${path}?${query}` : path;
};

/**
 * Query param-powered selection state for the Review tab, making the selected
 * queue and the trace open in focused review shareable URLs (same pattern as
 * the datasets tab's `useSelectedDatasetBySearchParam`).
 *
 * `startReview` is a one-shot intent param ("open the first to-do trace"),
 * not state: the page consumes it via `consumeStartReview` once the queue's
 * items have loaded, so reload/back after that point lands on the focused
 * trace itself (`selectedItemId`), not a re-triggered jump.
 */
export const useReviewQueueSearchParams = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedQueueId = searchParams.get(SELECTED_QUEUE_ID_QUERY_PARAM_KEY) ?? undefined;
  const openItemId = searchParams.get(SELECTED_ITEM_ID_QUERY_PARAM_KEY);
  const startReviewRequested = searchParams.has(START_REVIEW_QUERY_PARAM_KEY);

  const selectQueue = useCallback(
    (queueId: string | undefined, { preserveStartReview = false }: { preserveStartReview?: boolean } = {}) => {
      setSearchParams(
        (params) => {
          if (queueId) {
            params.set(SELECTED_QUEUE_ID_QUERY_PARAM_KEY, queueId);
          } else {
            params.delete(SELECTED_QUEUE_ID_QUERY_PARAM_KEY);
          }
          // Switching queues closes any focused trace; an explicit user
          // selection also voids a pending start-review intent (auto-select
          // preserves it so a bare `?startReview=true` link can resolve the
          // visitor's own queue first).
          params.delete(SELECTED_ITEM_ID_QUERY_PARAM_KEY);
          if (!preserveStartReview) {
            params.delete(START_REVIEW_QUERY_PARAM_KEY);
          }
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const setOpenItemId = useCallback(
    (itemId: string | null) => {
      setSearchParams(
        (params) => {
          if (itemId) {
            params.set(SELECTED_ITEM_ID_QUERY_PARAM_KEY, itemId);
          } else {
            params.delete(SELECTED_ITEM_ID_QUERY_PARAM_KEY);
          }
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const consumeStartReview = useCallback(
    (itemId: string | null) => {
      setSearchParams(
        (params) => {
          params.delete(START_REVIEW_QUERY_PARAM_KEY);
          if (itemId) {
            params.set(SELECTED_ITEM_ID_QUERY_PARAM_KEY, itemId);
          }
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  return { selectedQueueId, openItemId, startReviewRequested, selectQueue, setOpenItemId, consumeStartReview };
};
