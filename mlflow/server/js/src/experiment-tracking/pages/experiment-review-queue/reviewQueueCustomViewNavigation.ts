import { CREATE_CUSTOM_VIEW_QUERY_PARAM } from '@databricks/web-shared/model-trace-explorer';

import {
  SELECTED_ITEM_ID_QUERY_PARAM_KEY,
  SELECTED_QUEUE_ID_QUERY_PARAM_KEY,
  START_REVIEW_QUERY_PARAM_KEY,
} from './hooks/useReviewQueueSearchParams';
import { ExperimentPageTabName } from '../../constants';
import Routes from '../../routes';

// Both Traces pages open a trace from this param. V4 reads it only as a fallback for its own
// `traceId`, so a stale `traceId` must be dropped or it would win.
const SELECTED_TRACE_QUERY_PARAM = 'selectedEvaluationId';
const TRACES_V4_TRACE_ID_QUERY_PARAM = 'traceId';
const SPAN_ID_QUERY_PARAM = 'spanId';

export interface ReviewQueueCreateCustomViewUrlParams {
  experimentId: string;
  traceId: string;
  currentSearch: string;
}

/**
 * Builds a hash-router URL for the Traces page that opens the selected trace and asks
 * the drawer to enter its existing custom-view creation flow.
 */
export const getReviewQueueCreateCustomViewUrl = ({
  experimentId,
  traceId,
  currentSearch,
}: ReviewQueueCreateCustomViewUrlParams): string => {
  const searchParams = new URLSearchParams(currentSearch);
  searchParams.delete(SELECTED_QUEUE_ID_QUERY_PARAM_KEY);
  searchParams.delete(SELECTED_ITEM_ID_QUERY_PARAM_KEY);
  searchParams.delete(START_REVIEW_QUERY_PARAM_KEY);
  searchParams.delete(SPAN_ID_QUERY_PARAM);
  searchParams.delete(TRACES_V4_TRACE_ID_QUERY_PARAM);
  searchParams.set(SELECTED_TRACE_QUERY_PARAM, traceId);
  searchParams.set(CREATE_CUSTOM_VIEW_QUERY_PARAM, 'true');

  const path = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Traces);
  return `/#${path}?${searchParams.toString()}`;
};
