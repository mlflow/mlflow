import { useCallback } from 'react';

import type { ModelTrace, ModelTraceExplorerDisplayMode } from '@databricks/web-shared/model-trace-explorer';
import {
  GET_TRACE_BY_ID,
  ModelTraceExplorerUpdateTraceContextProvider,
} from '@databricks/web-shared/model-trace-explorer';
import {
  ModelTraceExplorerCustomViewRenderer,
  ModelTraceExplorerCustomViewSelector,
  useOptionalCustomViewDefinition,
} from '@databricks/web-shared/model-trace-explorer/custom-view';
import { useQueryClient } from '@databricks/web-shared/query-client';

import { REVIEW_QUEUE_TRACE_ASSESSMENTS_QUERY_KEY } from './hooks/useTraceAssessmentsQuery';

export interface ReviewQueueCustomViewSelectorProps {
  value: ModelTraceExplorerDisplayMode;
  onValueChange: (mode: ModelTraceExplorerDisplayMode) => void;
  isCustomViewEnabled: boolean;
  onCreateCustomView?: () => void;
}

export const ReviewQueueCustomViewSelector = ({
  value,
  onValueChange,
  isCustomViewEnabled,
  onCreateCustomView,
}: ReviewQueueCustomViewSelectorProps): JSX.Element => {
  const customViewDefinition = useOptionalCustomViewDefinition();
  const canCreateCustomView = Boolean(
    isCustomViewEnabled &&
    onCreateCustomView &&
    customViewDefinition?.isLoaded &&
    !customViewDefinition.hasReachedViewLimit,
  );

  return (
    <ModelTraceExplorerCustomViewSelector
      value={value}
      onValueChange={onValueChange}
      onCreateCustomView={onCreateCustomView}
      isCustomViewEnabled={isCustomViewEnabled}
      canCreateCustomView={canCreateCustomView}
      size="small"
      componentId="mlflow.experiment-review-queue.focused-review.custom-view-selector"
    />
  );
};

export interface ReviewQueueCustomViewRendererProps {
  modelTrace: ModelTrace;
  enableFeedback: boolean;
}

export const ReviewQueueCustomViewRenderer = ({
  modelTrace,
  enableFeedback,
}: ReviewQueueCustomViewRendererProps): JSX.Element => {
  const queryClient = useQueryClient();
  // OSS has no optimistic assessment cache, so after a feedback write refetch what focused review reads.
  const invalidateTraceQuery = useCallback(
    (traceId?: string) => {
      queryClient.invalidateQueries({ queryKey: [GET_TRACE_BY_ID, traceId] });
      queryClient.invalidateQueries({ queryKey: [REVIEW_QUEUE_TRACE_ASSESSMENTS_QUERY_KEY, traceId] });
    },
    [queryClient],
  );
  return (
    <ModelTraceExplorerUpdateTraceContextProvider
      modelTraceInfo={modelTrace.info}
      invalidateTraceQuery={invalidateTraceQuery}
    >
      <ModelTraceExplorerCustomViewRenderer modelTrace={modelTrace} enableFeedback={enableFeedback} />
    </ModelTraceExplorerUpdateTraceContextProvider>
  );
};
