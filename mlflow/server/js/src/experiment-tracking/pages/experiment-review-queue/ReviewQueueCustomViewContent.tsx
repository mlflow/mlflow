import type { ModelTrace, ModelTraceExplorerDisplayMode } from '@databricks/web-shared/model-trace-explorer';
import { ModelTraceExplorerUpdateTraceContextProvider } from '@databricks/web-shared/model-trace-explorer';
import {
  ModelTraceExplorerCustomViewRenderer,
  ModelTraceExplorerCustomViewSelector,
  useOptionalCustomViewDefinition,
} from '@databricks/web-shared/model-trace-explorer/custom-view';

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
}: ReviewQueueCustomViewRendererProps): JSX.Element => (
  <ModelTraceExplorerUpdateTraceContextProvider modelTraceInfo={modelTrace.info}>
    <ModelTraceExplorerCustomViewRenderer modelTrace={modelTrace} enableFeedback={enableFeedback} />
  </ModelTraceExplorerUpdateTraceContextProvider>
);
