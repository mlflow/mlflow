import { useEffect, useState } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

import { getLargeTraceDisplaySizeThreshold, shouldBlockLargeTraceDisplay } from './FeatureUtils';
import type { ModelTrace } from './ModelTrace.types';
import { getModelTraceId, getModelTraceSize } from './ModelTraceExplorer.utils';
import { ModelTraceExplorerErrorState } from './ModelTraceExplorerErrorState';
import { ModelTraceExplorerGenericErrorState } from './ModelTraceExplorerGenericErrorState';
import { ModelTraceExplorerSkeleton } from './ModelTraceExplorerSkeleton';
import { ModelTraceExplorerTraceTooLargeView } from './ModelTraceExplorerTraceTooLargeView';
import { ModelTraceExplorerViewStateProvider } from './ModelTraceExplorerViewStateContext';
import { ModelTraceHeaderDetails } from './ModelTraceHeaderDetails';
import { useGetModelTraceInfo } from './hooks/useGetModelTraceInfo';
import { useTraceCachedActions } from './hooks/useTraceCachedActions';
import { ModelTraceExplorerContent } from './ModelTraceExplorerContent';
import { ModelTraceExplorerComparisonView } from './ModelTraceExplorerComparisonView';

const ContextProviders = ({ children }: { traceId: string; children: React.ReactNode }) => {
  return <ErrorBoundary fallbackRender={ModelTraceExplorerErrorState}>{children}</ErrorBoundary>;
};

export const ModelTraceExplorerImpl = ({
  modelTrace: initialModelTrace,
  className,
  initialActiveView,
  selectedSpanId,
  onSelectSpan,
  collapseAssessmentPane,
  isInComparisonView,
  showLoadingState,
}: {
  modelTrace: ModelTrace;
  className?: string;
  initialActiveView?: 'summary' | 'detail';
  selectedSpanId?: string;
  onSelectSpan?: (selectedSpanId?: string) => void;
  /**
   * If set to `false`, the assessments pane will be expanded if there are any assessments.
   * If set to `'force-open'`, the assessments pane will be expanded regardless of whether there are any assessments.
   */
  collapseAssessmentPane?: boolean | 'force-open';
  isInComparisonView?: boolean;
  showLoadingState?: boolean;
}) => {
  const [modelTrace, setModelTrace] = useState(initialModelTrace);
  const [forceDisplay, setForceDisplay] = useState(false);
  const traceId = getModelTraceId(initialModelTrace);
  // older traces don't have a size, so we default to 0 to always display them
  const size = getModelTraceSize(initialModelTrace) ?? 0;
  // always displayable if the feature flag is disabled
  const isDisplayable = shouldBlockLargeTraceDisplay() ? size < getLargeTraceDisplaySizeThreshold() : true;
  const spanLength = initialModelTrace.data?.spans?.length ?? 0;
  const [assessmentsPaneEnabled, setAssessmentsPaneEnabled] = useState(traceId.startsWith('tr-'));
  const [isMountingTrace, setIsMountingTrace] = useState(true);

  const { isFetching } = useGetModelTraceInfo({
    traceId,
    setModelTrace,
    setAssessmentsPaneEnabled,
    enabled: isDisplayable,
  });

  const isTraceInitialLoading = isMountingTrace && isFetching;

  useEffect(() => {
    setModelTrace(initialModelTrace);
    setIsMountingTrace(true);
    // reset the model trace when the traceId changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [traceId, spanLength]);

  useEffect(() => {
    if (isMountingTrace && !isFetching) {
      setIsMountingTrace(false);
    }
  }, [isMountingTrace, isFetching]);

  const resetActionCache = useTraceCachedActions((state) => state.resetCache);

  // Reset the cache each time a trace explorer is mounted
  useEffect(() => {
    resetActionCache();
  }, [resetActionCache]);

  if (!isDisplayable && !forceDisplay) {
    return <ModelTraceExplorerTraceTooLargeView traceId={traceId} setForceDisplay={setForceDisplay} />;
  }

  return (
    <ContextProviders traceId={traceId}>
      <ModelTraceExplorerViewStateProvider
        modelTrace={modelTrace}
        initialActiveView={initialActiveView}
        selectedSpanIdOnRender={selectedSpanId}
        assessmentsPaneEnabled={assessmentsPaneEnabled}
        isInComparisonView={isInComparisonView}
        initialAssessmentsPaneCollapsed={collapseAssessmentPane}
        isTraceInitialLoading={isTraceInitialLoading}
      >
        {showLoadingState ? (
          <ModelTraceExplorerSkeleton />
        ) : (
          <>
            <ModelTraceHeaderDetails modelTraceInfo={modelTrace.info} />
            {isInComparisonView ? (
              <ModelTraceExplorerComparisonView modelTraceInfo={modelTrace.info} />
            ) : (
              <ModelTraceExplorerContent
                modelTraceInfo={modelTrace.info}
                className={className}
                selectedSpanId={selectedSpanId}
                onSelectSpan={onSelectSpan}
              />
            )}
          </>
        )}
      </ModelTraceExplorerViewStateProvider>
    </ContextProviders>
  );
};

export const ModelTraceExplorer = ModelTraceExplorerImpl;
