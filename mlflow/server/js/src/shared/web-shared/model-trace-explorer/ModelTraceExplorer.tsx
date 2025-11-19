import { useCallback, useEffect, useState } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

import { Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { getLargeTraceDisplaySizeThreshold, shouldBlockLargeTraceDisplay } from './FeatureUtils';
import type { ModelTrace } from './ModelTrace.types';
import { getModelTraceId, getModelTraceSize } from './ModelTraceExplorer.utils';
import { ModelTraceExplorerDetailView } from './ModelTraceExplorerDetailView';
import { ModelTraceExplorerErrorState } from './ModelTraceExplorerErrorState';
import { ModelTraceExplorerGenericErrorState } from './ModelTraceExplorerGenericErrorState';
import { ModelTraceExplorerTraceTooLargeView } from './ModelTraceExplorerTraceTooLargeView';
import {
  ModelTraceExplorerViewStateProvider,
  useModelTraceExplorerViewState,
} from './ModelTraceExplorerViewStateContext';
import { ModelTraceHeaderDetails } from './ModelTraceHeaderDetails';
import { useGetModelTraceInfo } from './hooks/useGetModelTraceInfo';
import { useTraceCachedActions } from './hooks/useTraceCachedActions';
import { ModelTraceExplorerSummaryView } from './summary-view/ModelTraceExplorerSummaryView';

const ModelTraceExplorerContent = ({
  modelTraceInfo,
  className,
  selectedSpanId,
  onSelectSpan,
}: {
  modelTraceInfo: ModelTrace['info'];
  className?: string;
  selectedSpanId?: string;
  onSelectSpan?: (selectedSpanId?: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const { activeView, setActiveView } = useModelTraceExplorerViewState();

  const handleValueChange = useCallback(
    (value: string) => {
      setActiveView(value as 'summary' | 'detail');
    },
    [
      // prettier-ignore
      setActiveView,
    ],
  );

  return (
    <Tabs.Root
      componentId="shared.model-trace-explorer.view-mode-toggle"
      value={activeView}
      onValueChange={handleValueChange}
      css={{
        '& > div:nth-of-type(2)': {
          marginBottom: 0,
          flexShrink: 0,
        },
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      <div css={{ paddingLeft: theme.spacing.md, paddingBottom: theme.spacing.sm }}>
        <ModelTraceHeaderDetails modelTraceInfo={modelTraceInfo} />
      </div>
      <Tabs.List css={{ paddingLeft: theme.spacing.md, flexShrink: 0 }}>
        <Tabs.Trigger value="summary">
          <FormattedMessage
            defaultMessage="Summary"
            description="Label for the summary view tab in the model trace explorer"
          />
        </Tabs.Trigger>
        <Tabs.Trigger value="detail">
          <FormattedMessage
            defaultMessage="Details & Timeline"
            description="Label for the details & timeline view tab in the model trace explorer"
          />
        </Tabs.Trigger>
      </Tabs.List>
      <Tabs.Content
        value="summary"
        css={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
        }}
      >
        <ModelTraceExplorerSummaryView />
      </Tabs.Content>
      <Tabs.Content
        value="detail"
        css={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
        }}
      >
        <ModelTraceExplorerDetailView
          modelTraceInfo={modelTraceInfo}
          className={className}
          selectedSpanId={selectedSpanId}
          onSelectSpan={onSelectSpan}
        />
      </Tabs.Content>
    </Tabs.Root>
  );
};

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
}: {
  modelTrace: ModelTrace;
  className?: string;
  initialActiveView?: 'summary' | 'detail';
  selectedSpanId?: string;
  onSelectSpan?: (selectedSpanId?: string) => void;
  collapseAssessmentPane?: boolean;
}) => {
  const [modelTrace, setModelTrace] = useState(initialModelTrace);
  const [forceDisplay, setForceDisplay] = useState(false);
  const traceId = getModelTraceId(initialModelTrace);
  // older traces don't have a size, so we default to 0 to always display them
  const size = getModelTraceSize(initialModelTrace) ?? 0;
  // always displayable if the feature flag is disabled
  const isDisplayable = shouldBlockLargeTraceDisplay() ? size < getLargeTraceDisplaySizeThreshold() : true;
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
  }, [traceId]);

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
        initialAssessmentsPaneCollapsed={collapseAssessmentPane}
        isTraceInitialLoading={isTraceInitialLoading}
      >
        <ModelTraceExplorerContent
          modelTraceInfo={modelTrace.info}
          className={className}
          selectedSpanId={selectedSpanId}
          onSelectSpan={onSelectSpan}
        />
      </ModelTraceExplorerViewStateProvider>
    </ContextProviders>
  );
};

export const ModelTraceExplorer = ModelTraceExplorerImpl;
