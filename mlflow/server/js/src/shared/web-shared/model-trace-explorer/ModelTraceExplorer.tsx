import { useEffect, useState } from 'react';
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
import { useGetModelTraceInfoV3 } from './hooks/useGetModelTraceInfoV3';
import { ModelTraceExplorerSummaryView } from './summary-view/ModelTraceExplorerSummaryView';
import { ModelTraceHeaderDetails } from './ModelTraceHeaderDetails';

const ModelTraceExplorerContent = ({
  modelTrace,
  className,
  selectedSpanId,
  onSelectSpan,
}: {
  modelTrace: ModelTrace;
  className?: string;
  selectedSpanId?: string;
  onSelectSpan?: (selectedSpanId?: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const { activeView, setActiveView } = useModelTraceExplorerViewState();

  return (
    <Tabs.Root
      componentId="model-trace-explorer"
      value={activeView}
      onValueChange={(value) => setActiveView(value as 'summary' | 'detail')}
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
        <ModelTraceHeaderDetails modelTrace={modelTrace} />
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
        <ModelTraceExplorerSummaryView modelTrace={modelTrace} />
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
          modelTrace={modelTrace}
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
}: {
  modelTrace: ModelTrace;
  className?: string;
  initialActiveView?: 'summary' | 'detail';
  selectedSpanId?: string;
  onSelectSpan?: (selectedSpanId?: string) => void;
}) => {
  const [modelTrace, setModelTrace] = useState(initialModelTrace);
  const [forceDisplay, setForceDisplay] = useState(false);
  const traceId = getModelTraceId(initialModelTrace);
  // older traces don't have a size, so we default to 0 to always display them
  const size = getModelTraceSize(initialModelTrace) ?? 0;
  // always displayable if the feature flag is disabled
  const isDisplayable = shouldBlockLargeTraceDisplay() ? size < getLargeTraceDisplaySizeThreshold() : true;
  const [assessmentsPaneEnabled, setAssessmentsPaneEnabled] = useState(traceId.startsWith('tr-'));

  useGetModelTraceInfoV3({
    traceId,
    setModelTrace,
    setAssessmentsPaneEnabled,
    enabled: isDisplayable && traceId.startsWith('tr-'),
  });

  useEffect(() => {
    setModelTrace(initialModelTrace);
    // reset the model trace when the traceId changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [traceId]);

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
      >
        <ModelTraceExplorerContent
          modelTrace={modelTrace}
          className={className}
          selectedSpanId={selectedSpanId}
          onSelectSpan={onSelectSpan}
        />
      </ModelTraceExplorerViewStateProvider>
    </ContextProviders>
  );
};

export const ModelTraceExplorer = ModelTraceExplorerImpl;
