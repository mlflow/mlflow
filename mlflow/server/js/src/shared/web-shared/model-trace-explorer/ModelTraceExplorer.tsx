import { useEffect, useState } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

import { Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { shouldEnableSummaryView } from './FeatureUtils';
import type { ModelTrace } from './ModelTrace.types';
import { getModelTraceId } from './ModelTraceExplorer.utils';
import { ModelTraceExplorerDetailView } from './ModelTraceExplorerDetailView';
import { ModelTraceExplorerErrorState } from './ModelTraceExplorerErrorState';
import { ModelTraceExplorerSkeleton } from './ModelTraceExplorerSkeleton';
import {
  ModelTraceExplorerViewStateProvider,
  useModelTraceExplorerViewState,
} from './ModelTraceExplorerViewStateContext';
import { useGetModelTraceInfoV3 } from './hooks/useGetModelTraceInfoV3';
import { ModelTraceExplorerSummaryView } from './summary-view/ModelTraceExplorerSummaryView';
import { ModelTraceHeaderDetails } from './ModelTraceHeaderDetails';

const ModelTraceExplorerImpl = ({
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
  const enableSummaryView = shouldEnableSummaryView();
  const { activeView, setActiveView } = useModelTraceExplorerViewState();

  if (!enableSummaryView) {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
        <ModelTraceExplorerDetailView
          modelTrace={modelTrace}
          className={className}
          selectedSpanId={selectedSpanId}
          onSelectSpan={onSelectSpan}
        />
      </div>
    );
  }

  return (
    <Tabs.Root
      componentId="model-trace-explorer"
      value={activeView}
      onValueChange={(value) => setActiveView(value as 'summary' | 'detail')}
      css={{
        '& > div:first-of-type': {
          marginBottom: 0,
          flexShrink: 0,
        },
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      <div css={{ paddingLeft: theme.spacing.md, paddingBottom: theme.spacing.md }}>
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

export const ModelTraceExplorer = ({
  modelTrace: initialModelTrace,
  className,
  initialActiveView = 'summary',
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
  const [assessmentsPaneEnabled, setAssessmentsPaneEnabled] = useState(true);
  const traceId = getModelTraceId(initialModelTrace);

  useGetModelTraceInfoV3({
    traceId,
    setModelTrace,
    setAssessmentsPaneEnabled,
  });

  useEffect(() => {
    setModelTrace(initialModelTrace);
    // reset the model trace when the traceId changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [traceId]);

  return (
    <ContextProviders traceId={traceId}>
      <ModelTraceExplorerViewStateProvider
        modelTrace={modelTrace}
        initialActiveView={initialActiveView}
        selectedSpanIdOnRender={selectedSpanId}
        assessmentsPaneEnabled={assessmentsPaneEnabled}
      >
        <ModelTraceExplorerImpl
          modelTrace={modelTrace}
          className={className}
          selectedSpanId={selectedSpanId}
          onSelectSpan={onSelectSpan}
        />
      </ModelTraceExplorerViewStateProvider>
    </ContextProviders>
  );
};

ModelTraceExplorer.Skeleton = ModelTraceExplorerSkeleton;
