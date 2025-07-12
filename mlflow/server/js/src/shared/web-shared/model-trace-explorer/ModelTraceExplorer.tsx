import { useEffect, useMemo, useState } from 'react';

import { Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { shouldEnableSummaryView } from './FeatureUtils';
import type { ModelTrace } from './ModelTrace.types';
import { getModelTraceId } from './ModelTraceExplorer.utils';
import { ModelTraceExplorerDetailView } from './ModelTraceExplorerDetailView';
import { ModelTraceExplorerSkeleton } from './ModelTraceExplorerSkeleton';
import {
  ModelTraceExplorerViewStateProvider,
  useModelTraceExplorerViewState,
} from './ModelTraceExplorerViewStateContext';
import { useGetModelTraceInfoV3 } from './hooks/useGetModelTraceInfoV3';
import { ModelTraceExplorerSummaryView } from './summary-view/ModelTraceExplorerSummaryView';
import type { ModelTraceInfoRefetchContextType } from './trace-context/ModelTraceInfoRefetchContext';
import { ModelTraceInfoRefetchContext } from './trace-context/ModelTraceInfoRefetchContext';

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

const ContextProviders = ({
  refetchContextValue,
  children,
}: {
  traceId: string;
  refetchContextValue: ModelTraceInfoRefetchContextType;
  children: React.ReactNode;
}) => {
  return (
    <ModelTraceInfoRefetchContext.Provider value={refetchContextValue}>
      {children}
    </ModelTraceInfoRefetchContext.Provider>
  );
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
  const { refetch } = useGetModelTraceInfoV3({
    traceId,
    setModelTrace,
    setAssessmentsPaneEnabled,
  });
  const refetchContextValue = useMemo(() => ({ refetchTraceInfo: refetch }), [refetch]);

  useEffect(() => {
    setModelTrace(initialModelTrace);
    refetch();
    // reset the model trace when the traceId changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [traceId]);

  return (
    <ContextProviders traceId={traceId} refetchContextValue={refetchContextValue}>
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
