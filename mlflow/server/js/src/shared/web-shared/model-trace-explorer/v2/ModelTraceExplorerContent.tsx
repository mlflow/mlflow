import React from 'react';

import type { ModelTrace } from './ModelTrace.types';
import { GenericSkeleton } from '@databricks/design-system';
import { ModelTraceExplorerDetailView } from './ModelTraceExplorerDetailView';
import { useModelTraceExplorerViewState } from './ModelTraceExplorerViewStateContext';
import { ModelTraceExplorerLinkedPromptsView } from '../linked-prompts/ModelTraceExplorerLinkedPromptsView';
import { shouldEnableModelTraceExplorerCustomTraceView } from '../FeatureUtils';
import { useCustomViewAssistantConnector } from '../custom-view/assistant/CustomViewAssistantConnector';
import { useModelTraceExplorerContext } from './ModelTraceExplorerContext';

// Custom View pulls in @a2ui (ESM-only). Lazy-load it so consumers that only
// need the standard trace explorer (e.g. the OSS notebook renderer) do not
// transitively import @a2ui on the static module graph.
const LazyModelTraceExplorerCustomView = React.lazy(() =>
  import('./custom-view/ModelTraceExplorerCustomView').then((module) => ({
    default: module.ModelTraceExplorerCustomView,
  })),
);

export const ModelTraceExplorerContent = ({
  modelTraceInfo,
  className,
  selectedSpanId,
  onSelectSpan,
  enableGraphView = true,
}: {
  modelTraceInfo: ModelTrace['info'];
  className?: string;
  selectedSpanId?: string;
  onSelectSpan?: (selectedSpanId?: string) => void;
  enableGraphView?: boolean;
}): JSX.Element => {
  const { activeView } = useModelTraceExplorerViewState();
  const { traceExplorerDisplayMode } = useModelTraceExplorerContext();
  // Gate custom views on a usable connector: without a host-provided openAssistant
  // e.g. multi-experiment views or consumers that don't mount the provider
  const { openAssistant } = useCustomViewAssistantConnector();
  const isCustomViewEnabled = shouldEnableModelTraceExplorerCustomTraceView() && Boolean(openAssistant);

  if (traceExplorerDisplayMode === 'default' && activeView === 'prompts') {
    return <ModelTraceExplorerLinkedPromptsView modelTraceInfo={modelTraceInfo} />;
  }

  if (isCustomViewEnabled && activeView === 'custom') {
    return (
      <React.Suspense fallback={<GenericSkeleton css={{ height: '100%', width: '100%' }} />}>
        <LazyModelTraceExplorerCustomView modelTraceInfo={modelTraceInfo} />
      </React.Suspense>
    );
  }

  if (isCustomViewEnabled && traceExplorerDisplayMode === 'custom') {
    return (
      <React.Suspense fallback={<GenericSkeleton css={{ height: '100%', width: '100%' }} />}>
        <LazyModelTraceExplorerCustomView modelTraceInfo={modelTraceInfo} />
      </React.Suspense>
    );
  }

  return (
    <ModelTraceExplorerDetailView
      modelTraceInfo={modelTraceInfo}
      className={className}
      selectedSpanId={selectedSpanId}
      onSelectSpan={onSelectSpan}
      enableGraphView={enableGraphView}
    />
  );
};
