import React, { useCallback } from 'react';

import type { ModelTrace } from './ModelTrace.types';
import { GenericSkeleton, Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { ModelTraceExplorerDetailView } from './ModelTraceExplorerDetailView';
import { useModelTraceExplorerViewState } from './ModelTraceExplorerViewStateContext';
import { ModelTraceExplorerSummaryView } from './summary-view/ModelTraceExplorerSummaryView';
import { ModelTraceExplorerLinkedPromptsView } from './linked-prompts/ModelTraceExplorerLinkedPromptsView';
import { shouldEnableModelTraceExplorerCustomTraceView } from './FeatureUtils';
import { useCustomViewAssistantConnector } from './custom-view/assistant/CustomViewAssistantConnector';

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
}: {
  modelTraceInfo: ModelTrace['info'];
  className?: string;
  selectedSpanId?: string;
  onSelectSpan?: (selectedSpanId?: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const { activeView, setActiveView, rootNode } = useModelTraceExplorerViewState();
  // Gate the tab on a usable connector: without a host-provided openAssistant
  // (e.g. multi-experiment views, or consumers that don't mount the provider)
  // there is no way to author a view, so the tab would be a dead end.
  const { openAssistant } = useCustomViewAssistantConnector();
  const isCustomViewEnabled = shouldEnableModelTraceExplorerCustomTraceView() && Boolean(openAssistant);

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
        // this is to remove the margin at the bottom of the <Tabs> component
        '& > div:nth-of-type(1)': {
          marginBottom: 0,
          flexShrink: 0,
        },
        display: 'flex',
        flex: 1,
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      <Tabs.List css={{ paddingLeft: theme.spacing.md, flexShrink: 0 }}>
        {rootNode && (
          <Tabs.Trigger value="summary">
            <FormattedMessage
              defaultMessage="Summary"
              description="Label for the summary view tab in the model trace explorer"
            />
          </Tabs.Trigger>
        )}
        <Tabs.Trigger value="detail">
          <FormattedMessage
            defaultMessage="Details & Timeline"
            description="Label for the details & timeline view tab in the model trace explorer"
          />
        </Tabs.Trigger>
        <Tabs.Trigger value="prompts">
          <FormattedMessage
            defaultMessage="Linked prompts"
            description="Label for the linked prompts view tab in the model trace explorer"
          />
        </Tabs.Trigger>
        {isCustomViewEnabled && (
          <Tabs.Trigger value="custom">
            <FormattedMessage
              defaultMessage="Custom view"
              description="Label for the custom view tab in the model trace explorer"
            />
          </Tabs.Trigger>
        )}
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
      <Tabs.Content
        value="prompts"
        css={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
        }}
      >
        <ModelTraceExplorerLinkedPromptsView modelTraceInfo={modelTraceInfo} />
      </Tabs.Content>
      {isCustomViewEnabled && (
        <Tabs.Content
          value="custom"
          mountMode="preserve"
          css={{
            display: 'flex',
            flexDirection: 'column',
            flex: 1,
            minHeight: 0,
          }}
        >
          <React.Suspense fallback={<GenericSkeleton css={{ height: '100%', width: '100%' }} />}>
            <LazyModelTraceExplorerCustomView modelTraceInfo={modelTraceInfo} />
          </React.Suspense>
        </Tabs.Content>
      )}
    </Tabs.Root>
  );
};
