import { useCallback } from 'react';

import type { ModelTrace } from './ModelTrace.types';
import { SidebarOpenIcon, SparkleIcon, Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { useAssistant, useRegisterAssistantContext } from '@mlflow/mlflow/src/assistant';
import { ModelTraceExplorerDetailView } from './ModelTraceExplorerDetailView';
import { isV3ModelTraceInfo } from './ModelTraceExplorer.utils';
import { useModelTraceExplorerViewState } from './ModelTraceExplorerViewStateContext';
import { ModelTraceExplorerSummaryView } from './summary-view/ModelTraceExplorerSummaryView';
import { ModelTraceExplorerLinkedPromptsView } from './linked-prompts/ModelTraceExplorerLinkedPromptsView';

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
  const { isLocalServer, openPanel } = useAssistant();

  const traceId = isV3ModelTraceInfo(modelTraceInfo) ? modelTraceInfo.trace_id : (modelTraceInfo.request_id ?? '');

  // Attach the trace the user is viewing as assistant page context so that, when the panel
  // opens, it is populated with this trace id. Only on a local server (assistant is local-only).
  useRegisterAssistantContext('traceId', isLocalServer && traceId ? traceId : null);

  const handleValueChange = useCallback(
    (value: string) => {
      // The "analyze" tab is an action that opens the assistant side panel rather than a
      // content view, so it must not change the active trace content.
      if (value === 'analyze') {
        openPanel();
        return;
      }
      setActiveView(value as 'summary' | 'detail');
    },
    [openPanel, setActiveView],
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
        {isLocalServer && (
          <Tabs.Trigger value="analyze">
            <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <SparkleIcon color="ai" />
              <FormattedMessage
                defaultMessage="Analyze in Assistant"
                description="Label for the tab that opens the MLflow assistant side panel to analyze the current trace"
              />
              <SidebarOpenIcon css={{ fontSize: theme.typography.fontSizeSm }} />
            </span>
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
    </Tabs.Root>
  );
};
