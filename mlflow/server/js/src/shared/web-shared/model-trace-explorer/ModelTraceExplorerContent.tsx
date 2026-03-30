import { useCallback } from 'react';

import type { ModelTrace } from './ModelTrace.types';
import { Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { ModelTraceExplorerDetailView } from './ModelTraceExplorerDetailView';
import { useModelTraceExplorerViewState } from './ModelTraceExplorerViewStateContext';
import { TraceViewSelector } from './TraceViewSelector';
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
  const { activeView, setActiveView, rootNode, activeTraceView, setActiveTraceView } = useModelTraceExplorerViewState();

  const traceId = rootNode?.traceId ?? null;

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
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0,
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
        </Tabs.List>
        <div css={{ paddingRight: theme.spacing.md }}>
          <TraceViewSelector
            traceId={traceId}
            activeViewId={activeTraceView?.view_id ?? null}
            onViewChange={setActiveTraceView}
          />
        </div>
      </div>
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
