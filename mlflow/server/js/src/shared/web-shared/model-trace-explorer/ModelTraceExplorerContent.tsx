import { Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { ModelTrace } from './ModelTrace.types';
import { useModelTraceExplorerViewState } from './ModelTraceExplorerViewStateContext';
import { useCallback } from 'react';
import { ModelTraceExplorerSummaryView } from './summary-view/ModelTraceExplorerSummaryView';
import { ModelTraceExplorerDetailView } from './ModelTraceExplorerDetailView';

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
