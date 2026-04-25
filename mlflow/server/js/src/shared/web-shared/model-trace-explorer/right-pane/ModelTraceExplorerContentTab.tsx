import { useCallback } from 'react';

import type { RadioChangeEvent } from '@databricks/design-system';
import { SegmentedControlButton, SegmentedControlGroup, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerDefaultSpanView } from './ModelTraceExplorerDefaultSpanView';
import type { ModelTraceExplorerRenderMode, ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { useModelTraceExplorerPreferences } from '../ModelTraceExplorerPreferencesContext';
import { SpanModelCostBadge } from './SpanModelCostBadge';

export function ModelTraceExplorerContentTab({
  activeSpan,
  className,
  searchFilter,
  activeMatch,
}: {
  activeSpan: ModelTraceSpanNode | undefined;
  className?: string;
  searchFilter: string;
  activeMatch: SearchMatch | null;
}) {
  const { theme } = useDesignSystemTheme();
  const { renderMode, setRenderMode } = useModelTraceExplorerPreferences();

  const handleSetRenderMode = useCallback(
    (event: RadioChangeEvent) => {
      setRenderMode(event.target.value as ModelTraceExplorerRenderMode);
    },
    [setRenderMode],
  );

  return (
    <div
      css={{
        overflowY: 'auto',
        paddingTop: theme.spacing.sm,
      }}
      className={className}
      data-testid="model-trace-explorer-content-tab"
    >
      <div
        css={{
          display: 'flex',
          justifyContent: 'flex-end',
          marginBottom: theme.spacing.sm,
          marginRight: 'auto',
          paddingInline: theme.spacing.sm,
        }}
      >
        <SpanModelCostBadge css={{ marginRight: 'auto' }} activeSpan={activeSpan} />
        <SegmentedControlGroup
          name="content-tab-render-mode"
          componentId="shared.model-trace-explorer.content-tab.render-mode"
          value={renderMode}
          size="small"
          onChange={handleSetRenderMode}
        >
          <SegmentedControlButton value="default">
            <FormattedMessage
              defaultMessage="Default"
              description="Label for the default render mode in the model trace explorer inputs/outputs tab"
            />
          </SegmentedControlButton>
          <SegmentedControlButton value="json">
            <FormattedMessage
              defaultMessage="JSON"
              description="Label for the JSON render mode in the model trace explorer inputs/outputs tab"
            />
          </SegmentedControlButton>
          <SegmentedControlButton value="table">
            <FormattedMessage
              defaultMessage="Table"
              description="Label for the Table render mode in the model trace explorer inputs/outputs tab"
            />
          </SegmentedControlButton>
        </SegmentedControlGroup>
      </div>
      <ModelTraceExplorerDefaultSpanView
        activeSpan={activeSpan}
        className={className}
        searchFilter={searchFilter}
        activeMatch={activeMatch}
        renderMode={renderMode}
      />
    </div>
  );
}
