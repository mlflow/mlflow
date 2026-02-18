import { Button, ChevronLeftIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { spanTimeFormatter } from '../timeline-tree/TimelineTree.utils';
import type { WorkflowNode } from './GraphView.types';

interface GraphViewSpanNavigatorProps {
  selectedWorkflowNode: WorkflowNode | null;
  currentSpanIndex: number;
  totalSpans: number;
  currentSpan: ModelTraceSpanNode | null;
  onNavigatePrev: () => void;
  onNavigateNext: () => void;
}

export const GraphViewSpanNavigator = ({
  selectedWorkflowNode,
  currentSpanIndex,
  totalSpans,
  currentSpan,
  onNavigatePrev,
  onNavigateNext,
}: GraphViewSpanNavigatorProps) => {
  const { theme } = useDesignSystemTheme();

  const currentSpanDuration = currentSpan ? spanTimeFormatter(currentSpan.end - currentSpan.start) : null;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: theme.spacing.xs,
        padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
        borderTop: `1px solid ${theme.colors.border}`,
        borderBottom: `1px solid ${theme.colors.border}`,
        backgroundColor: theme.colors.backgroundSecondary,
        flexShrink: 0,
        minHeight: 36,
      }}
    >
      {selectedWorkflowNode ? (
        <>
          <Button
            componentId="graph-view-span-navigator-prev"
            icon={<ChevronLeftIcon />}
            size="small"
            onClick={onNavigatePrev}
            disabled={currentSpanIndex <= 0}
            aria-label="Previous span"
          />
          <Typography.Text size="sm">
            {currentSpanIndex + 1} / {totalSpans}
          </Typography.Text>
          <Button
            componentId="graph-view-span-navigator-next"
            icon={<ChevronRightIcon />}
            size="small"
            onClick={onNavigateNext}
            disabled={currentSpanIndex >= totalSpans - 1}
            aria-label="Next span"
          />
          <Typography.Text size="sm" bold>
            {selectedWorkflowNode.displayName}
          </Typography.Text>
          {currentSpanDuration && (
            <Typography.Text size="sm" color="secondary">
              ({currentSpanDuration})
            </Typography.Text>
          )}
        </>
      ) : (
        <Typography.Text size="sm" color="secondary">
          <FormattedMessage
            defaultMessage="Click a graph node to navigate spans"
            description="Hint text shown in graph view span navigator when no node is selected"
          />
        </Typography.Text>
      )}
    </div>
  );
};
