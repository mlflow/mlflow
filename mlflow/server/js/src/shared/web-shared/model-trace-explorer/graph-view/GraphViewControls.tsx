import { useCallback } from 'react';

import { useDesignSystemTheme, SegmentedControlGroup, SegmentedControlButton } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { GraphViewMode } from './GraphView.types';

interface GraphViewControlsProps {
  viewMode: GraphViewMode;
  setViewMode: (mode: GraphViewMode) => void;
  hasLogicalWorkflow: boolean;
}

/**
 * View mode toggle controls for the graph view.
 * Positioned in the top-left corner of the canvas.
 * Zoom/pan controls are handled by React Flow natively.
 */
export const GraphViewControls = ({ viewMode, setViewMode, hasLogicalWorkflow }: GraphViewControlsProps) => {
  const { theme } = useDesignSystemTheme();

  const handleViewModeChange = useCallback(
    (value: string) => {
      setViewMode(value as GraphViewMode);
    },
    [setViewMode],
  );

  return (
    <div
      css={{
        position: 'absolute',
        top: theme.spacing.sm,
        left: theme.spacing.sm,
        zIndex: 10,
        backgroundColor: theme.colors.backgroundPrimary,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.border}`,
        padding: theme.spacing.xs,
        boxShadow: theme.general.shadowLow,
      }}
    >
      <SegmentedControlGroup
        componentId="shared.model-trace-explorer.graph-view-mode"
        name="graph-view-mode"
        value={viewMode}
        onChange={({ target: { value } }) => handleViewModeChange(value)}
      >
        <SegmentedControlButton value="all_spans">
          <FormattedMessage defaultMessage="All Spans" description="Button to show all spans in graph view" />
        </SegmentedControlButton>
        <SegmentedControlButton value="logical_workflow" disabled={!hasLogicalWorkflow}>
          <FormattedMessage defaultMessage="Workflow" description="Button to show logical workflow in graph view" />
        </SegmentedControlButton>
      </SegmentedControlGroup>
    </div>
  );
};
