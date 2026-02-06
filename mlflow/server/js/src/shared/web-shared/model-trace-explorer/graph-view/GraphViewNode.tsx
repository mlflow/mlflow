import { memo, useCallback } from 'react';

import { Handle, Position } from '@xyflow/react';
import { Tooltip, useDesignSystemTheme, XCircleIcon } from '@databricks/design-system';

import type { ModelSpanType } from '../ModelTrace.types';
import { getIconTypeForSpan, getSpanExceptionCount } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerIcon } from '../ModelTraceExplorerIcon';
import { spanTimeFormatter } from '../timeline-tree/TimelineTree.utils';
import type { SpanNodeData } from './GraphView.types';
import { getNodeBackgroundColor, getNodeBorderColor, truncateText } from './GraphView.utils';
import { GraphViewNodeTooltip } from './GraphViewNodeTooltip';

interface SpanNodeProps {
  data: SpanNodeData;
  selected?: boolean;
}

/**
 * React Flow custom node for rendering span tree nodes.
 * Shows span icon, name, and duration with colors based on span type.
 */
export const SpanNode = memo(function SpanNode({ data, selected }: SpanNodeProps) {
  const { theme } = useDesignSystemTheme();
  const { spanNode, isSelected, isOnHighlightedPath, onSelect, onViewDetails } = data;

  const spanType = spanNode.type as ModelSpanType | undefined;
  const hasException = getSpanExceptionCount(spanNode) > 0;

  // Error state takes priority for border color
  const backgroundColor = hasException
    ? theme.isDarkMode
      ? theme.colors.red800
      : theme.colors.red100
    : getNodeBackgroundColor(spanType, theme);

  const borderColor = hasException
    ? theme.colors.actionDangerPrimaryBackgroundDefault
    : isSelected || isOnHighlightedPath
      ? theme.colors.actionPrimaryBackgroundDefault
      : getNodeBorderColor(spanType, theme);
  const borderWidth = hasException || isSelected || isOnHighlightedPath ? 2 : 1;

  const iconType = getIconTypeForSpan(spanType ?? 'UNKNOWN');
  const title = truncateText(String(spanNode.title ?? ''), 14);
  const duration = spanTimeFormatter(spanNode.end - spanNode.start);

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onSelect();
    },
    [onSelect],
  );

  return (
    <Tooltip
      componentId="shared.model-trace-explorer.graph-view-node-tooltip"
      content={<GraphViewNodeTooltip span={spanNode} onViewDetails={onViewDetails} />}
      side="right"
      maxWidth={400}
    >
      <div
        onClick={handleClick}
        css={{
          width: 140,
          height: 48,
          backgroundColor,
          border: `${borderWidth}px solid ${borderColor}`,
          borderRadius: theme.borders.borderRadiusMd,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          padding: `0 ${theme.spacing.sm}px`,
          cursor: 'pointer',
          position: 'relative',
          '&:hover': {
            filter: 'brightness(0.95)',
          },
        }}
      >
        {/* Connection handles */}
        <Handle type="target" position={Position.Top} css={{ visibility: 'hidden' }} />
        <Handle type="source" position={Position.Bottom} css={{ visibility: 'hidden' }} />

        {/* Icon and title row */}
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <div css={{ width: 20, height: 20, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <ModelTraceExplorerIcon type={iconType} hasException={hasException} />
          </div>
          <span
            css={{
              fontSize: theme.typography.fontSizeSm,
              fontWeight: theme.typography.typographyBoldFontWeight,
              color: theme.colors.textPrimary,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {title}
          </span>
        </div>

        {/* Duration row */}
        <div
          css={{
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textSecondary,
            marginLeft: 20 + theme.spacing.xs,
          }}
        >
          {duration}
        </div>

        {/* Error indicator badge */}
        {hasException && (
          <div
            css={{
              position: 'absolute',
              top: -8,
              right: -8,
              width: 20,
              height: 20,
              borderRadius: '50%',
              backgroundColor: theme.colors.actionDangerPrimaryBackgroundDefault,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: theme.shadows.sm,
            }}
          >
            <XCircleIcon css={{ width: 14, height: 14, color: 'white' }} />
          </div>
        )}

        {/* Selection indicator ring */}
        {isSelected && (
          <div
            css={{
              position: 'absolute',
              top: -5,
              left: -5,
              right: -5,
              bottom: -5,
              border: `2px dashed ${theme.colors.actionPrimaryBackgroundDefault}`,
              borderRadius: theme.borders.borderRadiusMd + 2,
              opacity: 0.6,
              pointerEvents: 'none',
            }}
          />
        )}
      </div>
    </Tooltip>
  );
});

SpanNode.displayName = 'SpanNode';

// Export the node types map for React Flow
export const spanNodeTypes = {
  spanNode: SpanNode,
};
