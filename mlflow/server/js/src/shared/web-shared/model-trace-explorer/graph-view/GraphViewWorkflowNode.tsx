import { memo, useCallback, useMemo, useState } from 'react';

import { Handle, Position } from '@xyflow/react';
import {
  ChevronDownIcon,
  ChevronRightIcon,
  Tooltip,
  useDesignSystemTheme,
  XCircleIcon,
} from '@databricks/design-system';

import type { ModelSpanType, ModelTraceSpanNode } from '../ModelTrace.types';
import { getIconTypeForSpan, getSpanExceptionCount } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerIcon } from '../ModelTraceExplorerIcon';
import { spanTimeFormatter } from '../timeline-tree/TimelineTree.utils';
import type { WorkflowNodeData } from './GraphView.types';
import { getNodeBackgroundColor, getNodeBorderColor, truncateText } from './GraphView.utils';

interface WorkflowNodeTooltipContentProps {
  displayName: string;
  nodeType: string;
  count: number;
  spans: ModelTraceSpanNode[];
  onViewDetails: (span: ModelTraceSpanNode) => void;
}

const WorkflowNodeTooltipContent = ({
  displayName,
  nodeType,
  count,
  spans,
  onViewDetails,
}: WorkflowNodeTooltipContentProps) => {
  const { theme } = useDesignSystemTheme();
  const [isExpanded, setIsExpanded] = useState(false);

  const sortedSpans = useMemo(() => [...spans].sort((a, b) => a.start - b.start), [spans]);

  const handleToggleExpand = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setIsExpanded((prev) => !prev);
  }, []);

  const handleViewSpanDetails = useCallback(
    (e: React.MouseEvent, span: ModelTraceSpanNode) => {
      e.stopPropagation();
      onViewDetails(span);
    },
    [onViewDetails],
  );

  const tooltipTextPrimary = theme.colors.textPrimary;
  const tooltipTextSecondary = theme.colors.textSecondary;
  const tooltipBorder = theme.colors.borderDecorative;
  const tooltipAccent = theme.colors.actionPrimaryBackgroundDefault;
  const tooltipItemBg = theme.isDarkMode ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.04)';
  const tooltipItemHoverBg = theme.isDarkMode ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.08)';
  const tooltipBadgeBg = theme.isDarkMode ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.06)';

  return (
    <div
      css={{
        minWidth: 200,
        backgroundColor: theme.colors.backgroundPrimary,
        color: tooltipTextPrimary,
        padding: theme.spacing.sm,
        borderRadius: theme.borders.borderRadiusMd,
        boxShadow: theme.shadows.lg,
        margin: -theme.spacing.sm,
      }}
    >
      <div
        css={{
          fontWeight: theme.typography.typographyBoldFontWeight,
          marginBottom: theme.spacing.xs,
          color: tooltipTextPrimary,
        }}
      >
        {displayName}
      </div>
      <div css={{ color: tooltipTextSecondary, marginBottom: theme.spacing.xs }}>Type: {nodeType}</div>
      <div css={{ color: tooltipTextSecondary }}>
        {count} execution{count !== 1 ? 's' : ''}
      </div>

      {count > 0 && (
        <div
          css={{
            marginTop: theme.spacing.sm,
            paddingTop: theme.spacing.sm,
            borderTop: `1px solid ${tooltipBorder}`,
          }}
        >
          <div
            onClick={handleToggleExpand}
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
              cursor: 'pointer',
              color: tooltipAccent,
              fontSize: theme.typography.fontSizeSm,
              fontWeight: 500,
              '&:hover': { textDecoration: 'underline' },
            }}
          >
            {isExpanded ? (
              <ChevronDownIcon css={{ width: 14, height: 14 }} />
            ) : (
              <ChevronRightIcon css={{ width: 14, height: 14 }} />
            )}
            {isExpanded ? 'Hide' : 'Show'} {count} span{count !== 1 ? 's' : ''}
          </div>

          {isExpanded && (
            <div css={{ marginTop: theme.spacing.sm, maxHeight: 200, overflowY: 'auto' }}>
              {sortedSpans.map((span, index) => {
                const duration = spanTimeFormatter(span.end - span.start);
                const spanHasError = getSpanExceptionCount(span) > 0;
                return (
                  <div
                    key={span.key}
                    onClick={(e) => handleViewSpanDetails(e, span)}
                    css={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                      marginBottom: 2,
                      borderRadius: theme.borders.borderRadiusMd,
                      backgroundColor: spanHasError
                        ? theme.isDarkMode
                          ? 'rgba(239, 68, 68, 0.2)'
                          : 'rgba(239, 68, 68, 0.1)'
                        : tooltipItemBg,
                      cursor: 'pointer',
                      border: spanHasError ? `1px solid ${theme.colors.actionDangerPrimaryBackgroundDefault}` : 'none',
                      '&:hover': {
                        backgroundColor: spanHasError
                          ? theme.isDarkMode
                            ? 'rgba(239, 68, 68, 0.3)'
                            : 'rgba(239, 68, 68, 0.15)'
                          : tooltipItemHoverBg,
                      },
                    }}
                  >
                    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                      {spanHasError && (
                        <XCircleIcon
                          css={{ width: 12, height: 12, color: theme.colors.actionDangerPrimaryBackgroundDefault }}
                        />
                      )}
                      <span
                        css={{
                          fontSize: 10,
                          color: spanHasError
                            ? theme.colors.actionDangerPrimaryBackgroundDefault
                            : tooltipTextSecondary,
                          backgroundColor: spanHasError
                            ? theme.isDarkMode
                              ? 'rgba(239, 68, 68, 0.3)'
                              : 'rgba(239, 68, 68, 0.2)'
                            : tooltipBadgeBg,
                          padding: '2px 6px',
                          borderRadius: 8,
                          fontWeight: 600,
                        }}
                      >
                        #{index + 1}
                      </span>
                      <span
                        css={{
                          fontSize: theme.typography.fontSizeSm,
                          color: spanHasError ? theme.colors.actionDangerPrimaryBackgroundDefault : tooltipTextPrimary,
                        }}
                      >
                        {truncateText(String(span.title ?? displayName), 18)}
                      </span>
                    </div>
                    <span css={{ fontSize: theme.typography.fontSizeSm, color: tooltipTextSecondary }}>{duration}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      <div
        css={{
          marginTop: theme.spacing.sm,
          paddingTop: theme.spacing.sm,
          borderTop: `1px solid ${tooltipBorder}`,
          fontSize: theme.typography.fontSizeSm,
          color: tooltipTextSecondary,
        }}
      >
        Click node to highlight • Click a span above for details
      </div>
    </div>
  );
};

interface WorkflowNodeProps {
  data: WorkflowNodeData;
  selected?: boolean;
}

/**
 * React Flow custom node for rendering aggregated workflow nodes.
 * Shows node type icon, display name, and execution count badge.
 */
export const WorkflowNode = memo(function WorkflowNode({ data }: WorkflowNodeProps) {
  const { theme } = useDesignSystemTheme();
  const { displayName, nodeType, count, spans, isSelected, isOnHighlightedPath, onSelect, onViewSpanDetails } = data;

  const spanType = nodeType as ModelSpanType | undefined;

  // Check if ANY span in this aggregated node has an exception
  const errorCount = useMemo(() => spans.reduce((acc, span) => acc + getSpanExceptionCount(span), 0), [spans]);
  const hasException = errorCount > 0;

  const backgroundColor = hasException
    ? theme.isDarkMode
      ? theme.colors.red800
      : theme.colors.red100
    : getNodeBackgroundColor(spanType, theme);

  const isHighlighted = isSelected || isOnHighlightedPath;
  const borderColor = hasException
    ? theme.colors.actionDangerPrimaryBackgroundDefault
    : isHighlighted
      ? theme.colors.actionPrimaryBackgroundDefault
      : getNodeBorderColor(spanType, theme);
  const borderWidth = hasException || isHighlighted ? 2 : 1;

  const iconType = getIconTypeForSpan(spanType ?? 'UNKNOWN');
  const title = truncateText(displayName, 16);

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onSelect();
    },
    [onSelect],
  );

  return (
    <Tooltip
      componentId="shared.model-trace-explorer.workflow-node-tooltip"
      content={
        <WorkflowNodeTooltipContent
          displayName={displayName}
          nodeType={nodeType}
          count={count}
          spans={spans}
          onViewDetails={onViewSpanDetails}
        />
      }
      side="right"
      maxWidth={350}
    >
      <div
        onClick={handleClick}
        css={{
          width: 160,
          height: 56,
          backgroundColor,
          border: `${borderWidth}px solid ${borderColor}`,
          borderRadius: theme.borders.borderRadiusMd,
          display: 'flex',
          alignItems: 'center',
          padding: `0 ${theme.spacing.sm}px`,
          cursor: 'pointer',
          position: 'relative',
          '&:hover': { filter: 'brightness(0.95)' },
        }}
      >
        {/* Connection handles */}
        <Handle type="target" position={Position.Top} css={{ visibility: 'hidden' }} />
        <Handle type="source" position={Position.Bottom} css={{ visibility: 'hidden' }} />

        {/* Icon */}
        <div
          css={{
            width: 24,
            height: 24,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginRight: theme.spacing.sm,
          }}
        >
          <ModelTraceExplorerIcon type={iconType} hasException={hasException} />
        </div>

        {/* Title */}
        <span
          css={{
            fontSize: theme.typography.fontSizeMd,
            fontWeight: theme.typography.typographyBoldFontWeight,
            color: hasException ? theme.colors.actionDangerPrimaryBackgroundDefault : theme.colors.textPrimary,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {title}
        </span>

        {/* Error badge */}
        {hasException && (
          <div
            css={{
              position: 'absolute',
              top: -10,
              left: -10,
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

        {/* Count badge */}
        {count > 1 && (
          <div
            css={{
              position: 'absolute',
              top: -12,
              right: -12,
              width: 24,
              height: 24,
              borderRadius: '50%',
              backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 11,
              fontWeight: 600,
              color: theme.colors.white,
            }}
          >
            {count > 99 ? '99+' : count}
          </div>
        )}

        {/* Selection indicator */}
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

WorkflowNode.displayName = 'WorkflowNode';

// Export the node types map for React Flow
export const workflowNodeTypes = {
  workflowNode: WorkflowNode,
};
