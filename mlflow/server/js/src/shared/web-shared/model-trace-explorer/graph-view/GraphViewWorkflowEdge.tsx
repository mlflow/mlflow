import { memo, useMemo } from 'react';

import { EdgeLabelRenderer, getBezierPath, type Position } from '@xyflow/react';
import { useDesignSystemTheme } from '@databricks/design-system';

import type { WorkflowEdgeData } from './GraphView.types';

interface WorkflowEdgeProps {
  id: string;
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  sourcePosition: Position;
  targetPosition: Position;
  data?: WorkflowEdgeData;
}

/**
 * React Flow custom edge for workflow view.
 * Supports regular edges, back-edges (cycles), and nested call edges.
 */
export const WorkflowEdge = memo(function WorkflowEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: WorkflowEdgeProps) {
  const { theme } = useDesignSystemTheme();

  const count = data?.count ?? 1;
  const isBackEdge = data?.isBackEdge ?? false;
  const isNestedCall = data?.isNestedCall ?? false;
  const isHighlighted = data?.isHighlighted ?? false;

  // Calculate path based on edge type
  const edgePath = useMemo(() => {
    if (isBackEdge) {
      // Back-edge: render as a curved arc on the right side
      const offset = Math.max(80, Math.abs(targetY - sourceY) * 0.5);
      return `M ${sourceX} ${sourceY}
              C ${sourceX + offset} ${sourceY + 40},
                ${targetX + offset} ${targetY - 40},
                ${targetX} ${targetY}`;
    }

    if (isNestedCall) {
      // Nested call edge: render offset to the left
      const offsetX = -30;
      const deltaY = Math.abs(targetY - sourceY);
      const controlPointOffset = Math.min(deltaY * 0.4, 25);

      return `M ${sourceX + offsetX} ${sourceY}
              C ${sourceX + offsetX - 15} ${sourceY + controlPointOffset},
                ${targetX + offsetX - 15} ${targetY - controlPointOffset},
                ${targetX + offsetX} ${targetY}`;
    }

    // Regular forward edge - use React Flow's bezier path
    const [path] = getBezierPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
    });
    return path;
  }, [sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, isBackEdge, isNestedCall]);

  // Determine stroke color based on edge type
  const strokeColor = useMemo(() => {
    if (isBackEdge) {
      return theme.colors.actionDangerPrimaryBackgroundDefault;
    }
    if (isNestedCall) {
      return isHighlighted
        ? theme.isDarkMode
          ? '#f97316'
          : '#ea580c'
        : theme.isDarkMode
          ? 'rgba(249, 115, 22, 0.6)'
          : 'rgba(234, 88, 12, 0.5)';
    }
    return isHighlighted
      ? theme.colors.actionPrimaryBackgroundDefault
      : theme.isDarkMode
        ? 'rgba(255, 255, 255, 0.3)'
        : 'rgba(0, 0, 0, 0.2)';
  }, [isBackEdge, isNestedCall, isHighlighted, theme]);

  const strokeWidth = isBackEdge || isHighlighted ? 2.5 : isNestedCall ? 2 : 1.5;
  const strokeDasharray = isBackEdge ? '6 3' : isNestedCall ? '5 3' : undefined;

  // Calculate label position
  const labelX = isBackEdge
    ? (sourceX + targetX) / 2 + 50
    : isNestedCall
      ? (sourceX + targetX) / 2 - 45
      : (sourceX + targetX) / 2;
  const labelY = (sourceY + targetY) / 2;

  return (
    <>
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        strokeDasharray={strokeDasharray}
        markerEnd={`url(#workflow-arrow-${isBackEdge ? 'back' : isNestedCall ? 'nested' : isHighlighted ? 'highlighted' : 'default'})`}
      />

      {/* Edge count label (show if count > 1) */}
      {count > 1 && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: 'all',
            }}
          >
            <div
              css={{
                backgroundColor: theme.colors.backgroundPrimary,
                border: `1px solid ${strokeColor}`,
                borderRadius: 10,
                padding: '2px 8px',
                fontSize: 11,
                fontWeight: 500,
                color: strokeColor,
              }}
            >
              ×{count}
            </div>
          </div>
        </EdgeLabelRenderer>
      )}

      {/* Label for nested call edges */}
      {isNestedCall && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY + (count > 1 ? 18 : 0)}px)`,
              pointerEvents: 'none',
            }}
          >
            <span
              css={{
                fontSize: 9,
                fontStyle: 'italic',
                fontWeight: 500,
                color: strokeColor,
              }}
            >
              nested
            </span>
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
});

WorkflowEdge.displayName = 'WorkflowEdge';

/**
 * SVG defs for workflow edge arrow markers.
 * Should be included once in the React Flow container.
 */
export const WorkflowEdgeMarkerDefs = () => {
  const { theme } = useDesignSystemTheme();

  const defaultArrowColor = theme.isDarkMode ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.2)';
  const nestedCallColor = theme.isDarkMode ? 'rgba(249, 115, 22, 0.6)' : 'rgba(234, 88, 12, 0.5)';

  return (
    <svg style={{ position: 'absolute', top: 0, left: 0 }}>
      <defs>
        <marker
          id="workflow-arrow-default"
          markerWidth="10"
          markerHeight="7"
          refX="9"
          refY="3.5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <polygon points="0 0, 10 3.5, 0 7" fill={defaultArrowColor} />
        </marker>
        <marker
          id="workflow-arrow-highlighted"
          markerWidth="10"
          markerHeight="7"
          refX="9"
          refY="3.5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <polygon points="0 0, 10 3.5, 0 7" fill={theme.colors.actionPrimaryBackgroundDefault} />
        </marker>
        <marker
          id="workflow-arrow-back"
          markerWidth="10"
          markerHeight="7"
          refX="9"
          refY="3.5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <polygon points="0 0, 10 3.5, 0 7" fill={theme.colors.actionDangerPrimaryBackgroundDefault} />
        </marker>
        <marker
          id="workflow-arrow-nested"
          markerWidth="10"
          markerHeight="7"
          refX="9"
          refY="3.5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <polygon points="0 0, 10 3.5, 0 7" fill={nestedCallColor} />
        </marker>
      </defs>
    </svg>
  );
};

// Export the edge types map for React Flow
export const workflowEdgeTypes = {
  workflowEdge: WorkflowEdge,
};
