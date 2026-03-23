import { memo, useMemo } from 'react';

import { EdgeLabelRenderer, getBezierPath, type Position } from '@xyflow/react';
import { useDesignSystemTheme } from '@databricks/design-system';

import type { WorkflowEdgeData } from './GraphView.types';

// Nested call edge colors — no semantic theme token exists for this.
// Centralized here so dark/light mode values are easy to update.
const NESTED_CALL_COLOR = { dark: 'rgba(249, 115, 22, 0.6)', light: 'rgba(234, 88, 12, 0.5)' };
const NESTED_CALL_HIGHLIGHT_COLOR = { dark: 'rgba(249, 115, 22, 1)', light: 'rgba(234, 88, 12, 1)' };
const CONDITIONAL_EDGE_COLOR = { dark: 'rgba(45, 212, 191, 0.6)', light: 'rgba(13, 148, 136, 0.5)' };
const CONDITIONAL_EDGE_HIGHLIGHT_COLOR = { dark: 'rgba(45, 212, 191, 1)', light: 'rgba(13, 148, 136, 1)' };
const UNEXECUTED_EDGE_COLOR = { dark: 'rgba(255, 255, 255, 0.12)', light: 'rgba(0, 0, 0, 0.08)' };

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
  const isConditional = data?.isConditional ?? false;
  const isExecuted = data?.isExecuted ?? true;
  const conditionLabel = data?.conditionLabel;
  const stepSequence = data?.stepSequence;
  const isReturnEdge = data?.isReturnEdge ?? false;

  // Calculate path based on edge type
  const edgePath = useMemo(() => {
    if (isBackEdge) {
      const deltaX = sourceX - targetX;
      const deltaY = Math.abs(targetY - sourceY);
      const isShortLoop = deltaY < 250;

      if (isShortLoop) {
        // Short loop (adjacent layers, e.g. tool→agent): compact curve that
        // fans out from the source's horizontal position. The curve bows outward
        // on whichever side the source sits relative to the target, so multiple
        // back-edges spread naturally instead of stacking.
        const sign = deltaX >= 0 ? 1 : -1;
        const bowOut = sign * Math.max(30, Math.abs(deltaX) * 0.4 + 20);
        return `M ${sourceX} ${sourceY}
                C ${sourceX + bowOut} ${sourceY - deltaY * 0.3},
                  ${targetX + bowOut} ${targetY + deltaY * 0.3},
                  ${targetX} ${targetY}`;
      }

      // Long back-edge (multiple layers): wider arc on the right
      const offset = Math.max(80, deltaY * 0.5);
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

    // Upward-flowing edge (return path, e.g. tool→agent): curve to the RIGHT
    // side so it's clearly separated from the forward (downward) edge between
    // the same pair of nodes.
    if (sourceY > targetY) {
      const deltaY = Math.abs(targetY - sourceY);
      const bowRight = Math.max(60, deltaY * 0.4);
      return `M ${sourceX} ${sourceY}
              C ${sourceX + bowRight} ${sourceY - deltaY * 0.3},
                ${targetX + bowRight} ${targetY + deltaY * 0.3},
                ${targetX} ${targetY}`;
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
  const defaultEdgeColor = theme.isDarkMode ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.2)';
  const strokeColor = useMemo(() => {
    if (!isExecuted) {
      return theme.isDarkMode ? UNEXECUTED_EDGE_COLOR.dark : UNEXECUTED_EDGE_COLOR.light;
    }
    if (isBackEdge) {
      return theme.colors.actionDangerPrimaryBackgroundDefault;
    }
    if (isNestedCall) {
      const palette = isHighlighted ? NESTED_CALL_HIGHLIGHT_COLOR : NESTED_CALL_COLOR;
      return theme.isDarkMode ? palette.dark : palette.light;
    }
    return isHighlighted ? theme.colors.actionPrimaryBackgroundDefault : defaultEdgeColor;
  }, [isBackEdge, isNestedCall, isHighlighted, isExecuted, theme, defaultEdgeColor]);

  const strokeWidth = !isExecuted ? 1 : isBackEdge || isHighlighted ? 2.5 : isNestedCall ? 2 : 1.5;
  const strokeDasharray = !isExecuted
    ? '4 4'
    : isBackEdge
      ? '6 3'
      : isNestedCall
        ? '5 3'
        : isConditional
          ? '6 3'
          : undefined;

  // Calculate label position — for back-edges, place near the curve's bow-out
  const labelPos = useMemo(() => {
    if (isBackEdge) {
      const deltaX = sourceX - targetX;
      const sign = deltaX >= 0 ? 1 : -1;
      const bowOut = sign * Math.max(30, Math.abs(deltaX) * 0.4 + 20);
      return { x: (sourceX + targetX) / 2 + bowOut * 0.6, y: (sourceY + targetY) / 2 };
    }
    if (isNestedCall) {
      return { x: (sourceX + targetX) / 2 - 45, y: (sourceY + targetY) / 2 };
    }
    // For upward-flowing edges (return paths), place label along the right-side curve
    const isUpward = sourceY > targetY;
    if (isUpward) {
      const deltaY = Math.abs(targetY - sourceY);
      const bowRight = Math.max(60, deltaY * 0.4);
      return { x: (sourceX + targetX) / 2 + bowRight * 0.6, y: (sourceY + targetY) / 2 };
    }
    return { x: (sourceX + targetX) / 2, y: (sourceY + targetY) / 2 };
  }, [sourceX, sourceY, targetX, targetY, isBackEdge, isNestedCall]);
  const labelX = labelPos.x;
  const labelY = labelPos.y;

  return (
    <>
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        strokeDasharray={strokeDasharray}
        markerEnd={`url(#workflow-arrow-${!isExecuted ? 'unexecuted' : isBackEdge ? 'back' : isNestedCall ? 'nested' : isHighlighted ? 'highlighted' : 'default'})`}
      />

      {/* Step sequence label on edges — shows execution order */}
      {stepSequence && stepSequence.length > 0 && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: 'all',
              zIndex: 10,
            }}
          >
            <div
              css={{
                backgroundColor: theme.colors.backgroundPrimary,
                border: `1px solid ${strokeColor}`,
                borderRadius: 10,
                padding: '1px 6px',
                fontSize: 10,
                fontWeight: 600,
                color: strokeColor,
                whiteSpace: 'nowrap',
              }}
            >
              {stepSequence.join(', ')}
            </div>
          </div>
        </EdgeLabelRenderer>
      )}

      {/* Edge count label (show if count > 1 and no step sequence) */}
      {count > 1 && !stepSequence && (
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
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY + (stepSequence || count > 1 ? 18 : 0)}px)`,
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

      {/* Label for conditional edges */}
      {isConditional && conditionLabel && isExecuted && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY + (stepSequence || count > 1 ? 18 : 0)}px)`,
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
              {conditionLabel}
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
  const nestedCallColor = theme.isDarkMode ? NESTED_CALL_COLOR.dark : NESTED_CALL_COLOR.light;
  const conditionalColor = theme.isDarkMode ? CONDITIONAL_EDGE_COLOR.dark : CONDITIONAL_EDGE_COLOR.light;
  const unexecutedColor = theme.isDarkMode ? UNEXECUTED_EDGE_COLOR.dark : UNEXECUTED_EDGE_COLOR.light;

  return (
    <svg style={{ position: 'absolute', top: 0, left: 0 }}>
      <defs>
        <marker
          id="workflow-arrow-default"
          markerWidth="7"
          markerHeight="5"
          refX="6"
          refY="2.5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <polygon points="0 0, 7 2.5, 0 5" fill={defaultArrowColor} />
        </marker>
        <marker
          id="workflow-arrow-highlighted"
          markerWidth="7"
          markerHeight="5"
          refX="6"
          refY="2.5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <polygon points="0 0, 7 2.5, 0 5" fill={theme.colors.actionPrimaryBackgroundDefault} />
        </marker>
        <marker
          id="workflow-arrow-back"
          markerWidth="7"
          markerHeight="5"
          refX="6"
          refY="2.5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <polygon points="0 0, 7 2.5, 0 5" fill={theme.colors.actionDangerPrimaryBackgroundDefault} />
        </marker>
        <marker
          id="workflow-arrow-nested"
          markerWidth="7"
          markerHeight="5"
          refX="6"
          refY="2.5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <polygon points="0 0, 7 2.5, 0 5" fill={nestedCallColor} />
        </marker>
        <marker
          id="workflow-arrow-conditional"
          markerWidth="7"
          markerHeight="5"
          refX="6"
          refY="2.5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <polygon points="0 0, 7 2.5, 0 5" fill={conditionalColor} />
        </marker>
        <marker
          id="workflow-arrow-unexecuted"
          markerWidth="7"
          markerHeight="5"
          refX="6"
          refY="2.5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <polygon points="0 0, 7 2.5, 0 5" fill={unexecutedColor} />
        </marker>
      </defs>
    </svg>
  );
};

// Export the edge types map for React Flow
export const workflowEdgeTypes = {
  workflowEdge: WorkflowEdge,
};
