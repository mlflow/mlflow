import { memo } from 'react';

import { BaseEdge, getBezierPath, type Position } from '@xyflow/react';
import { useDesignSystemTheme } from '@databricks/design-system';

import type { SpanEdgeData } from './GraphView.types';

interface SpanEdgeProps {
  id: string;
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  sourcePosition: Position;
  targetPosition: Position;
  data?: SpanEdgeData;
  markerEnd?: string;
}

/**
 * React Flow custom edge for span tree view.
 * Renders a bezier curve with optional highlighting.
 */
export const SpanEdge = memo(function SpanEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  markerEnd,
}: SpanEdgeProps) {
  const { theme } = useDesignSystemTheme();
  const isHighlighted = data?.isHighlighted ?? false;

  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const strokeColor = isHighlighted ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.borderDecorative;
  const strokeWidth = isHighlighted ? 2 : 1;

  return (
    <BaseEdge
      id={id}
      path={edgePath}
      style={{
        stroke: strokeColor,
        strokeWidth,
      }}
      markerEnd={markerEnd}
    />
  );
});

SpanEdge.displayName = 'SpanEdge';

// Export the edge types map for React Flow
export const spanEdgeTypes = {
  spanEdge: SpanEdge,
};
