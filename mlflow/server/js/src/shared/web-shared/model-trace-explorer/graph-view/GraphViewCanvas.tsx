import { useCallback, useEffect, useMemo } from 'react';

import { ReactFlow, useReactFlow, ReactFlowProvider, MiniMap, type Node } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useDesignSystemTheme } from '@databricks/design-system';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { GraphLayout, SpanNodeData, SpanFlowNode, SpanFlowEdge } from './GraphView.types';
import { spanNodeTypes } from './GraphViewNode';
import { spanEdgeTypes } from './GraphViewEdge';
import { getNodeBackgroundColor } from './GraphView.utils';

interface GraphViewCanvasProps {
  layout: GraphLayout;
  selectedNodeKey?: string | number;
  highlightedPathNodeIds: Set<string>;
  highlightedPathEdgeIds: Set<string>;
  onSelectNode: (node: ModelTraceSpanNode | undefined) => void;
  onViewDetails: (node: ModelTraceSpanNode) => void;
}

// Inner component that uses React Flow hooks
const GraphViewCanvasInner = ({
  layout,
  selectedNodeKey,
  highlightedPathNodeIds,
  highlightedPathEdgeIds,
  onSelectNode,
  onViewDetails,
}: GraphViewCanvasProps) => {
  const { theme } = useDesignSystemTheme();
  const { fitView } = useReactFlow();

  const selectedNodeId = selectedNodeKey !== null && selectedNodeKey !== undefined ? String(selectedNodeKey) : null;

  // Convert layout nodes to React Flow nodes
  const nodes: SpanFlowNode[] = useMemo(() => {
    return layout.nodes.map(
      (node): SpanFlowNode => ({
        id: node.id,
        type: 'spanNode',
        position: { x: node.x, y: node.y },
        // Width and height are needed for MiniMap to render nodes correctly
        width: node.width,
        height: node.height,
        data: {
          spanNode: node.spanNode,
          label: String(node.spanNode.title ?? ''),
          isSelected: node.id === selectedNodeId,
          isOnHighlightedPath: highlightedPathNodeIds.has(node.id),
          onSelect: () => onSelectNode(node.spanNode),
          onViewDetails: () => onViewDetails(node.spanNode),
        },
      }),
    );
  }, [layout.nodes, selectedNodeId, highlightedPathNodeIds, onSelectNode, onViewDetails]);

  // Convert layout edges to React Flow edges
  const edges: SpanFlowEdge[] = useMemo(() => {
    return layout.edges.map((edge): SpanFlowEdge => {
      const edgeId = `${edge.sourceId}-${edge.targetId}`;
      return {
        id: edgeId,
        source: edge.sourceId,
        target: edge.targetId,
        type: 'spanEdge',
        data: {
          isHighlighted: highlightedPathEdgeIds.has(edgeId),
        },
      };
    });
  }, [layout.edges, highlightedPathEdgeIds]);

  // Fit view when layout changes
  useEffect(() => {
    if (layout.nodes.length > 0) {
      // Small delay to ensure nodes are rendered
      const timeout = setTimeout(() => {
        fitView({ padding: 0.1, duration: 200 });
      }, 50);
      return () => clearTimeout(timeout);
    }
    return undefined;
  }, [layout.nodes.length, layout.width, layout.height, fitView]);

  // Handle pane click to deselect
  const handlePaneClick = useCallback(() => {
    onSelectNode(undefined);
  }, [onSelectNode]);

  // MiniMap node color function
  const miniMapNodeColor = useCallback(
    (node: Node) => {
      const spanNode = (node.data as SpanNodeData)?.spanNode;
      if (spanNode) {
        return getNodeBackgroundColor(spanNode.type, theme);
      }
      return theme.colors.backgroundSecondary;
    },
    [theme],
  );

  return (
    <div css={{ flex: 1, width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={spanNodeTypes}
        edgeTypes={spanEdgeTypes}
        onPaneClick={handlePaneClick}
        fitView
        fitViewOptions={{ padding: 0.1 }}
        minZoom={0.1}
        maxZoom={3}
        defaultEdgeOptions={{
          type: 'spanEdge',
        }}
        proOptions={{ hideAttribution: true }}
        style={{
          backgroundColor: theme.colors.backgroundPrimary,
        }}
      >
        <MiniMap
          nodeColor={miniMapNodeColor}
          nodeStrokeWidth={1}
          nodeBorderRadius={4}
          zoomable
          pannable
          maskColor={theme.isDarkMode ? 'rgba(0, 0, 0, 0.7)' : 'rgba(255, 255, 255, 0.7)'}
          style={{
            backgroundColor: theme.isDarkMode ? 'rgba(0, 0, 0, 0.5)' : 'rgba(255, 255, 255, 0.8)',
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
          }}
        />
      </ReactFlow>
    </div>
  );
};

/**
 * React Flow canvas for rendering the span tree graph.
 * Provides zoom, pan, and minimap functionality.
 */
export const GraphViewCanvas = (props: GraphViewCanvasProps) => {
  return (
    <ReactFlowProvider>
      <GraphViewCanvasInner {...props} />
    </ReactFlowProvider>
  );
};
