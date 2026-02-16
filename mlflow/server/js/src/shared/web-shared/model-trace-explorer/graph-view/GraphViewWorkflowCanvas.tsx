import { useCallback, useEffect, useMemo, useRef } from 'react';

import { ReactFlow, useReactFlow, useNodesInitialized, ReactFlowProvider } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useDesignSystemTheme } from '@databricks/design-system';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { WorkflowLayout, WorkflowNode, WorkflowFlowNode, WorkflowFlowEdge } from './GraphView.types';
import { workflowNodeTypes } from './GraphViewWorkflowNode';
import { WorkflowEdgeMarkerDefs, workflowEdgeTypes } from './GraphViewWorkflowEdge';

interface GraphViewWorkflowCanvasProps {
  layout: WorkflowLayout;
  selectedNodeId: string | null;
  highlightedPathNodeIds: Set<string>;
  highlightedPathEdgeIds: Set<string>;
  onSelectNode: (node: WorkflowNode | null) => void;
  onViewSpanDetails: (span: ModelTraceSpanNode) => void;
}

// Inner component that uses React Flow hooks
const GraphViewWorkflowCanvasInner = ({
  layout,
  selectedNodeId,
  highlightedPathNodeIds,
  highlightedPathEdgeIds,
  onSelectNode,
  onViewSpanDetails,
}: GraphViewWorkflowCanvasProps) => {
  const { theme } = useDesignSystemTheme();
  const { fitView } = useReactFlow();

  // Create a map from node ID to node for callbacks
  const nodeMap = useMemo(() => {
    const map = new Map<string, WorkflowNode>();
    for (const node of layout.nodes) {
      map.set(node.id, node);
    }
    return map;
  }, [layout.nodes]);

  // Stabilize onViewSpanDetails via ref to avoid recreating all node data
  const onViewSpanDetailsRef = useRef(onViewSpanDetails);
  onViewSpanDetailsRef.current = onViewSpanDetails;
  const stableOnViewSpanDetails = useCallback((span: ModelTraceSpanNode) => {
    onViewSpanDetailsRef.current(span);
  }, []);

  // Convert layout nodes to React Flow nodes
  const nodes: WorkflowFlowNode[] = useMemo(() => {
    return layout.nodes.map(
      (node): WorkflowFlowNode => ({
        id: node.id,
        type: 'workflowNode',
        position: { x: node.x, y: node.y },
        data: {
          displayName: node.displayName,
          nodeType: node.nodeType,
          count: node.count,
          spans: node.spans,
          isSelected: node.id === selectedNodeId,
          isOnHighlightedPath: highlightedPathNodeIds.has(node.id),
          onViewSpanDetails: stableOnViewSpanDetails,
        },
      }),
    );
  }, [layout.nodes, selectedNodeId, highlightedPathNodeIds, stableOnViewSpanDetails]);

  // Convert layout edges to React Flow edges
  const edges: WorkflowFlowEdge[] = useMemo(() => {
    return layout.edges.map((edge): WorkflowFlowEdge => {
      const edgeId = `${edge.sourceId}->${edge.targetId}`;
      return {
        id: edgeId,
        source: edge.sourceId,
        target: edge.targetId,
        type: 'workflowEdge',
        data: {
          count: edge.count,
          isBackEdge: edge.isBackEdge,
          isNestedCall: edge.isNestedCall ?? false,
          isHighlighted: highlightedPathEdgeIds.has(edgeId),
        },
      };
    });
  }, [layout.edges, highlightedPathEdgeIds]);

  // Re-center the view when nodes are initialized or layout changes
  const nodesInitialized = useNodesInitialized();
  useEffect(() => {
    if (nodesInitialized && layout.nodes.length > 0) {
      fitView({ padding: 0.15, duration: 200 });
    }
  }, [nodesInitialized, layout.nodes.length, layout.width, layout.height, fitView]);

  // Handle node click for selection (replaces per-node onSelect closures)
  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: WorkflowFlowNode) => {
      const workflowNode = nodeMap.get(node.id) ?? null;
      onSelectNode(workflowNode);
    },
    [nodeMap, onSelectNode],
  );

  // Handle pane click to deselect
  const handlePaneClick = useCallback(() => {
    onSelectNode(null);
  }, [onSelectNode]);

  return (
    <div css={{ flex: 1, width: '100%', height: '100%', position: 'relative' }}>
      {/* SVG marker definitions for edge arrows */}
      <WorkflowEdgeMarkerDefs />

      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={workflowNodeTypes}
        edgeTypes={workflowEdgeTypes}
        onNodeClick={handleNodeClick}
        onPaneClick={handlePaneClick}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.1}
        maxZoom={3}
        defaultEdgeOptions={{
          type: 'workflowEdge',
        }}
        proOptions={{ hideAttribution: true }}
        style={{
          backgroundColor: theme.colors.backgroundPrimary,
        }}
      />
    </div>
  );
};

/**
 * React Flow canvas for rendering the aggregated workflow graph.
 * Provides zoom and pan functionality for workflow visualization.
 */
export const GraphViewWorkflowCanvas = (props: GraphViewWorkflowCanvasProps) => {
  return (
    <ReactFlowProvider>
      <GraphViewWorkflowCanvasInner {...props} />
    </ReactFlowProvider>
  );
};
