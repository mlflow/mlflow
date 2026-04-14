import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  ReactFlow,
  useReactFlow,
  useNodesInitialized,
  ReactFlowProvider,
  Panel,
  applyNodeChanges,
  type OnNodesChange,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useDesignSystemTheme } from '@databricks/design-system';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { WorkflowLayout, WorkflowNode, WorkflowFlowNode, WorkflowFlowEdge } from './GraphView.types';
import { workflowNodeTypes } from './GraphViewWorkflowNode';
import { WorkflowEdgeMarkerDefs, workflowEdgeTypes } from './GraphViewWorkflowEdge';
import { GraphViewFloatingToolbar } from './GraphViewFloatingToolbar';

interface GraphViewWorkflowCanvasProps {
  layout: WorkflowLayout;
  selectedNodeId: string | null;
  highlightedPathNodeIds: Set<string>;
  highlightedPathEdgeIds: Set<string>;
  onSelectNode: (node: WorkflowNode | null) => void;
  onViewSpanDetails: (span: ModelTraceSpanNode) => void;
  isGraphExpanded: boolean;
  onToggleGraphExpand: () => void;
}

// Inner component that uses React Flow hooks
const GraphViewWorkflowCanvasInner = ({
  layout,
  selectedNodeId,
  highlightedPathNodeIds,
  highlightedPathEdgeIds,
  onSelectNode,
  onViewSpanDetails,
  isGraphExpanded,
  onToggleGraphExpand,
}: GraphViewWorkflowCanvasProps) => {
  const { theme } = useDesignSystemTheme();
  const { fitView, getZoom, setCenter, zoomIn, zoomOut } = useReactFlow();

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

  // Build React Flow nodes from layout, stored in state to support dragging
  const buildFlowNodes = useCallback((): WorkflowFlowNode[] => {
    return layout.nodes.map(
      (node): WorkflowFlowNode => ({
        id: node.id,
        type: 'workflowNode',
        position: { x: node.x, y: node.y },
        draggable: true,
        data: {
          displayName: node.displayName,
          nodeType: node.nodeType,
          count: node.count,
          spans: node.spans,
          isSelected: node.id === selectedNodeId,
          isOnHighlightedPath: highlightedPathNodeIds.has(node.id),
          onViewSpanDetails: stableOnViewSpanDetails,
          nodeWidth: node.width,
          nodeHeight: node.height,
        },
      }),
    );
  }, [layout.nodes, selectedNodeId, highlightedPathNodeIds, stableOnViewSpanDetails]);

  const [nodes, setNodes] = useState<WorkflowFlowNode[]>(buildFlowNodes);

  // Update nodes when layout or selection changes, preserving user-dragged positions
  const prevLayoutRef = useRef(layout);
  useEffect(() => {
    const layoutChanged = prevLayoutRef.current !== layout;
    prevLayoutRef.current = layout;

    if (layoutChanged) {
      setNodes(buildFlowNodes());
    } else {
      setNodes((prev) =>
        prev.map((flowNode) => {
          const layoutNode = nodeMap.get(flowNode.id);
          if (!layoutNode) return flowNode;
          return {
            ...flowNode,
            data: {
              ...flowNode.data,
              isSelected: flowNode.id === selectedNodeId,
              isOnHighlightedPath: highlightedPathNodeIds.has(flowNode.id),
            },
          };
        }),
      );
    }
  }, [layout, selectedNodeId, highlightedPathNodeIds, buildFlowNodes, nodeMap]);

  // Handle node changes (drag, select, etc.) to persist dragged positions
  const onNodesChange: OnNodesChange<WorkflowFlowNode> = useCallback((changes) => {
    setNodes((nds) => applyNodeChanges(changes, nds));
  }, []);

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
      fitView({ padding: 0.2, duration: 200 });
    }
  }, [nodesInitialized, layout.nodes.length, layout.width, layout.height, fitView]);

  // Track whether selection originated from a graph node click (skip panning)
  const isLocalClickRef = useRef(false);

  // Pan to selected node when selection changes from outside (e.g. tree click)
  const prevSelectedRef = useRef(selectedNodeId);
  useEffect(() => {
    if (prevSelectedRef.current === selectedNodeId) {
      return;
    }
    prevSelectedRef.current = selectedNodeId;

    if (isLocalClickRef.current) {
      isLocalClickRef.current = false;
      return;
    }

    if (!selectedNodeId) {
      return;
    }

    // Find the node's current position and center the view on it
    const flowNode = nodes.find((n) => n.id === selectedNodeId);
    if (flowNode) {
      const nodeWidth = flowNode.data.nodeWidth ?? 160;
      const nodeHeight = flowNode.data.nodeHeight ?? 56;
      setCenter(flowNode.position.x + nodeWidth / 2, flowNode.position.y + nodeHeight / 2, {
        zoom: getZoom(),
        duration: 300,
      });
    }
  }, [selectedNodeId, nodes, setCenter, getZoom]);

  // Handle node click for selection (replaces per-node onSelect closures)
  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: WorkflowFlowNode) => {
      isLocalClickRef.current = true;
      const workflowNode = nodeMap.get(node.id) ?? null;
      onSelectNode(workflowNode);
    },
    [nodeMap, onSelectNode],
  );

  // Handle pane click to deselect
  const handlePaneClick = useCallback(() => {
    isLocalClickRef.current = true;
    onSelectNode(null);
  }, [onSelectNode]);

  const handleFitView = useCallback(() => {
    fitView({ padding: 0.2, duration: 200 });
  }, [fitView]);

  const handleZoomIn = useCallback(() => {
    zoomIn({ duration: 200 });
  }, [zoomIn]);

  const handleZoomOut = useCallback(() => {
    zoomOut({ duration: 200 });
  }, [zoomOut]);

  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      css={{ flex: 1, width: '100%', height: '100%', position: 'relative' }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* SVG marker definitions for edge arrows */}
      <WorkflowEdgeMarkerDefs />

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        nodeTypes={workflowNodeTypes}
        edgeTypes={workflowEdgeTypes}
        onNodeClick={handleNodeClick}
        onPaneClick={handlePaneClick}
        nodesDraggable
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={3}
        defaultEdgeOptions={{
          type: 'workflowEdge',
        }}
        proOptions={{ hideAttribution: true }}
        style={{
          backgroundColor: theme.colors.backgroundPrimary,
        }}
      >
        <Panel
          position="bottom-right"
          css={{
            margin: 0,
            opacity: isHovered ? 1 : 0,
            transition: 'opacity 0.2s ease-in-out',
            pointerEvents: isHovered ? 'auto' : 'none',
            '&:focus-within': { opacity: 1, pointerEvents: 'auto' },
          }}
        >
          <GraphViewFloatingToolbar
            isGraphExpanded={isGraphExpanded}
            onToggleGraphExpand={onToggleGraphExpand}
            onFitView={handleFitView}
            onZoomIn={handleZoomIn}
            onZoomOut={handleZoomOut}
          />
        </Panel>
      </ReactFlow>
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
