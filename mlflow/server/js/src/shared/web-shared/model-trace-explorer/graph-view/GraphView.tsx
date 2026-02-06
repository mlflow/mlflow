import { useCallback, useMemo, useRef, useState } from 'react';

import { Empty, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useResizeObserver } from '@databricks/web-shared/hooks';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import type { GraphViewMode, WorkflowNode } from './GraphView.types';
import { DEFAULT_GRAPH_LAYOUT_CONFIG, DEFAULT_WORKFLOW_LAYOUT_CONFIG } from './GraphView.types';
import { computeGraphLayout } from './GraphView.utils';
import { computeWorkflowLayout } from './GraphView.workflow';
import { hasAnyGraphNodeAttributes } from './GraphView.filters';
import { GraphViewCanvas } from './GraphViewCanvas';
import { GraphViewWorkflowCanvas } from './GraphViewWorkflowCanvas';
import { GraphViewControls } from './GraphViewControls';
import { GraphViewSidebar } from './GraphViewSidebar';

interface GraphViewProps {
  className?: string;
}

/**
 * Computes the path from a node to the root, returning sets of node IDs and edge IDs.
 */
function computePathToRoot(
  nodeId: string | undefined,
  layout: ReturnType<typeof computeGraphLayout>,
): { nodeIds: Set<string>; edgeIds: Set<string> } {
  const nodeIds = new Set<string>();
  const edgeIds = new Set<string>();

  if (!nodeId) {
    return { nodeIds, edgeIds };
  }

  const nodeMap = new Map(layout.nodes.map((n) => [n.id, n]));
  let currentId: string | undefined = nodeId;

  while (currentId) {
    nodeIds.add(currentId);
    const currentNode = nodeMap.get(currentId);
    if (!currentNode) break;

    const parentId = currentNode.spanNode.parentId;
    if (parentId) {
      edgeIds.add(`${parentId}-${currentId}`);
      currentId = parentId;
    } else {
      break;
    }
  }

  return { nodeIds, edgeIds };
}

/**
 * Computes the path from a workflow node to root nodes.
 */
function computeWorkflowPathToRoot(
  nodeId: string | null,
  layout: ReturnType<typeof computeWorkflowLayout> | null,
): { nodeIds: Set<string>; edgeIds: Set<string> } {
  const nodeIds = new Set<string>();
  const edgeIds = new Set<string>();

  if (!nodeId || !layout || layout.nodes.length === 0) {
    return { nodeIds, edgeIds };
  }

  nodeIds.add(nodeId);

  const incomingEdgesMap = new Map<string, typeof layout.edges>();
  for (const edge of layout.edges) {
    if (!incomingEdgesMap.has(edge.targetId)) {
      incomingEdgesMap.set(edge.targetId, []);
    }
    incomingEdgesMap.get(edge.targetId)!.push(edge);
  }

  const toProcess = [nodeId];
  const visited = new Set<string>([nodeId]);

  while (toProcess.length > 0) {
    const currentNodeId = toProcess.shift()!;
    const incomingEdges = incomingEdgesMap.get(currentNodeId) || [];

    for (const edge of incomingEdges) {
      const parentId = edge.sourceId;
      edgeIds.add(`${edge.sourceId}->${edge.targetId}`);

      if (!visited.has(parentId)) {
        visited.add(parentId);
        nodeIds.add(parentId);
        toProcess.push(parentId);
      }
    }
  }

  return { nodeIds, edgeIds };
}

/**
 * Main Graph View component that visualizes trace spans as a directed graph.
 * Uses React Flow for rendering with zoom, pan, and minimap functionality.
 */
export const GraphView = ({ className }: GraphViewProps) => {
  const { theme } = useDesignSystemTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const size = useResizeObserver({ ref: containerRef });

  const { rootNode, selectedNode, setSelectedNode } = useModelTraceExplorerViewState();

  // Sidebar span state
  const [sidebarSpan, setSidebarSpan] = useState<ModelTraceSpanNode | undefined>(undefined);

  // View mode state
  const [viewMode, setViewMode] = useState<GraphViewMode>('all_spans');

  // Selected workflow node state
  const [selectedWorkflowNode, setSelectedWorkflowNode] = useState<WorkflowNode | null>(null);

  // Check if the trace has any graph node attributes
  const hasLogicalWorkflow = useMemo(() => (rootNode ? hasAnyGraphNodeAttributes(rootNode) : false), [rootNode]);

  // Compute span tree layout
  const spanLayout = useMemo(() => computeGraphLayout(rootNode, DEFAULT_GRAPH_LAYOUT_CONFIG), [rootNode]);

  // Compute workflow layout
  const workflowLayout = useMemo(
    () => (hasLogicalWorkflow ? computeWorkflowLayout(rootNode, DEFAULT_WORKFLOW_LAYOUT_CONFIG) : null),
    [rootNode, hasLogicalWorkflow],
  );

  // Determine which layout to use
  const isWorkflowMode = viewMode === 'logical_workflow' && hasLogicalWorkflow;
  const currentLayout = isWorkflowMode && workflowLayout ? workflowLayout : spanLayout;

  // Compute highlighted paths
  const { nodeIds: highlightedPathNodeIds, edgeIds: highlightedPathEdgeIds } = useMemo(
    () => computePathToRoot(selectedNode?.key !== undefined ? String(selectedNode.key) : undefined, spanLayout),
    [selectedNode, spanLayout],
  );

  const { nodeIds: highlightedWorkflowNodeIds, edgeIds: highlightedWorkflowEdgeIds } = useMemo(
    () => computeWorkflowPathToRoot(selectedWorkflowNode?.id ?? null, workflowLayout),
    [selectedWorkflowNode, workflowLayout],
  );

  // Handlers
  const handleSelectWorkflowNode = useCallback((node: WorkflowNode | null) => {
    setSelectedWorkflowNode(node);
  }, []);

  const handleSelectNode = useCallback(
    (node: ModelTraceSpanNode | undefined) => {
      setSelectedNode(node);
    },
    [setSelectedNode],
  );

  const handleViewDetails = useCallback((node: ModelTraceSpanNode) => {
    setSidebarSpan(node);
  }, []);

  const handleCloseSidebar = useCallback(() => {
    setSidebarSpan(undefined);
  }, []);

  // Empty state when no trace data
  if (!rootNode) {
    return (
      <div
        ref={containerRef}
        className={className}
        css={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
          alignItems: 'center',
          justifyContent: 'center',
          padding: theme.spacing.lg,
        }}
      >
        <Empty
          description={
            <FormattedMessage
              defaultMessage="No trace data available"
              description="Empty state message when there is no trace data to display in graph view"
            />
          }
        />
      </div>
    );
  }

  // Empty state when layout has no nodes
  if (currentLayout.nodes.length === 0) {
    return (
      <div
        ref={containerRef}
        className={className}
        css={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
          alignItems: 'center',
          justifyContent: 'center',
          padding: theme.spacing.lg,
        }}
      >
        <Empty
          description={
            <FormattedMessage
              defaultMessage="Unable to generate graph layout"
              description="Empty state message when graph layout computation fails"
            />
          }
        />
      </div>
    );
  }

  return (
    <div
      className={className}
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
          borderBottom: `1px solid ${theme.colors.border}`,
          backgroundColor: theme.colors.backgroundSecondary,
          flexShrink: 0,
        }}
      >
        <Typography.Text size="sm" color="secondary">
          {isWorkflowMode ? (
            <FormattedMessage
              defaultMessage="{count} {count, plural, one {node} other {nodes}}"
              description="Count of workflow nodes displayed in graph view"
              values={{ count: currentLayout.nodes.length }}
            />
          ) : (
            <FormattedMessage
              defaultMessage="{count} {count, plural, one {span} other {spans}}"
              description="Count of spans displayed in graph view"
              values={{ count: currentLayout.nodes.length }}
            />
          )}
        </Typography.Text>
        <Typography.Text size="sm" color="secondary">
          <FormattedMessage defaultMessage="Scroll to zoom, drag to pan" description="Navigation hint for graph view" />
        </Typography.Text>
      </div>

      {/* Main content area */}
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          flex: 1,
          minHeight: 0,
          overflow: 'hidden',
        }}
      >
        {/* Graph canvas container */}
        <div
          ref={containerRef}
          css={{
            display: 'flex',
            flexDirection: 'column',
            flex: 1,
            minHeight: 0,
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {/* Main canvas - switches between span tree and workflow graph */}
          {isWorkflowMode && workflowLayout ? (
            <GraphViewWorkflowCanvas
              layout={workflowLayout}
              selectedNodeId={selectedWorkflowNode?.id ?? null}
              highlightedPathNodeIds={highlightedWorkflowNodeIds}
              highlightedPathEdgeIds={highlightedWorkflowEdgeIds}
              onSelectNode={handleSelectWorkflowNode}
              onViewSpanDetails={handleViewDetails}
            />
          ) : (
            <GraphViewCanvas
              layout={spanLayout}
              selectedNodeKey={selectedNode?.key}
              highlightedPathNodeIds={highlightedPathNodeIds}
              highlightedPathEdgeIds={highlightedPathEdgeIds}
              onSelectNode={handleSelectNode}
              onViewDetails={handleViewDetails}
            />
          )}

          {/* View mode toggle controls */}
          <GraphViewControls viewMode={viewMode} setViewMode={setViewMode} hasLogicalWorkflow={hasLogicalWorkflow} />
        </div>

        {/* Sidebar */}
        {sidebarSpan && <GraphViewSidebar span={sidebarSpan} onClose={handleCloseSidebar} />}
      </div>
    </div>
  );
};
