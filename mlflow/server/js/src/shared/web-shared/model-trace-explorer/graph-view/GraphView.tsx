import { useCallback, useMemo, useRef, useState } from 'react';

import { Button, CloseIcon, Empty, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useResizeObserver } from '@databricks/web-shared/hooks';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import ModelTraceExplorerResizablePane from '../ModelTraceExplorerResizablePane';
import { ModelTraceExplorerRightPaneTabs, RIGHT_PANE_MIN_WIDTH } from '../right-pane/ModelTraceExplorerRightPaneTabs';
import type { WorkflowNode } from './GraphView.types';
import { DEFAULT_WORKFLOW_LAYOUT_CONFIG } from './GraphView.types';
import { computeWorkflowLayout } from './GraphView.workflow';
import { GraphViewWorkflowCanvas } from './GraphViewWorkflowCanvas';

interface GraphViewProps {
  className?: string;
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
 * Graph View component that visualizes trace spans as an aggregated workflow graph.
 * Spans are grouped by name and connected based on parent-child hierarchy.
 * The right detail pane is hidden by default and opens when a span is clicked.
 */
export const GraphView = ({ className }: GraphViewProps) => {
  const { theme } = useDesignSystemTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const size = useResizeObserver({ ref: containerRef });
  const [paneWidth, setPaneWidth] = useState(500);

  const { rootNode, selectedNode, setSelectedNode, activeTab, setActiveTab, updatePaneSizeRatios, getPaneSizeRatios } =
    useModelTraceExplorerViewState();

  // Selected workflow node state
  const [selectedWorkflowNode, setSelectedWorkflowNode] = useState<WorkflowNode | null>(null);

  // Right detail pane is collapsed by default, opens on span click
  const [isDetailsPaneOpen, setIsDetailsPaneOpen] = useState(false);

  // Compute workflow layout (groups spans by name)
  const workflowLayout = useMemo(() => computeWorkflowLayout(rootNode, DEFAULT_WORKFLOW_LAYOUT_CONFIG), [rootNode]);

  // Compute highlighted paths
  const { nodeIds: highlightedWorkflowNodeIds, edgeIds: highlightedWorkflowEdgeIds } = useMemo(
    () => computeWorkflowPathToRoot(selectedWorkflowNode?.id ?? null, workflowLayout),
    [selectedWorkflowNode, workflowLayout],
  );

  const onSizeRatioChange = useCallback(
    (ratio: number) => {
      updatePaneSizeRatios({ graphPane: ratio });
    },
    [updatePaneSizeRatios],
  );

  // Handlers
  const handleSelectWorkflowNode = useCallback((node: WorkflowNode | null) => {
    setSelectedWorkflowNode(node);
  }, []);

  const handleViewSpanDetails = useCallback(
    (node: ModelTraceSpanNode) => {
      setSelectedNode(node);
      setIsDetailsPaneOpen(true);
    },
    [setSelectedNode],
  );

  const handleCloseDetailsPane = useCallback(() => {
    setIsDetailsPaneOpen(false);
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
  if (workflowLayout.nodes.length === 0) {
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

  const graphCanvas = (
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
          <FormattedMessage
            defaultMessage="{count} {count, plural, one {node} other {nodes}}"
            description="Count of workflow nodes displayed in graph view"
            values={{ count: workflowLayout.nodes.length }}
          />
        </Typography.Text>
        <Typography.Text size="sm" color="secondary">
          <FormattedMessage defaultMessage="Scroll to zoom, drag to pan" description="Navigation hint for graph view" />
        </Typography.Text>
      </div>

      <GraphViewWorkflowCanvas
        layout={workflowLayout}
        selectedNodeId={selectedWorkflowNode?.id ?? null}
        highlightedPathNodeIds={highlightedWorkflowNodeIds}
        highlightedPathEdgeIds={highlightedWorkflowEdgeIds}
        onSelectNode={handleSelectWorkflowNode}
        onViewSpanDetails={handleViewSpanDetails}
      />
    </div>
  );

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
      {isDetailsPaneOpen ? (
        <ModelTraceExplorerResizablePane
          initialRatio={getPaneSizeRatios().graphPane}
          paneWidth={paneWidth}
          setPaneWidth={setPaneWidth}
          onRatioChange={onSizeRatioChange}
          leftChild={graphCanvas}
          leftMinWidth={300}
          rightChild={
            <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
              <div
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'flex-end',
                  padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                  borderBottom: `1px solid ${theme.colors.border}`,
                  flexShrink: 0,
                }}
              >
                <Button
                  componentId="graph-view-close-details-pane"
                  icon={<CloseIcon />}
                  size="small"
                  onClick={handleCloseDetailsPane}
                  aria-label="Close details pane"
                />
              </div>
              <ModelTraceExplorerRightPaneTabs
                activeSpan={selectedNode}
                searchFilter=""
                activeMatch={null}
                activeTab={activeTab}
                setActiveTab={setActiveTab}
              />
            </div>
          }
          rightMinWidth={RIGHT_PANE_MIN_WIDTH}
        />
      ) : (
        graphCanvas
      )}
    </div>
  );
};
