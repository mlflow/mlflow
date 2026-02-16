import { Global } from '@emotion/react';
import { values } from 'lodash';
import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';

import { Empty, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useResizeObserver } from '@databricks/web-shared/hooks';
import { FormattedMessage } from '@databricks/i18n';
import { ResizableBox } from 'react-resizable';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import ModelTraceExplorerResizablePane from '../ModelTraceExplorerResizablePane';
import { ModelTraceExplorerRightPaneTabs, RIGHT_PANE_MIN_WIDTH } from '../right-pane/ModelTraceExplorerRightPaneTabs';
import { useModelTraceSearch } from '../hooks/useModelTraceSearch';
import { TimelineTree } from '../timeline-tree';
import {
  DEFAULT_EXPAND_DEPTH,
  getModelTraceSpanNodeDepth,
  getTimelineTreeNodesMap,
  SPAN_INDENT_WIDTH,
} from '../timeline-tree/TimelineTree.utils';
import { DEFAULT_WORKFLOW_LAYOUT_CONFIG } from './GraphView.types';
import { computeWorkflowLayout } from './GraphView.workflow';
import { GraphViewWorkflowCanvas } from './GraphViewWorkflowCanvas';
import { GraphViewSpanNavigator } from './GraphViewSpanNavigator';
import { useGraphTreeLinkedState } from './useGraphTreeLinkedState';

const GRAPH_MIN_HEIGHT = 150;
const LEFT_PANE_MIN_WIDTH_LARGE_SPACINGS = 7;
const LEFT_PANE_HEADER_MIN_WIDTH_PX = 275;

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
  let head = 0;

  while (head < toProcess.length) {
    const currentNodeId = toProcess[head++];
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
 * Graph View component that visualizes trace spans as an aggregated workflow graph
 * with a linked tree view below. The graph canvas is on top, a span navigator bar
 * in the middle, and the tree view with details pane on the bottom.
 */
export const GraphView = ({ className }: GraphViewProps) => {
  const { theme } = useDesignSystemTheme();
  const outerContainerRef = useRef<HTMLDivElement>(null);
  const outerSize = useResizeObserver({ ref: outerContainerRef });
  const [graphHeight, setGraphHeight] = useState(300);
  const [isResizing, setIsResizing] = useState(false);
  const [paneWidth, setPaneWidth] = useState(500);

  const { rootNode, activeTab, setActiveTab, updatePaneSizeRatios, getPaneSizeRatios, topLevelNodes } =
    useModelTraceExplorerViewState();

  const {
    selectedWorkflowNode,
    currentSpanIndex,
    sortedSpans,
    expandedKeys,
    setExpandedKeys,
    treeContainerRef,
    handleSelectWorkflowNode,
    handleNavigateSpan,
    selectedNode,
    setSelectedNode,
  } = useGraphTreeLinkedState();

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

  // When a span is clicked in the graph node's popover, find its index and navigate
  const handleViewSpanDetails = useCallback(
    (node: ModelTraceSpanNode) => {
      if (selectedWorkflowNode) {
        const idx = sortedSpans.findIndex((s) => s.key === node.key);
        if (idx >= 0) {
          handleNavigateSpan(idx);
          return;
        }
      }
      // Fallback: just select the node directly
      setSelectedNode(node);
    },
    [selectedWorkflowNode, sortedSpans, handleNavigateSpan, setSelectedNode],
  );

  // Search/filter support for the tree
  const { searchFilter, spanFilterState, setSpanFilterState, filteredTreeNodes } = useModelTraceSearch({
    treeNodes: topLevelNodes,
    selectedNode,
    setSelectedNode,
    setActiveTab,
    setExpandedKeys,
    modelTraceInfo: null,
  });

  // Expand filtered tree nodes only when the search filter changes
  const prevSpanFilterStateRef = useRef(spanFilterState);
  useLayoutEffect(() => {
    if (prevSpanFilterStateRef.current !== spanFilterState) {
      prevSpanFilterStateRef.current = spanFilterState;
      const list = values(getTimelineTreeNodesMap(filteredTreeNodes, DEFAULT_EXPAND_DEPTH)).map((node) => node.key);
      setExpandedKeys(new Set(list));
    }
  }, [filteredTreeNodes, spanFilterState, setExpandedKeys]);

  const leftPaneMinWidth = useMemo(() => {
    const depths = filteredTreeNodes.map(getModelTraceSpanNodeDepth);
    const maxDepth = depths.length > 0 ? Math.max(...depths) : 0;
    const minWidthForSpans = maxDepth * SPAN_INDENT_WIDTH + LEFT_PANE_MIN_WIDTH_LARGE_SPACINGS * theme.spacing.lg;
    return Math.max(LEFT_PANE_HEADER_MIN_WIDTH_PX, minWidthForSpans);
  }, [filteredTreeNodes, theme.spacing.lg]);

  const { traceStartTime, traceEndTime } = useMemo(() => {
    if (!topLevelNodes || topLevelNodes.length === 0) {
      return { traceStartTime: 0, traceEndTime: 0 };
    }
    return {
      traceStartTime: Math.min(...topLevelNodes.map((node) => node.start)),
      traceEndTime: Math.max(...topLevelNodes.map((node) => node.end)),
    };
  }, [topLevelNodes]);

  // Empty state when no trace data
  if (!rootNode) {
    return (
      <div
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

  const leftPane = (
    <div
      ref={outerContainerRef}
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        minWidth: 0,
        overflow: 'hidden',
      }}
    >
      {isResizing && (
        <Global
          styles={{
            'body, :host': {
              userSelect: 'none',
            },
          }}
        />
      )}

      {/* Graph canvas section (vertically resizable) */}
      <ResizableBox
        axis="y"
        width={Infinity}
        height={graphHeight}
        minConstraints={[Infinity, GRAPH_MIN_HEIGHT]}
        maxConstraints={[Infinity, (outerSize?.height ?? 600) - 200]}
        onResize={(_e, { size }) => {
          setGraphHeight(size.height);
        }}
        onResizeStart={() => setIsResizing(true)}
        onResizeStop={() => setIsResizing(false)}
        handle={
          <div
            css={{
              height: theme.spacing.sm,
              cursor: 'ns-resize',
              backgroundColor: 'transparent',
              position: 'relative',
              flexShrink: 0,
              zIndex: 1,
              ':hover': {
                backgroundColor: 'rgba(0,0,0,0.1)',
              },
            }}
          />
        }
        css={{
          display: 'flex',
          flexDirection: 'column',
          flexShrink: 0,
        }}
      >
        <div
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
              <FormattedMessage
                defaultMessage="Scroll to zoom, drag background to pan, drag nodes to reposition"
                description="Navigation hint for graph view"
              />
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
      </ResizableBox>

      {/* Span navigator bar */}
      <GraphViewSpanNavigator
        selectedWorkflowNode={selectedWorkflowNode}
        currentSpanIndex={currentSpanIndex}
        totalSpans={sortedSpans.length}
        currentSpan={sortedSpans[currentSpanIndex] ?? null}
        onNavigatePrev={() => handleNavigateSpan(currentSpanIndex - 1)}
        onNavigateNext={() => handleNavigateSpan(currentSpanIndex + 1)}
      />

      {/* Tree section (takes remaining space below graph) */}
      <div
        ref={treeContainerRef}
        css={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
          overflow: 'hidden',
        }}
      >
        <TimelineTree
          rootNodes={filteredTreeNodes}
          selectedNode={selectedNode}
          traceStartTime={traceStartTime}
          traceEndTime={traceEndTime}
          setSelectedNode={setSelectedNode}
          css={{ flex: 1 }}
          expandedKeys={expandedKeys}
          setExpandedKeys={setExpandedKeys}
          spanFilterState={spanFilterState}
          setSpanFilterState={setSpanFilterState}
        />
      </div>
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
      <ModelTraceExplorerResizablePane
        initialRatio={getPaneSizeRatios().graphPane}
        paneWidth={paneWidth}
        setPaneWidth={setPaneWidth}
        onRatioChange={onSizeRatioChange}
        leftChild={leftPane}
        leftMinWidth={leftPaneMinWidth}
        rightChild={
          <ModelTraceExplorerRightPaneTabs
            activeSpan={selectedNode}
            searchFilter={searchFilter}
            activeMatch={null}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
          />
        }
        rightMinWidth={RIGHT_PANE_MIN_WIDTH}
      />
    </div>
  );
};
