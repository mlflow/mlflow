import { Global } from '@emotion/react';
import { clamp, values, isString } from 'lodash';
import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useResizeObserver } from '@databricks/web-shared/hooks';
import { FormattedMessage } from '@databricks/i18n';
import { ResizableBox } from 'react-resizable';

import type { ModelTrace, ModelTraceSpanNode } from './ModelTrace.types';
import type { ModelTraceExplorerResizablePaneRef } from './ModelTraceExplorerResizablePane';
import ModelTraceExplorerResizablePane from './ModelTraceExplorerResizablePane';
import ModelTraceExplorerSearchBox from './ModelTraceExplorerSearchBox';
import { useModelTraceExplorerViewState } from './ModelTraceExplorerViewStateContext';
import { useModelTraceSearch } from './hooks/useModelTraceSearch';
import { ModelTraceExplorerRightPaneTabs, RIGHT_PANE_MIN_WIDTH } from './right-pane/ModelTraceExplorerRightPaneTabs';
import { TimelineTree } from './timeline-tree';
import {
  DEFAULT_EXPAND_DEPTH,
  getModelTraceSpanNodeDepth,
  getTimelineTreeNodesMap,
  SPAN_INDENT_WIDTH,
} from './timeline-tree/TimelineTree.utils';
import { DEFAULT_WORKFLOW_LAYOUT_CONFIG } from './graph-view/GraphView.types';
import { computeWorkflowLayout } from './graph-view/GraphView.workflow';
import { GraphViewWorkflowCanvas } from './graph-view/GraphViewWorkflowCanvas';
import { GraphViewSpanNavigator } from './graph-view/GraphViewSpanNavigator';
import { useGraphTreeLinkedState } from './graph-view/useGraphTreeLinkedState';

// Default horizontal ratio when graph is enabled (matches graphPane in PaneSizeRatios)
const DEFAULT_GRAPH_PANE_RATIO = 0.75;
const LEFT_PANE_MIN_WIDTH_LARGE_SPACINGS = 7;
const LEFT_PANE_HEADER_MIN_WIDTH_PX = 275;
const GRAPH_MIN_HEIGHT = 150;
// Minimum space reserved for the tree section below the graph
const TREE_MIN_HEIGHT = 200;
// Default ratio of vertical space allocated to the graph canvas
const DEFAULT_GRAPH_HEIGHT_RATIO = 0.65;

export const ModelTraceExplorerDetailView = ({
  modelTraceInfo,
  className,
  selectedSpanId: _selectedSpanId,
  onSelectSpan,
}: {
  modelTraceInfo: ModelTrace['info'];
  className?: string;
  selectedSpanId?: string;
  onSelectSpan?: (selectedSpanId?: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const paneRef = useRef<ModelTraceExplorerResizablePaneRef>(null);
  // Use window width as a rough estimate to avoid a visible flash before ResizeObserver fires.
  // The useLayoutEffect inside ResizablePane will correct to the exact value on first measurement.
  const [paneWidth, setPaneWidth] = useState(() => Math.round(window.innerWidth * DEFAULT_GRAPH_PANE_RATIO));
  const outerContainerRef = useRef<HTMLDivElement>(null);
  const outerSize = useResizeObserver({ ref: outerContainerRef });
  const [isResizing, setIsResizing] = useState(false);

  // Ratio-based graph height: same pattern as ModelTraceExplorerResizablePane.
  // Store the ratio in a ref so it persists across container resizes without
  // causing re-renders. The pixel height is derived from containerHeight * ratio.
  // When containerHeight is not yet available (first render before ResizeObserver fires),
  // we fall back to a pure CSS flex layout so the graph always occupies the same
  // proportional space regardless of timing.
  const graphHeightRatio = useRef(DEFAULT_GRAPH_HEIGHT_RATIO);
  const containerHeight = outerSize?.height ?? 0;
  const hasContainerMeasurement = containerHeight > 0;
  const maxGraphHeight = Math.max(GRAPH_MIN_HEIGHT, containerHeight - TREE_MIN_HEIGHT);
  const graphHeight = hasContainerMeasurement
    ? clamp(containerHeight * graphHeightRatio.current, GRAPH_MIN_HEIGHT, maxGraphHeight)
    : 0;

  const {
    rootNode,
    activeTab,
    setActiveTab,
    showGraph,
    setShowGraph,
    updatePaneSizeRatios,
    getPaneSizeRatios,
    topLevelNodes,
  } = useModelTraceExplorerViewState();

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

  const workflowLayout = useMemo(() => computeWorkflowLayout(rootNode, DEFAULT_WORKFLOW_LAYOUT_CONFIG), [rootNode]);
  const graphAvailable = !!rootNode && workflowLayout.nodes.length > 0;
  const hasGraph = showGraph && graphAvailable;

  const onSizeRatioChange = useCallback(
    (ratio: number) => {
      if (showGraph) {
        updatePaneSizeRatios({ graphPane: ratio });
      } else {
        updatePaneSizeRatios({ detailsPane: ratio });
      }
    },
    [updatePaneSizeRatios, showGraph],
  );

  const handleToggleGraph = useCallback(() => {
    const newShowGraph = !showGraph;
    setShowGraph(newShowGraph);

    // Reset graph height ratio to default when toggling on,
    // so the graph always opens at the same proportional size
    if (newShowGraph) {
      graphHeightRatio.current = DEFAULT_GRAPH_HEIGHT_RATIO;
    }

    const containerWidth = paneRef.current?.getContainerWidth();
    if (containerWidth) {
      const newRatio = newShowGraph ? getPaneSizeRatios().graphPane : getPaneSizeRatios().detailsPane;
      const maxWidth = containerWidth - RIGHT_PANE_MIN_WIDTH;
      const newWidth = clamp(containerWidth * newRatio, LEFT_PANE_HEADER_MIN_WIDTH_PX, maxWidth);
      paneRef.current?.updateRatio(newWidth);
      setPaneWidth(newWidth);
    }
  }, [showGraph, setShowGraph, getPaneSizeRatios, setPaneWidth]);

  const handleGraphResize = useCallback(
    (_e: React.SyntheticEvent, { size }: { size: { height: number } }) => {
      // Update the ratio so it persists across container resizes
      if (containerHeight > 0) {
        graphHeightRatio.current = size.height / containerHeight;
      }
    },
    [containerHeight],
  );

  const { nodeIds: highlightedWorkflowNodeIds, edgeIds: highlightedWorkflowEdgeIds } = useMemo(() => {
    const nodeIds = new Set<string>();
    const edgeIds = new Set<string>();
    const nodeId = selectedWorkflowNode?.id ?? null;

    if (!nodeId || !workflowLayout || workflowLayout.nodes.length === 0) {
      return { nodeIds, edgeIds };
    }

    nodeIds.add(nodeId);

    const incomingEdgesMap = new Map<string, typeof workflowLayout.edges>();
    for (const edge of workflowLayout.edges) {
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
  }, [selectedWorkflowNode, workflowLayout]);

  const handleViewSpanDetails = useCallback(
    (node: ModelTraceSpanNode) => {
      if (selectedWorkflowNode) {
        const idx = sortedSpans.findIndex((s) => s.key === node.key);
        if (idx >= 0) {
          handleNavigateSpan(idx);
          return;
        }
      }
      setSelectedNode(node);
    },
    [selectedWorkflowNode, sortedSpans, handleNavigateSpan, setSelectedNode],
  );

  const {
    matchData,
    searchFilter,
    setSearchFilter,
    spanFilterState,
    setSpanFilterState,
    filteredTreeNodes,
    handleNextSearchMatch,
    handlePreviousSearchMatch,
  } = useModelTraceSearch({
    treeNodes: topLevelNodes,
    selectedNode,
    setSelectedNode,
    setActiveTab,
    setExpandedKeys,
    modelTraceInfo,
  });

  const onSelectNode = (node?: ModelTraceSpanNode) => {
    setSelectedNode(node);
    if (isString(node?.key)) {
      onSelectSpan?.(node?.key);
    }
  };

  useLayoutEffect(() => {
    const list = values(getTimelineTreeNodesMap(filteredTreeNodes, DEFAULT_EXPAND_DEPTH)).map((node) => node.key);
    setExpandedKeys(new Set(list));
  }, [filteredTreeNodes, setExpandedKeys]);

  const leftPaneMinWidth = useMemo(() => {
    const minWidthForSpans =
      Math.max(...filteredTreeNodes.map(getModelTraceSpanNodeDepth)) * SPAN_INDENT_WIDTH +
      LEFT_PANE_MIN_WIDTH_LARGE_SPACINGS * theme.spacing.lg;
    return Math.max(LEFT_PANE_HEADER_MIN_WIDTH_PX, minWidthForSpans);
  }, [filteredTreeNodes, theme.spacing.lg]);

  const { traceStartTime, traceEndTime } = useMemo(() => {
    if (!topLevelNodes || topLevelNodes.length === 0) {
      return { traceStartTime: 0, traceEndTime: 0 };
    }

    const traceStartTime = Math.min(...topLevelNodes.map((node) => node.start));
    const traceEndTime = Math.max(...topLevelNodes.map((node) => node.end));

    return { traceStartTime, traceEndTime };
  }, [topLevelNodes]);

  const graphCanvasContent = (
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
  );

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
      }}
      className={className}
    >
      <div
        css={{
          padding: theme.spacing.xs,
          borderBottom: `1px solid ${theme.colors.border}`,
        }}
      >
        <ModelTraceExplorerSearchBox
          searchFilter={searchFilter}
          setSearchFilter={setSearchFilter}
          matchData={matchData}
          handleNextSearchMatch={handleNextSearchMatch}
          handlePreviousSearchMatch={handlePreviousSearchMatch}
        />
      </div>
      <ModelTraceExplorerResizablePane
        ref={paneRef}
        initialRatio={hasGraph ? getPaneSizeRatios().graphPane : getPaneSizeRatios().detailsPane}
        paneWidth={paneWidth}
        setPaneWidth={setPaneWidth}
        onRatioChange={onSizeRatioChange}
        leftChild={
          <div
            ref={outerContainerRef}
            css={{
              display: 'flex',
              flexDirection: 'column',
              flex: 1,
              minWidth: leftPaneMinWidth,
              minHeight: 0,
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

            {hasGraph && (
              <>
                {hasContainerMeasurement ? (
                  // Once the container is measured, use ResizableBox for manual drag resizing.
                  // The height is derived from containerHeight * graphHeightRatio.
                  <ResizableBox
                    axis="y"
                    width={Infinity}
                    height={graphHeight}
                    minConstraints={[Infinity, GRAPH_MIN_HEIGHT]}
                    maxConstraints={[Infinity, maxGraphHeight]}
                    onResize={handleGraphResize}
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
                    {graphCanvasContent}
                  </ResizableBox>
                ) : (
                  // Before the container is measured (first render), use CSS flex so the
                  // graph takes the same proportional space as the ratio-based height will.
                  // flex: 2 / flex: 1 gives ~67% which closely matches DEFAULT_GRAPH_HEIGHT_RATIO.
                  <div
                    css={{
                      display: 'flex',
                      flexDirection: 'column',
                      flex: 2,
                      minHeight: GRAPH_MIN_HEIGHT,
                      overflow: 'hidden',
                    }}
                  >
                    {graphCanvasContent}
                  </div>
                )}

                <GraphViewSpanNavigator
                  selectedWorkflowNode={selectedWorkflowNode}
                  currentSpanIndex={currentSpanIndex}
                  totalSpans={sortedSpans.length}
                  currentSpan={sortedSpans[currentSpanIndex] ?? null}
                  onNavigatePrev={() => handleNavigateSpan(currentSpanIndex - 1)}
                  onNavigateNext={() => handleNavigateSpan(currentSpanIndex + 1)}
                />
              </>
            )}

            <div
              ref={treeContainerRef}
              css={{
                display: 'flex',
                flexDirection: 'column',
                flex: 1,
                minHeight: 0,
              }}
            >
              <TimelineTree
                rootNodes={filteredTreeNodes}
                selectedNode={selectedNode}
                traceStartTime={traceStartTime}
                traceEndTime={traceEndTime}
                setSelectedNode={onSelectNode}
                css={{ flex: 1 }}
                expandedKeys={expandedKeys}
                setExpandedKeys={setExpandedKeys}
                spanFilterState={spanFilterState}
                setSpanFilterState={setSpanFilterState}
                showGraph={showGraph && graphAvailable}
                onToggleGraph={graphAvailable ? handleToggleGraph : undefined}
              />
            </div>
          </div>
        }
        leftMinWidth={leftPaneMinWidth}
        rightChild={
          <ModelTraceExplorerRightPaneTabs
            activeSpan={selectedNode}
            searchFilter={searchFilter}
            activeMatch={matchData.match}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
          />
        }
        rightMinWidth={RIGHT_PANE_MIN_WIDTH}
      />
    </div>
  );
};
