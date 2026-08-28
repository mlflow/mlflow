import { Global } from '@emotion/react';
import { clamp, values, isString } from 'lodash';
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import { useResizeObserver } from '../../hooks/useResizeObserver';
import { ResizableBox } from 'react-resizable';

import type { ModelTrace, ModelTraceSpanNode } from './ModelTrace.types';
import type { ModelTraceExplorerTab as SharedModelTraceExplorerTab } from '../ModelTrace.types';
import { getIconTypeForSpan } from './ModelTraceExplorer.utils';
import { ModelTraceExplorerIcon } from './ModelTraceExplorerIcon';
import { useModelTraceExplorerContext } from './ModelTraceExplorerContext';
import type { ModelTraceExplorerResizablePaneRef } from './ModelTraceExplorerResizablePane';
import ModelTraceExplorerResizablePane from './ModelTraceExplorerResizablePane';
import ModelTraceExplorerSearchBox from './ModelTraceExplorerSearchBox';
import { useModelTraceExplorerViewState } from './ModelTraceExplorerViewStateContext';
import { useModelTraceSearch } from '../hooks/useModelTraceSearch';
import { ModelTraceExplorerRightPaneTabs, RIGHT_PANE_MIN_WIDTH } from './right-pane/ModelTraceExplorerRightPaneTabs';
import { TimelineTree } from './timeline-tree/TimelineTree';
import {
  DEFAULT_EXPAND_DEPTH,
  getModelTraceSpanNodeDepth,
  getTimelineTreeNodesMap,
  SPAN_INDENT_WIDTH,
} from './timeline-tree/TimelineTree.utils';
import { DEFAULT_WORKFLOW_LAYOUT_CONFIG, EXPANDED_WORKFLOW_LAYOUT_CONFIG } from '../graph-view/GraphView.types';
import { computeWorkflowPathToRoot } from '../graph-view/GraphView.utils';
import { computeWorkflowLayout } from '../graph-view/GraphView.workflow';
import { GraphViewSpanNavigator } from '../graph-view/GraphViewSpanNavigator';
import { useGraphTreeLinkedState } from '../graph-view/useGraphTreeLinkedState';

const GraphViewWorkflowCanvas = React.lazy(() =>
  import('../graph-view/GraphViewWorkflowCanvas').then((m) => ({
    default: m.GraphViewWorkflowCanvas,
  })),
);

const renderRedesignedGraphNodeIcon = (spanType: string, hasException: boolean): JSX.Element => (
  <ModelTraceExplorerIcon type={getIconTypeForSpan(spanType)} hasException={hasException} />
);

const LEFT_PANE_MIN_WIDTH_LARGE_SPACINGS = 7;
const LEFT_PANE_HEADER_MIN_WIDTH_PX = 280;
const GRAPH_MIN_HEIGHT = 120;
// Minimum space reserved for the tree section above the graph
const TREE_MIN_HEIGHT = 200;
// Default ratio of vertical space allocated to the graph canvas (below the tree).
// Kept small so the span tree remains the primary view.
const DEFAULT_GRAPH_HEIGHT_RATIO = 0.25;
// Expanded ratio when the user clicks the expand button on the graph.
const EXPANDED_GRAPH_HEIGHT_RATIO = 0.75;
// Ratio of the container width the left pane occupies when graph is fully expanded.
const EXPANDED_PANE_WIDTH_RATIO = 0.65;

const ResizeHandle = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement> & { handleAxis?: string }>(
  function ResizeHandle(props, ref) {
    const { theme } = useDesignSystemTheme();
    const { handleAxis: _handleAxis, ...domProps } = props;

    return (
      <div
        ref={ref}
        css={{
          height: 8,
          cursor: 'ns-resize',
          backgroundColor: 'transparent',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          zIndex: 1,
          ':hover': {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
          },
        }}
        {...domProps}
      />
    );
  },
);

const GRAPH_NAVIGATOR_HEIGHT = 36;

export const ModelTraceExplorerDetailView = ({
  modelTraceInfo,
  className,
  selectedSpanId: _selectedSpanId,
  onSelectSpan,
  enableGraphView = true,
}: {
  modelTraceInfo: ModelTrace['info'];
  className?: string;
  selectedSpanId?: string;
  onSelectSpan?: (selectedSpanId?: string) => void;
  /**
   * When `false`, the graph canvas and toggle button are suppressed entirely.
   * Defaults to `true` to preserve existing behavior.
   */
  enableGraphView?: boolean;
}): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const paneRef = useRef<ModelTraceExplorerResizablePaneRef>(null);
  // Use window width as a rough estimate to avoid a visible flash before ResizeObserver fires.
  // The useLayoutEffect inside ResizablePane will correct to the exact value on first measurement.
  const [paneWidth, setPaneWidth] = useState(() =>
    Math.round(window.innerWidth * (window.innerWidth <= 768 ? 0.33 : 0.25)),
  );
  const outerContainerRef = useRef<HTMLDivElement>(null);
  const outerSize = useResizeObserver({ ref: outerContainerRef });
  const [isResizing, setIsResizing] = useState(false);
  const [isGraphExpanded, setIsGraphExpanded] = useState(false);
  const preExpandPaneRatioRef = useRef<number | null>(null);
  const { isSearchVisible } = useModelTraceExplorerContext();

  // Ratio-based graph height: same pattern as ModelTraceExplorerResizablePane.
  // Store the ratio in a ref so it persists across container resizes without
  // causing re-renders. The pixel height is derived from containerHeight * ratio.
  // When containerHeight is not yet available (first render before ResizeObserver fires),
  // we fall back to a pure CSS flex layout so the graph always occupies the same
  // proportional space regardless of timing.
  const graphHeightRatio = useRef(DEFAULT_GRAPH_HEIGHT_RATIO);
  const containerHeight = outerSize?.height ?? 0;
  const hasContainerMeasurement = containerHeight > 0;
  const maxGraphHeight = Math.max(GRAPH_MIN_HEIGHT, containerHeight - TREE_MIN_HEIGHT - GRAPH_NAVIGATOR_HEIGHT);
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
    selectedNode: viewStateSelectedNode,
    setSelectedNode: setViewStateSelectedNode,
  } = useModelTraceExplorerViewState();

  const activeLayoutConfig = isGraphExpanded ? EXPANDED_WORKFLOW_LAYOUT_CONFIG : DEFAULT_WORKFLOW_LAYOUT_CONFIG;
  const workflowLayout = useMemo(
    () => computeWorkflowLayout(rootNode, activeLayoutConfig),
    [rootNode, activeLayoutConfig],
  );

  const {
    selectedWorkflowNode,
    navigatorNode,
    currentSpanIndex,
    sortedSpans,
    expandedKeys,
    setExpandedKeys,
    treeContainerRef,
    handleSelectWorkflowNode,
    handleNavigateSpan,
    selectedNode,
    setSelectedNode,
  } = useGraphTreeLinkedState({
    workflowNodes: workflowLayout.nodes,
    selectedNode: viewStateSelectedNode,
    setSelectedNode: setViewStateSelectedNode,
    topLevelNodes,
  });

  // The shared useModelTraceSearch hook drives node selection and tab activation together via a
  // single setSelectedNodeAndTab callback. v2's view state keeps these as separate setters, so
  // bridge them here. The shared tab union carries 'chat' and 'links' tabs that v2 does not
  // surface separately, so fold those onto 'content'.
  const setSearchSelectedNodeAndTab = useCallback(
    (node: ModelTraceSpanNode, tab: SharedModelTraceExplorerTab) => {
      setSelectedNode(node);
      setActiveTab(tab === 'chat' || tab === 'links' ? 'content' : tab);
    },
    [setSelectedNode, setActiveTab],
  );

  const graphAvailable = enableGraphView && Boolean(rootNode) && workflowLayout.nodes.length > 0;
  const hasGraph = showGraph && graphAvailable;

  const onSizeRatioChange = useCallback(
    (ratio: number) => {
      updatePaneSizeRatios({ detailsPane: ratio });
    },
    [updatePaneSizeRatios],
  );

  const handleToggleGraph = useCallback(() => {
    const newShowGraph = !showGraph;
    setShowGraph(newShowGraph);

    // Reset graph height ratio to default when toggling on,
    // so the graph always opens at the same proportional size
    if (newShowGraph) {
      graphHeightRatio.current = DEFAULT_GRAPH_HEIGHT_RATIO;
    }

    // If graph was expanded, restore pane width to pre-expand state
    if (!newShowGraph && isGraphExpanded) {
      const containerWidth = paneRef.current?.getContainerWidth();
      if (containerWidth && preExpandPaneRatioRef.current !== null) {
        const restoredWidth = clamp(
          containerWidth * preExpandPaneRatioRef.current,
          LEFT_PANE_HEADER_MIN_WIDTH_PX,
          containerWidth - RIGHT_PANE_MIN_WIDTH,
        );
        setPaneWidth(restoredWidth);
        paneRef.current?.updateRatio(restoredWidth);
      }
      preExpandPaneRatioRef.current = null;
    }

    setIsGraphExpanded(false);
  }, [showGraph, setShowGraph, isGraphExpanded, setPaneWidth]);

  const handleToggleGraphExpand = useCallback(() => {
    setIsGraphExpanded((prev) => {
      const next = !prev;

      // Update vertical ratio
      graphHeightRatio.current = next ? EXPANDED_GRAPH_HEIGHT_RATIO : DEFAULT_GRAPH_HEIGHT_RATIO;

      // Update horizontal pane width
      const containerWidth = paneRef.current?.getContainerWidth();
      if (containerWidth) {
        if (next) {
          // Save the current ratio before expanding
          preExpandPaneRatioRef.current = paneWidth / containerWidth;
          const expandedWidth = clamp(
            containerWidth * EXPANDED_PANE_WIDTH_RATIO,
            LEFT_PANE_HEADER_MIN_WIDTH_PX,
            containerWidth - RIGHT_PANE_MIN_WIDTH,
          );
          setPaneWidth(expandedWidth);
          paneRef.current?.updateRatio(expandedWidth);
        } else {
          // Restore the pre-expand ratio
          const restoreRatio = preExpandPaneRatioRef.current ?? (window.innerWidth <= 768 ? 0.33 : 0.25);
          const restoredWidth = clamp(
            containerWidth * restoreRatio,
            LEFT_PANE_HEADER_MIN_WIDTH_PX,
            containerWidth - RIGHT_PANE_MIN_WIDTH,
          );
          setPaneWidth(restoredWidth);
          paneRef.current?.updateRatio(restoredWidth);
          preExpandPaneRatioRef.current = null;
        }
      }

      return next;
    });
  }, [paneWidth, setPaneWidth]);

  const handleGraphResize = useCallback(
    (_e: React.SyntheticEvent, { size }: { size: { height: number } }) => {
      // Update the ratio so it persists across container resizes
      if (containerHeight > 0) {
        graphHeightRatio.current = size.height / containerHeight;
        // Sync the expanded state with the actual ratio so the button icon
        // reflects the current size after a manual drag.
        setIsGraphExpanded(graphHeightRatio.current >= (DEFAULT_GRAPH_HEIGHT_RATIO + EXPANDED_GRAPH_HEIGHT_RATIO) / 2);
      }
    },
    [containerHeight],
  );

  const { nodeIds: highlightedWorkflowNodeIds, edgeIds: highlightedWorkflowEdgeIds } = useMemo(
    () => computeWorkflowPathToRoot(selectedWorkflowNode?.id ?? null, workflowLayout),
    [selectedWorkflowNode, workflowLayout],
  );

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
    setSelectedNodeAndTab: setSearchSelectedNodeAndTab,
    setExpandedKeys,
    modelTraceInfo,
  });

  useEffect(() => {
    if (!isSearchVisible) {
      setSearchFilter('');
    }
  }, [isSearchVisible, setSearchFilter]);

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

  const graphCanvasContent = hasGraph ? (
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
      <React.Suspense fallback={null}>
        <GraphViewWorkflowCanvas
          layout={workflowLayout}
          selectedNodeId={selectedWorkflowNode?.id ?? null}
          highlightedPathNodeIds={highlightedWorkflowNodeIds}
          highlightedPathEdgeIds={highlightedWorkflowEdgeIds}
          onSelectNode={handleSelectWorkflowNode}
          onViewSpanDetails={handleViewSpanDetails}
          isGraphExpanded={isGraphExpanded}
          onToggleGraphExpand={handleToggleGraphExpand}
          renderNodeIcon={renderRedesignedGraphNodeIcon}
        />
      </React.Suspense>
    </div>
  ) : null;

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
      {isSearchVisible && (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            flexShrink: 0,
            borderBottom: `1px solid ${theme.colors.border}`,
            backgroundColor: theme.colors.backgroundPrimary,
            padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
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
      )}
      <ModelTraceExplorerResizablePane
        ref={paneRef}
        initialRatio={getPaneSizeRatios().detailsPane}
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

            {/* Tree is primary — always on top, fills remaining space above graph */}
            <div
              ref={treeContainerRef}
              css={{
                display: 'flex',
                flexDirection: 'column',
                // Pre-measurement: flex: 3 gives tree ~75% vs graph flex: 1 ~25%.
                // Post-measurement: flex: 1 fills all space above the fixed-height graph.
                flex: hasGraph && !hasContainerMeasurement ? 3 : 1,
                minHeight: hasGraph ? TREE_MIN_HEIGHT : 0,
                overflow: 'auto',
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

            {/* Graph is secondary — below the tree */}
            {hasGraph && (
              <>
                {hasContainerMeasurement ? (
                  <ResizableBox
                    axis="y"
                    width={Infinity}
                    height={graphHeight}
                    minConstraints={[Infinity, GRAPH_MIN_HEIGHT]}
                    maxConstraints={[Infinity, maxGraphHeight]}
                    resizeHandles={['n']}
                    onResize={handleGraphResize}
                    onResizeStart={() => setIsResizing(true)}
                    onResizeStop={() => setIsResizing(false)}
                    handle={<ResizeHandle />}
                    css={{
                      display: 'flex',
                      flexDirection: 'column',
                      flexShrink: 0,
                      position: 'relative',
                    }}
                  >
                    {graphCanvasContent}
                  </ResizableBox>
                ) : (
                  <div
                    css={{
                      display: 'flex',
                      flexDirection: 'column',
                      flex: 1,
                      minHeight: GRAPH_MIN_HEIGHT,
                      overflow: 'hidden',
                    }}
                  >
                    {graphCanvasContent}
                  </div>
                )}
                <GraphViewSpanNavigator
                  selectedWorkflowNode={navigatorNode}
                  currentSpanIndex={currentSpanIndex}
                  totalSpans={sortedSpans.length}
                  currentSpan={sortedSpans[currentSpanIndex] ?? null}
                  onNavigatePrev={() => handleNavigateSpan(currentSpanIndex - 1)}
                  onNavigateNext={() => handleNavigateSpan(currentSpanIndex + 1)}
                />
              </>
            )}
          </div>
        }
        leftMinWidth={leftPaneMinWidth}
        rightChild={
          <ModelTraceExplorerRightPaneTabs
            activeSpan={selectedNode}
            modelTraceInfo={modelTraceInfo}
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
