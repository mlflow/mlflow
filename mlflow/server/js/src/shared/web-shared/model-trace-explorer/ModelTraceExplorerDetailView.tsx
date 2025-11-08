import { values, isString } from 'lodash';
import { useLayoutEffect, useMemo, useRef, useState } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';

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
  useTimelineTreeExpandedNodes,
} from './timeline-tree/TimelineTree.utils';

// this is the number of large spacings we need in order to
// properly calculate the min width for the left pane. it's:
// - 1 for left and right padding
// - 4 for the right collapse button + time marker
// - 1 for span icon
// - 1 for buffer (leave some space to render text)
const LEFT_PANE_MIN_WIDTH_LARGE_SPACINGS = 7;
const LEFT_PANE_HEADER_MIN_WIDTH_PX = 275;

const getDefaultSplitRatio = (): number => {
  if (window.innerWidth <= 768) {
    return 0.33;
  }

  return 0.25;
};

export const ModelTraceExplorerDetailView = ({
  modelTrace,
  className,
  selectedSpanId,
  onSelectSpan,
}: {
  modelTrace: ModelTrace;
  className?: string;
  selectedSpanId?: string;
  onSelectSpan?: (selectedSpanId?: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const initialRatio = getDefaultSplitRatio();
  const paneRef = useRef<ModelTraceExplorerResizablePaneRef>(null);
  const [paneWidth, setPaneWidth] = useState(500);

  const {
    rootNode: treeNode,
    selectedNode,
    setSelectedNode,
    activeTab,
    setActiveTab,
  } = useModelTraceExplorerViewState();

  const { expandedKeys, setExpandedKeys } = useTimelineTreeExpandedNodes({
    rootNodes: treeNode ? [treeNode] : [],
    // nodes beyond this depth will be collapsed
    initialExpandDepth: DEFAULT_EXPAND_DEPTH,
  });

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
    treeNode,
    selectedNode,
    setSelectedNode,
    setActiveTab,
    setExpandedKeys,
    modelTraceInfo: modelTrace?.info,
  });

  const onSelectNode = (node?: ModelTraceSpanNode) => {
    setSelectedNode(node);
    if (isString(node?.key)) {
      onSelectSpan?.(node?.key);
    }
  };

  // initial render
  useLayoutEffect(() => {
    // expand all nodes up to the default depth when the tree changes
    const list = values(getTimelineTreeNodesMap(filteredTreeNodes, DEFAULT_EXPAND_DEPTH)).map((node) => node.key);
    setExpandedKeys(new Set(list));
  }, [filteredTreeNodes, setExpandedKeys]);

  const leftPaneMinWidth = useMemo(() => {
    // min width necessary to render all the spans in the tree accounting for indentation
    const minWidthForSpans =
      Math.max(...filteredTreeNodes.map(getModelTraceSpanNodeDepth)) * SPAN_INDENT_WIDTH +
      LEFT_PANE_MIN_WIDTH_LARGE_SPACINGS * theme.spacing.lg;
    // min width necessary to render the header, given that it has a bunch of buttons
    return Math.max(LEFT_PANE_HEADER_MIN_WIDTH_PX, minWidthForSpans);
  }, [filteredTreeNodes, theme.spacing.lg]);

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
        initialRatio={initialRatio}
        paneWidth={paneWidth}
        setPaneWidth={setPaneWidth}
        leftChild={
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              flex: 1,
              minWidth: leftPaneMinWidth,
            }}
          >
            <TimelineTree
              rootNodes={filteredTreeNodes}
              selectedNode={selectedNode}
              traceStartTime={treeNode?.start ?? 0}
              traceEndTime={treeNode?.end ?? 0}
              setSelectedNode={onSelectNode}
              css={{ flex: 1 }}
              expandedKeys={expandedKeys}
              setExpandedKeys={setExpandedKeys}
              spanFilterState={spanFilterState}
              setSpanFilterState={setSpanFilterState}
            />
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
