import { values, isString } from 'lodash';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';

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
  modelTraceInfo,
  className,
  selectedSpanId,
  onSelectSpan,
}: {
  modelTraceInfo: ModelTrace['info'];
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
    isInComparisonView,
  } = useModelTraceExplorerViewState();

  // If the parsed root is a synthetic root, render its children as top-level nodes
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore - syntheticRoot is a runtime flag set in parseModelTraceToTree
  const isSyntheticRoot = Boolean(treeNode?.syntheticRoot);
  const rootNodesForRender = useMemo(
    () => (treeNode ? (isSyntheticRoot ? treeNode.children ?? [] : [treeNode]) : []),
    [treeNode, isSyntheticRoot],
  );

  const { expandedKeys, setExpandedKeys } = useTimelineTreeExpandedNodes({
    rootNodes: rootNodesForRender,
    // nodes beyond this depth will be collapsed
    initialExpandDepth: DEFAULT_EXPAND_DEPTH,
  });

  // Track which keys we've auto-expanded so we don't override user collapses on refresh
  const seenAutoExpandedKeysRef = useRef<Set<string | number>>(new Set());

  // When a synthetic root transitions to a true root span (once export completes),
  // ensure the new root is expanded so the tree does not appear collapsed.
  const prevRootKeyRef = useRef<string | number | undefined>(undefined);
  useEffect(() => {
    const currentRootKey = treeNode?.key;
    const prevRootKey = prevRootKeyRef.current;
    if (currentRootKey && currentRootKey !== prevRootKey) {
      // If new root is not expanded yet, expand it but preserve existing expansions
      if (!expandedKeys.has(currentRootKey)) {
        // If synthetic root, do not expand it; expand its immediate children instead
        if (isSyntheticRoot) {
          const next = new Set(expandedKeys);
          (treeNode?.children ?? []).forEach((child) => next.add(child.key as string | number));
          setExpandedKeys(next);
          (treeNode?.children ?? []).forEach((child) => seenAutoExpandedKeysRef.current.add(child.key as any));
        } else {
          const next = new Set(expandedKeys);
          next.add(currentRootKey);
          setExpandedKeys(next);
          // mark root as auto-expanded to avoid re-adding collapsed nodes later
          seenAutoExpandedKeysRef.current.add(currentRootKey);
        }
        // mark root as auto-expanded to avoid re-adding collapsed nodes later
        seenAutoExpandedKeysRef.current.add(currentRootKey);
      }
      prevRootKeyRef.current = currentRootKey;
    }
  }, [treeNode?.key, expandedKeys, setExpandedKeys]);

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
    modelTraceInfo,
  });

  const onSelectNode = (node?: ModelTraceSpanNode) => {
    setSelectedNode(node);
    if (isString(node?.key)) {
      onSelectSpan?.(node?.key);
    }
  };

  // initial render
  useLayoutEffect(() => {
    // On first load, expand nodes up to the default depth.
    // Do not reset expanded nodes on background refresh to avoid flicker.
    if (!expandedKeys || expandedKeys.size === 0) {
      const list = values(getTimelineTreeNodesMap(rootNodesForRender, DEFAULT_EXPAND_DEPTH)).map((node) => node.key);
      const initial = new Set(list);
      setExpandedKeys(initial);
      // remember which keys we auto-expanded at bootstrap
      seenAutoExpandedKeysRef.current = initial;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rootNodesForRender]);

  // Ensure newly arrived spans are auto-expanded by default, but do not re-expand
  // spans that the user has explicitly collapsed.
  useEffect(() => {
    const defaultDepthKeys = values(getTimelineTreeNodesMap(rootNodesForRender, DEFAULT_EXPAND_DEPTH)).map(
      (node) => node.key,
    );

    // Only auto-expand keys we have never auto-expanded before (i.e., truly new)
    const newKeys = defaultDepthKeys.filter((k) => !seenAutoExpandedKeysRef.current.has(k));
    if (newKeys.length > 0) {
      const next = new Set(expandedKeys);
      newKeys.forEach((k) => next.add(k));
      setExpandedKeys(next);
      newKeys.forEach((k) => seenAutoExpandedKeysRef.current.add(k));
    }
  }, [rootNodesForRender, expandedKeys, setExpandedKeys]);

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
