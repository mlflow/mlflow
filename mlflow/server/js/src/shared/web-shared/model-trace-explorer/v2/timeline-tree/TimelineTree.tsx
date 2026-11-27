import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { getTimelineTreeExpandedRows, SPAN_ROW_HEIGHT, SPAN_ROW_HEIGHT_WITH_METADATA } from './TimelineTree.utils';
import { TimelineTreeHeader } from './TimelineTreeHeader';
import { getTimelineTreeRowHeight, TimelineTreeNode } from './TimelineTreeNode';
import { TimelineTreeGanttBars } from './gantt/TimelineTreeGanttBars';
import type { ModelTraceSpanNode, SpanFilterState } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { useModelTraceExplorerPreferences } from '../ModelTraceExplorerPreferencesContext';
import { useTimelineTreeRowVirtualizer } from './useTimelineTreeRowVirtualizer';

export const TimelineTree = <NodeType extends ModelTraceSpanNode & { children?: NodeType[] }>({
  rootNodes,
  selectedNode,
  setSelectedNode,
  traceStartTime,
  traceEndTime,
  expandedKeys,
  setExpandedKeys,
  spanFilterState,
  setSpanFilterState,
  showGraph,
  onToggleGraph,
  className,
}: {
  selectedNode?: NodeType;
  setSelectedNode: (node: ModelTraceSpanNode) => void;
  traceStartTime: number;
  traceEndTime: number;
  rootNodes: NodeType[];
  expandedKeys: Set<string | number>;
  setExpandedKeys: (keys: Set<string | number>) => void;
  spanFilterState: SpanFilterState;
  setSpanFilterState: (state: SpanFilterState) => void;
  showGraph?: boolean;
  onToggleGraph?: () => void;
  className?: string;
}): React.ReactElement => {
  const { theme } = useDesignSystemTheme();
  const [scrollElement, setScrollElement] = useState<HTMLDivElement | null>(null);
  const scrollTopBeforeViewChange = useRef<number>();

  const onSpanClick = useCallback(
    (node: ModelTraceSpanNode) => {
      setSelectedNode?.(node);
    },
    [
      // comment to prevent prettier format after copybara
      setSelectedNode,
    ],
  );

  const {
    showTimelineTreeGantt: showTimelineInfo,
    setShowTimelineTreeGantt: setShowTimelineInfo,
    comparisonNodeMap,
    comparisonNodeRoleMap,
  } = useModelTraceExplorerViewState();
  const { timelineTreeMetrics } = useModelTraceExplorerPreferences();
  const rowHeight = timelineTreeMetrics.length > 1 ? SPAN_ROW_HEIGHT_WITH_METADATA : SPAN_ROW_HEIGHT;
  const expandedRows = useMemo(
    () => getTimelineTreeExpandedRows(rootNodes, expandedKeys, selectedNode?.key),
    [rootNodes, expandedKeys, selectedNode?.key],
  );
  const expandedNodesList = useMemo(() => expandedRows.map(({ node }) => node), [expandedRows]);
  const getRowHeight = useCallback(
    (node: ModelTraceSpanNode) =>
      getTimelineTreeRowHeight(
        node,
        timelineTreeMetrics,
        comparisonNodeMap[String(node.key)],
        comparisonNodeRoleMap[String(node.key)],
      ),
    [comparisonNodeMap, comparisonNodeRoleMap, timelineTreeMetrics],
  );
  const setShowTimelineInfoPreservingScroll = useCallback(
    (show: boolean) => {
      if (show === showTimelineInfo) {
        scrollTopBeforeViewChange.current = undefined;
        return;
      }
      const currentRowHeight = showTimelineInfo ? SPAN_ROW_HEIGHT : rowHeight;
      const targetRowHeight = show ? SPAN_ROW_HEIGHT : rowHeight;
      scrollTopBeforeViewChange.current = scrollElement
        ? (scrollElement.scrollTop / currentRowHeight) * targetRowHeight
        : undefined;
      setShowTimelineInfo(show);
    },
    [rowHeight, scrollElement, setShowTimelineInfo, showTimelineInfo],
  );
  const { rowVirtualizer, virtualized } = useTimelineTreeRowVirtualizer(
    expandedNodesList,
    scrollElement,
    selectedNode?.key ?? '',
    getRowHeight,
    !showTimelineInfo,
  );
  const virtualRows = rowVirtualizer.getVirtualItems();
  useLayoutEffect(() => {
    if (scrollElement && scrollTopBeforeViewChange.current !== undefined) {
      scrollElement.scrollTop = scrollTopBeforeViewChange.current;
      scrollTopBeforeViewChange.current = undefined;
    }
  }, [scrollElement, showTimelineInfo]);

  const treeElement = useMemo(
    () => (
      <div
        ref={setScrollElement}
        css={{
          flex: 1,
          overflow: 'auto',
          minHeight: '100%',
          boxSizing: 'border-box',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {showTimelineInfo ? (
          <TimelineTreeGanttBars
            nodes={expandedNodesList}
            selectedKey={selectedNode?.key ?? ''}
            onSelect={onSpanClick}
            traceStartTime={traceStartTime}
            traceEndTime={traceEndTime}
            expandedKeys={expandedKeys}
            setExpandedKeys={setExpandedKeys}
            scrollElement={scrollElement}
          />
        ) : virtualized ? (
          <div css={{ position: 'relative', flexShrink: 0 }} style={{ height: rowVirtualizer.getTotalSize() }}>
            {virtualRows.map(({ index, start }) => {
              const row = expandedRows[index];
              return (
                <div
                  key={row.node.key}
                  ref={rowVirtualizer.measureElement}
                  data-index={index}
                  css={{ position: 'absolute', top: 0, left: 0, width: '100%' }}
                  style={{ transform: `translateY(${start}px)` }}
                >
                  <TimelineTreeNode
                    node={row.node}
                    expandedKeys={expandedKeys}
                    setExpandedKeys={setExpandedKeys}
                    selectedKey={selectedNode?.key ?? ''}
                    traceStartTime={traceStartTime}
                    traceEndTime={traceEndTime}
                    onSelect={onSpanClick}
                    linesToRender={row.linesToRender}
                    renderChildren={false}
                  />
                </div>
              );
            })}
          </div>
        ) : (
          <>
            {rootNodes.map((node) => (
              <TimelineTreeNode
                key={node.key}
                node={node}
                expandedKeys={expandedKeys}
                setExpandedKeys={setExpandedKeys}
                selectedKey={selectedNode?.key ?? ''}
                traceStartTime={traceStartTime}
                traceEndTime={traceEndTime}
                onSelect={onSpanClick}
                linesToRender={[]}
              />
            ))}
          </>
        )}
      </div>
    ),
    [
      showTimelineInfo,
      expandedNodesList,
      selectedNode?.key,
      onSpanClick,
      traceStartTime,
      traceEndTime,
      rootNodes,
      expandedKeys,
      setExpandedKeys,
      scrollElement,
      virtualRows,
      virtualized,
      rowVirtualizer,
      expandedRows,
    ],
  );

  return (
    <div
      css={{
        height: '100%',
        borderRadius: theme.legacyBorders.borderRadiusMd,
        overflow: 'hidden',
        display: 'flex',
      }}
      className={className}
    >
      <div
        css={{
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          <TimelineTreeHeader
            showTimelineInfo={showTimelineInfo}
            setShowTimelineInfo={setShowTimelineInfoPreservingScroll}
            spanFilterState={spanFilterState}
            setSpanFilterState={setSpanFilterState}
            showGraph={showGraph}
            onToggleGraph={onToggleGraph}
          />
          {rootNodes.length > 0 ? (
            <div css={{ flex: 1, overflowY: 'auto', display: 'flex' }}>{treeElement}</div>
          ) : (
            <div
              css={{
                flex: 1,
                display: 'flex',
                justifyContent: 'center',
                padding: theme.spacing.md,
                paddingTop: theme.spacing.lg,
              }}
            >
              <FormattedMessage
                defaultMessage="No results found. Try using a different search term."
                description="Model trace explorer > no results found"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
