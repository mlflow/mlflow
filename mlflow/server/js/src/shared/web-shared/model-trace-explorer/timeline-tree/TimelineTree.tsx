import { useCallback, useMemo } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { getTimelineTreeExpandedNodesList } from './TimelineTree.utils';
import { TimelineTreeHeader } from './TimelineTreeHeader';
import { TimelineTreeNode } from './TimelineTreeNode';
import { TimelineTreeGanttBars } from './gantt/TimelineTreeGanttBars';
import type { ModelTraceSpanNode, SpanFilterState } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

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
  className?: string;
}) => {
  const { theme } = useDesignSystemTheme();

  const onSpanClick = useCallback(
    (node) => {
      setSelectedNode?.(node);
    },
    [
      // comment to prevent prettier format after copybara
      setSelectedNode,
    ],
  );

  const { showTimelineTreeGantt: showTimelineInfo, setShowTimelineTreeGantt: setShowTimelineInfo } =
    useModelTraceExplorerViewState();

  const expandedNodesList = useMemo(
    () => getTimelineTreeExpandedNodesList(rootNodes, expandedKeys),
    [rootNodes, expandedKeys],
  );

  const treeElement = useMemo(
    () => (
      <div
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
          />
        ) : (
          rootNodes.map((node) => (
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
          ))
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
            setShowTimelineInfo={setShowTimelineInfo}
            spanFilterState={spanFilterState}
            setSpanFilterState={setSpanFilterState}
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
