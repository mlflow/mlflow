import { useCallback, useMemo, useRef, useState } from 'react';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import {
  getSpanNodeParentIds,
  getTimelineTreeNodesMap,
  useTimelineTreeExpandedNodes,
  DEFAULT_EXPAND_DEPTH,
} from '../timeline-tree/TimelineTree.utils';
import type { WorkflowNode } from './GraphView.types';

export const useGraphTreeLinkedState = () => {
  const { selectedNode, setSelectedNode, topLevelNodes } = useModelTraceExplorerViewState();

  const [selectedWorkflowNode, setSelectedWorkflowNode] = useState<WorkflowNode | null>(null);
  const [currentSpanIndex, setCurrentSpanIndex] = useState(0);

  const { expandedKeys, setExpandedKeys } = useTimelineTreeExpandedNodes({
    rootNodes: topLevelNodes,
    initialExpandDepth: DEFAULT_EXPAND_DEPTH,
  });

  const treeContainerRef = useRef<HTMLDivElement>(null);

  const nodeMap = useMemo(() => getTimelineTreeNodesMap(topLevelNodes), [topLevelNodes]);

  const sortedSpans = useMemo(() => {
    if (!selectedWorkflowNode) {
      return [];
    }
    return [...selectedWorkflowNode.spans].sort((a, b) => a.start - b.start);
  }, [selectedWorkflowNode]);

  const focusSpanInTree = useCallback(
    (span: ModelTraceSpanNode) => {
      setSelectedNode(span);

      const parents = getSpanNodeParentIds(span, nodeMap);
      setExpandedKeys((prev) => new Set([...prev, ...parents]));

      requestAnimationFrame(() => {
        const el = treeContainerRef.current?.querySelector(`[data-testid="timeline-tree-node-${span.key}"]`);
        el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      });
    },
    [setSelectedNode, nodeMap, setExpandedKeys],
  );

  const handleSelectWorkflowNode = useCallback(
    (node: WorkflowNode | null) => {
      setSelectedWorkflowNode(node);
      if (node && node.spans.length > 0) {
        setCurrentSpanIndex(0);
        const sorted = [...node.spans].sort((a, b) => a.start - b.start);
        focusSpanInTree(sorted[0]);
      }
    },
    [focusSpanInTree],
  );

  const handleNavigateSpan = useCallback(
    (index: number) => {
      if (index < 0 || index >= sortedSpans.length) {
        return;
      }
      setCurrentSpanIndex(index);
      focusSpanInTree(sortedSpans[index]);
    },
    [sortedSpans, focusSpanInTree],
  );

  return {
    selectedWorkflowNode,
    setSelectedWorkflowNode,
    currentSpanIndex,
    sortedSpans,
    expandedKeys,
    setExpandedKeys,
    treeContainerRef,
    handleSelectWorkflowNode,
    handleNavigateSpan,
    selectedNode,
    setSelectedNode,
  };
};
