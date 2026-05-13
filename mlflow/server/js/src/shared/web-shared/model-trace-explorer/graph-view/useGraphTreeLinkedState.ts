import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import {
  getSpanNodeParentIds,
  getTimelineTreeNodesMap,
  useTimelineTreeExpandedNodes,
  DEFAULT_EXPAND_DEPTH,
} from '../timeline-tree/TimelineTree.utils';
import type { WorkflowNode } from './GraphView.types';

export const useGraphTreeLinkedState = (workflowNodes?: WorkflowNode[]) => {
  const { selectedNode, setSelectedNode, topLevelNodes } = useModelTraceExplorerViewState();

  // selectedWorkflowNode tracks which graph node is highlighted (set by both tree sync and graph click)
  const [selectedWorkflowNode, setSelectedWorkflowNode] = useState<WorkflowNode | null>(null);
  // navigatorNode tracks which node the span navigator displays (set only by explicit graph clicks)
  const [navigatorNode, setNavigatorNode] = useState<WorkflowNode | null>(null);
  const [currentSpanIndex, setCurrentSpanIndex] = useState(0);

  // Flag to prevent circular updates when graph selection triggers tree selection
  const isGraphSelectingRef = useRef(false);

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
      isGraphSelectingRef.current = true;
      setSelectedWorkflowNode(node);
      setNavigatorNode(node);
      if (node && node.spans.length > 0) {
        setCurrentSpanIndex(0);
        const sorted = [...node.spans].sort((a, b) => a.start - b.start);
        focusSpanInTree(sorted[0]);
      }
      // Reset flag after the current event loop to allow subsequent tree selections
      requestAnimationFrame(() => {
        isGraphSelectingRef.current = false;
      });
    },
    [focusSpanInTree],
  );

  const handleNavigateSpan = useCallback(
    (index: number) => {
      if (index < 0 || index >= sortedSpans.length) {
        return;
      }
      isGraphSelectingRef.current = true;
      setCurrentSpanIndex(index);
      focusSpanInTree(sortedSpans[index]);
      requestAnimationFrame(() => {
        isGraphSelectingRef.current = false;
      });
    },
    [sortedSpans, focusSpanInTree],
  );

  // Bidirectional linking: when a span is selected in the tree (not from graph click),
  // find and highlight the corresponding workflow node in the graph
  useEffect(() => {
    if (isGraphSelectingRef.current || !workflowNodes || workflowNodes.length === 0) {
      return;
    }

    if (!selectedNode) {
      setSelectedWorkflowNode(null);
      return;
    }

    const matchingNode = workflowNodes.find((wn) => wn.spans.some((s) => s.key === selectedNode.key)) ?? null;
    if (matchingNode && matchingNode.id !== selectedWorkflowNode?.id) {
      setSelectedWorkflowNode(matchingNode);
      const spanIndex = [...matchingNode.spans]
        .sort((a, b) => a.start - b.start)
        .findIndex((s) => s.key === selectedNode.key);
      setCurrentSpanIndex(Math.max(0, spanIndex));
    }
  }, [selectedNode, workflowNodes, selectedWorkflowNode?.id]);

  return {
    selectedWorkflowNode,
    setSelectedWorkflowNode,
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
  };
};
