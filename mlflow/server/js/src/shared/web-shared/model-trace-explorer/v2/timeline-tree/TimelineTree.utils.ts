import { values } from 'lodash';
import type { Dispatch, SetStateAction } from 'react';
import { useState } from 'react';

import type { TimelineTreeNode } from './TimelineTree.types';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import { getSpanExceptionCount } from '../../spanExceptionEvents';

// expand all nodes by default
export const DEFAULT_EXPAND_DEPTH: number = Infinity;
export const SPAN_INDENT_WIDTH = 16;
export const SPAN_ROW_HEIGHT = 32;
export const TimelineTreeZIndex = {
  HIGH: 5,
  NORMAL: 3,
  LOW: 1,
};

interface TimelineTreeSpanConstraints {
  min: number;
  max: number;
}

// Gets the min and max start and end times of the tree
export const getTimelineTreeSpanConstraints = (
  nodes: TimelineTreeNode[],
  constraints: TimelineTreeSpanConstraints = { min: Number.MAX_SAFE_INTEGER, max: 0 },
): TimelineTreeSpanConstraints => {
  nodes.forEach((node) => {
    const { start, end, children } = node;
    if (start < constraints.min) {
      constraints.min = start;
    }
    if (end > constraints.max) {
      constraints.max = end;
    }
    getTimelineTreeSpanConstraints(children ?? [], constraints);
  });

  return constraints;
};

// Gets a flat list of all expanded nodes in the tree
export const getTimelineTreeExpandedNodesList = <T extends TimelineTreeNode & { children?: T[] }>(
  nodes: T[],
  expandedKeys: Set<string | number>,
): T[] => {
  const expandedNodesFlat: T[] = [];
  const traverseExpanded = (traversedNode: T | undefined) => {
    if (!traversedNode) {
      return;
    }
    expandedNodesFlat.push(traversedNode);
    if (expandedKeys.has(traversedNode.key)) {
      traversedNode.children?.forEach(traverseExpanded);
    }
  };

  nodes.forEach(traverseExpanded);
  return expandedNodesFlat;
};

// Gets a flat list of all nodes in the tree (regardless of expansion status)
export const getTimelineTreeNodesList = <T extends TimelineTreeNode & { children?: T[] }>(nodes: T[]): T[] => {
  const expandedNodesFlat: T[] = [];
  const traverseExpanded = (traversedNode: T | undefined) => {
    if (!traversedNode) {
      return;
    }
    expandedNodesFlat.push(traversedNode);
    traversedNode.children?.forEach(traverseExpanded);
  };

  nodes.forEach(traverseExpanded);
  return expandedNodesFlat;
};

export const getTimelineTreeNodesMap = <T extends TimelineTreeNode & { children?: T[] }>(
  nodes: T[],
  expandDepth: number = Infinity,
): { [nodeId: string]: T } => {
  const nodesMap: { [nodeId: string]: T } = {};

  const traverse = (traversedNode: T | undefined, depth: number) => {
    if (!traversedNode || depth > expandDepth) {
      return;
    }
    nodesMap[traversedNode.key] = traversedNode;
    traversedNode.children?.forEach((child: T) => traverse(child, depth + 1));
  };

  nodes.forEach(traverse, 0);
  return nodesMap;
};

interface UseTimelineTreeExpandedNodesResult {
  expandedKeys: Set<string | number>;
  setExpandedKeys: Dispatch<SetStateAction<Set<string | number>>>;
}

export const useTimelineTreeExpandedNodes = <T extends ModelTraceSpanNode & { children?: T[] }>(
  params: {
    rootNodes?: T[];
    // nodes beyond this depth will be collapsed
    initialExpandDepth?: number;
  } = {},
): UseTimelineTreeExpandedNodesResult => {
  const [expandedKeys, setExpandedKeys] = useState<Set<string | number>>(() => {
    if (params.rootNodes) {
      const list = values(getTimelineTreeNodesMap(params.rootNodes, params.initialExpandDepth)).map((node) => node.key);
      return new Set(list);
    }
    return new Set();
  });

  return {
    expandedKeys,
    setExpandedKeys,
  };
};

interface UseTimelineTreeSelectedNodeResult {
  selectedNode: ModelTraceSpanNode | undefined;
  setSelectedNode: Dispatch<SetStateAction<ModelTraceSpanNode | undefined>>;
}

export const useTimelineTreeSelectedNode = (): UseTimelineTreeSelectedNodeResult => {
  const [selectedNode, setSelectedNode] = useState<ModelTraceSpanNode | undefined>(undefined);

  return {
    selectedNode,
    setSelectedNode,
  };
};

export const spanTimeFormatter = (executionTimeUs: number): string => {
  // Convert to different units based on the time scale
  if (executionTimeUs === 0) {
    return '0s';
  } else if (executionTimeUs >= 60 * 1e6) {
    // More than or equal to 1 minute
    const executionTimeMin = executionTimeUs / 1e6 / 60;
    return `${executionTimeMin.toFixed(2)}m`;
  } else if (executionTimeUs >= 1e5) {
    // More than or equal to 0.1 second. this
    // is to avoid showing 3-digit ms numbers
    const executionTimeSec = executionTimeUs / 1e6;
    return `${executionTimeSec.toFixed(2)}s`;
  } else {
    // Less than 0.1 second (milliseconds)
    const executionTimeMs = executionTimeUs / 1e3;
    return `${executionTimeMs.toFixed(2)}ms`;
  }
};

export const getActiveChildIndex = (node: ModelTraceSpanNode, activeNodeId: string): number => {
  if (node.key === activeNodeId) {
    return 0;
  }

  return (node.children ?? []).findIndex((child) => getActiveChildIndex(child, activeNodeId) > -1);
};

export const getModelTraceSpanNodeDepth = (node: ModelTraceSpanNode): number => {
  if (!node.children || node.children?.length === 0) {
    return 0;
  }

  const childDepths = node.children.map(getModelTraceSpanNodeDepth);
  return Math.max(...childDepths) + 1;
};

export const getSpanNodeParentIds = (
  node: ModelTraceSpanNode,
  nodeMap: { [nodeId: string]: ModelTraceSpanNode },
): Set<string | number> => {
  const parents = new Set<string | number>();

  let currentNode = node;
  while (currentNode && currentNode.parentId) {
    parents.add(currentNode.parentId);
    currentNode = nodeMap[currentNode.parentId];
  }

  return parents;
};

export const isNodeImportant = (node: ModelTraceSpanNode): boolean => {
  // root node is shown at top level, so we don't need to
  // show it in the intermediate nodes list
  if (!node.parentId) {
    return false;
  }

  return (
    [
      ModelSpanType.AGENT,
      ModelSpanType.RETRIEVER,
      ModelSpanType.CHAT_MODEL,
      ModelSpanType.TOOL,
      ModelSpanType.LLM,
    ].includes(node.type ?? ModelSpanType.UNKNOWN) || getSpanExceptionCount(node) > 0
  );
};
