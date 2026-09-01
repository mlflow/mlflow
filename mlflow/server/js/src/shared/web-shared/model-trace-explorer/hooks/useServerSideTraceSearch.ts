import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { ModelTrace, ModelTraceInfoV3, ModelTraceSpanNode } from '../ModelTrace.types';
import { getExperimentTraceV3 } from '../api';
import { isV3ModelTraceInfo, parseModelTraceToTreeWithMultipleRoots } from '../ModelTraceExplorer.utils';

const DEBOUNCE_MS = 400;

type SearchableTreeNode<NodeType> = {
  key: string | number;
  children?: NodeType[];
  events?: Array<{ name: string }>;
};

export const restoreServerSearchHierarchy = <NodeType extends SearchableTreeNode<NodeType>>(
  sourceNodes: NodeType[],
  matchedNodes: NodeType[],
  showParents: boolean,
  showExceptions: boolean,
): NodeType[] => {
  const matchedKeys = new Set<string>();
  const collectMatchedKeys = (node: NodeType) => {
    matchedKeys.add(String(node.key));
    node.children?.forEach(collectMatchedKeys);
  };
  matchedNodes.forEach(collectMatchedKeys);

  const filterNode = (node: NodeType): NodeType[] => {
    const filteredChildren = (node.children ?? []).flatMap(filterNode);
    const isMatch =
      matchedKeys.has(String(node.key)) || (showExceptions && node.events?.some((event) => event.name === 'exception'));

    if (isMatch || (showParents && filteredChildren.length > 0)) {
      return [{ ...node, children: filteredChildren }];
    }
    return filteredChildren;
  };

  return sourceNodes.flatMap(filterNode);
};

export const useServerSideTraceSearch = <NodeType = ModelTraceSpanNode>({
  modelTraceInfo,
  enabled,
  parseTraceToTree,
}: {
  modelTraceInfo: ModelTrace['info'];
  enabled: boolean;
  parseTraceToTree?: (trace: ModelTrace) => NodeType[];
}): {
  searchFilter: string;
  setSearchFilter: (filter: string) => void;
  filteredTreeNodes: NodeType[] | null;
  isSearching: boolean;
} => {
  const [searchFilter, setSearchFilter] = useState('');
  const [filteredTreeNodes, setFilteredTreeNodes] = useState<NodeType[] | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const abortRef = useRef(0);

  const traceId = useMemo(() => {
    if (isV3ModelTraceInfo(modelTraceInfo)) {
      return (modelTraceInfo as ModelTraceInfoV3).trace_id;
    }
    return undefined;
  }, [modelTraceInfo]);

  const executeSearch = useCallback(
    async (filter: string) => {
      if (!traceId || !enabled) return;
      const requestId = ++abortRef.current;

      if (!filter.trim()) {
        setFilteredTreeNodes(null);
        setIsSearching(false);
        return;
      }

      setIsSearching(true);
      try {
        const resp = await getExperimentTraceV3({
          traceId,
          filter: filter.trim(),
        });
        if (requestId !== abortRef.current) return;

        const spans = resp?.trace?.spans ?? [];
        const pseudoTrace: ModelTrace = {
          info: modelTraceInfo,
          data: { spans },
        };
        const nodes = parseTraceToTree
          ? parseTraceToTree(pseudoTrace)
          : (parseModelTraceToTreeWithMultipleRoots(pseudoTrace) as NodeType[]);
        setFilteredTreeNodes(nodes);
      } catch {
        if (requestId === abortRef.current) {
          setFilteredTreeNodes(null);
        }
      } finally {
        if (requestId === abortRef.current) {
          setIsSearching(false);
        }
      }
    },
    [traceId, enabled, modelTraceInfo, parseTraceToTree],
  );

  useEffect(() => {
    abortRef.current += 1;
    setFilteredTreeNodes(null);
    setIsSearching(false);
  }, [traceId, enabled]);

  useEffect(() => {
    if (!enabled) return;
    if (!searchFilter.trim()) {
      abortRef.current += 1;
      setFilteredTreeNodes(null);
      setIsSearching(false);
      return;
    }
    const timer = setTimeout(() => executeSearch(searchFilter), DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [searchFilter, executeSearch, enabled]);

  return {
    searchFilter,
    setSearchFilter,
    filteredTreeNodes,
    isSearching,
  };
};
