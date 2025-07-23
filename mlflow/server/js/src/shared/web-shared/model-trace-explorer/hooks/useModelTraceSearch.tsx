import { compact, isNil } from 'lodash';
import { useCallback, useLayoutEffect, useMemo, useState } from 'react';

import type {
  ModelTraceExplorerTab,
  ModelTraceSpanNode,
  SearchMatch,
  SpanFilterState,
  ModelTrace,
} from '../ModelTrace.types';
import { searchTree } from '../ModelTraceExplorer.utils';
import {
  getSpanNodeParentIds,
  getTimelineTreeNodesList,
  getTimelineTreeNodesMap,
} from '../timeline-tree/TimelineTree.utils';

const getDefaultSpanFilterState = (treeNode: ModelTraceSpanNode | null): SpanFilterState => {
  const spanTypeDisplayState: Record<string, boolean> = {};

  // populate the spanTypeDisplayState with
  // all span types that exist on the trace
  if (treeNode) {
    const allSpanTypes = compact(getTimelineTreeNodesList<ModelTraceSpanNode>([treeNode]).map((node) => node.type));
    allSpanTypes.forEach((spanType) => {
      spanTypeDisplayState[spanType] = true;
    });
  }

  return {
    showParents: true,
    showExceptions: true,
    spanTypeDisplayState,
  };
};

const getTabForMatch = (match: SearchMatch): ModelTraceExplorerTab => {
  switch (match.section) {
    case 'inputs':
    case 'outputs':
      return 'content';
    case 'attributes':
      return 'attributes';
    case 'events':
      return 'events';
    default:
      // shouldn't happen
      return 'content';
  }
};

export const useModelTraceSearch = ({
  treeNode,
  selectedNode,
  setSelectedNode,
  setActiveTab,
  setExpandedKeys,
  modelTraceInfo,
}: {
  treeNode: ModelTraceSpanNode | null;
  selectedNode: ModelTraceSpanNode | undefined;
  setSelectedNode: (node: ModelTraceSpanNode) => void;
  setActiveTab: (tab: ModelTraceExplorerTab) => void;
  setExpandedKeys: React.Dispatch<React.SetStateAction<Set<string | number>>>;
  modelTraceInfo: ModelTrace['info'] | null;
}): {
  searchFilter: string;
  setSearchFilter: (filter: string) => void;
  spanFilterState: SpanFilterState;
  setSpanFilterState: (state: SpanFilterState) => void;
  filteredTreeNodes: ModelTraceSpanNode[];
  matchData: {
    match: SearchMatch | null;
    totalMatches: number;
    currentMatchIndex: number;
  };
  handleNextSearchMatch: () => void;
  handlePreviousSearchMatch: () => void;
} => {
  const [searchFilter, setSearchFilter] = useState<string>('');
  const [spanFilterState, setSpanFilterState] = useState<SpanFilterState>(() => getDefaultSpanFilterState(treeNode));
  const [activeMatchIndex, setActiveMatchIndex] = useState(0);
  const { filteredTreeNodes, matches } = useMemo(() => {
    if (isNil(treeNode)) {
      return {
        filteredTreeNodes: [],
        matches: [],
      };
    }

    return searchTree(treeNode, searchFilter, spanFilterState);
    // use the span ID to determine whether the state should be recomputed.
    // using the whole object seems to cause the state to be reset at
    // unexpected times.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [treeNode?.key, searchFilter, spanFilterState, modelTraceInfo]);

  const nodeMap = useMemo(() => {
    return getTimelineTreeNodesMap(filteredTreeNodes);
  }, [filteredTreeNodes]);

  const selectMatch = useCallback(
    (newMatchIndex: number) => {
      if (newMatchIndex >= matches.length || newMatchIndex < 0) {
        return;
      }
      setActiveMatchIndex(newMatchIndex);
      const match = matches[newMatchIndex];
      setSelectedNode(match.span);
      setActiveTab(getTabForMatch(match));
      // Make sure parents are expanded
      const parents = getSpanNodeParentIds(match.span, nodeMap);
      setExpandedKeys((expandedKeys) => {
        // set.union seems to not be available in all environments
        return new Set([...expandedKeys, ...parents]);
      });
    },
    [matches, setSelectedNode, setActiveTab, nodeMap, setExpandedKeys],
  );

  const handleNextSearchMatch = useCallback(() => {
    selectMatch(activeMatchIndex + 1);
  }, [activeMatchIndex, selectMatch]);

  const handlePreviousSearchMatch = useCallback(() => {
    selectMatch(activeMatchIndex - 1);
  }, [activeMatchIndex, selectMatch]);

  useLayoutEffect(() => {
    if (filteredTreeNodes.length === 0) {
      return;
    }

    // this case can trigger on two conditions:
    // 1. the search term is cleared, therefore there are no matches
    // 2. the search term only matches on span names, which don't count
    //    as matches since we don't support jumping to them.
    if (matches.length === 0) {
      // if the selected node is no longer in the tree, then select
      // the first node. this can occur from condition #2 above
      const selectedNodeKey = selectedNode?.key ?? '';
      if (!(selectedNodeKey in nodeMap)) {
        const newSpan = filteredTreeNodes[0];
        setSelectedNode(newSpan);
        setActiveTab(newSpan?.chatMessages ? 'chat' : 'content');
      } else {
        // another reason the tree can change is if modelTraceInfo changes.
        // (e.g. tags/assessments were updated). if this happens, we need
        // to reselect the updated node from the node map, otherwise the
        // updates will not be reflected in the UI.
        setSelectedNode(nodeMap[selectedNodeKey]);
      }

      // otherwise, if search was cleared, then we don't want to
      // do anything. this is to preserve the user's context
      // (e.g. they've jumped to a span and now want to dive deeper)
      return;
    }

    // when matches update, select the first match
    setActiveMatchIndex(0);
    setSelectedNode(matches[0].span);
    setActiveTab(getTabForMatch(matches[0]));
    // don't subscribe to selectedNode to prevent infinite loop
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filteredTreeNodes, matches, setSelectedNode]);

  return {
    matchData: {
      match: matches[activeMatchIndex] ?? null,
      totalMatches: matches.length,
      currentMatchIndex: activeMatchIndex,
    },
    searchFilter: searchFilter.toLowerCase().trim(),
    setSearchFilter,
    spanFilterState,
    setSpanFilterState,
    filteredTreeNodes,
    handleNextSearchMatch,
    handlePreviousSearchMatch,
  };
};
