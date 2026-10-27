import { compact } from 'lodash';
import { useCallback, useLayoutEffect, useMemo, useState } from 'react';

import type {
  ModelTraceExplorerTab,
  ModelTraceSpanNode,
  SearchMatch,
  SpanFilterState,
  ModelTrace,
} from '../ModelTrace.types';
import { SpanLogLevel } from '../ModelTrace.types';
import { getMatchesFromSpan, getSpanExceptionCount, getSpanLogLevel, searchTree } from '../ModelTraceExplorer.utils';
import {
  getSpanNodeParentIds,
  getTimelineTreeNodesList,
  getTimelineTreeNodesMap,
} from '../timeline-tree/TimelineTree.utils';

const EMPTY_COMPARISON_NODE_MAP: Record<string, ModelTraceSpanNode> = {};

const getDefaultSpanFilterState = (treeNodes: ModelTraceSpanNode[]): SpanFilterState => {
  const spanTypeDisplayState: Record<string, boolean> = {};

  // populate the spanTypeDisplayState with
  // all span types that exist on the trace
  if (treeNodes && treeNodes.length > 0) {
    const allSpanTypes = compact(getTimelineTreeNodesList<ModelTraceSpanNode>(treeNodes).map((node) => node.type));
    allSpanTypes.forEach((spanType) => {
      spanTypeDisplayState[spanType] = true;
    });
  }

  return {
    showParents: true,
    showExceptions: true,
    spanTypeDisplayState,
    // DEBUG is the lowest level, so the default threshold shows every span.
    minLogLevel: SpanLogLevel.DEBUG,
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
    case 'links':
      return 'links';
    default:
      // shouldn't happen
      return 'content';
  }
};

// Searches a merged comparison tree: a span matches when either its baseline or its candidate side matches.
// Comparison trees exist only with the new eval runs UI, so a span type absent from the filter map stays visible.
const searchComparisonTree = (
  rootNode: ModelTraceSpanNode,
  searchFilter: string,
  spanFilterState: SpanFilterState,
  comparisonNodeMap: Record<string, ModelTraceSpanNode>,
): { filteredTreeNodes: ModelTraceSpanNode[]; matches: SearchMatch[] } => {
  const searchFilterLowercased = searchFilter.toLowerCase().trim();
  const allSpanTypesSelected = Object.values(spanFilterState.spanTypeDisplayState).every(Boolean);
  const logLevelThresholdAtMin = spanFilterState.minLogLevel === SpanLogLevel.DEBUG;
  if (searchFilterLowercased === '' && allSpanTypesSelected && logLevelThresholdAtMin) {
    return { filteredTreeNodes: [rootNode], matches: [] };
  }

  const childResults = (rootNode.children ?? []).map((child) =>
    searchComparisonTree(child, searchFilterLowercased, spanFilterState, comparisonNodeMap),
  );
  const filteredChildren = childResults.flatMap(({ filteredTreeNodes }) => filteredTreeNodes);
  const childMatches = childResults.flatMap(({ matches }) => matches);
  const baselineMatches = getMatchesFromSpan(rootNode, searchFilterLowercased);
  const comparisonNode = comparisonNodeMap[String(rootNode.key)];
  const baselineMatchKeys = new Set(
    baselineMatches.map(
      ({ section, key, isKeyMatch, matchIndex }) => `${section}:${key}:${String(isKeyMatch)}:${matchIndex}`,
    ),
  );
  // The Links tab shows only the displayed (baseline) span's links, so candidate link matches can't be shown.
  const comparisonMatches = comparisonNode
    ? getMatchesFromSpan(comparisonNode, searchFilterLowercased)
        .filter(
          ({ section, key, isKeyMatch, matchIndex }) =>
            section !== 'links' && !baselineMatchKeys.has(`${section}:${key}:${String(isKeyMatch)}:${matchIndex}`),
        )
        .map((match) => ({ ...match, span: rootNode }))
    : [];
  const spanMatches = [...baselineMatches, ...comparisonMatches];
  const spanName = String(rootNode.title ?? '').toLowerCase();
  const comparisonSpanName = String(comparisonNode?.title ?? '').toLowerCase();
  const nodeMatchesSearch =
    spanMatches.length > 0 ||
    spanName.includes(searchFilterLowercased) ||
    comparisonSpanName.includes(searchFilterLowercased);
  const spanTypeIsDisplayed = rootNode.type ? (spanFilterState.spanTypeDisplayState[rootNode.type] ?? true) : true;
  const spanPassesLogLevel =
    getSpanLogLevel(rootNode) >= spanFilterState.minLogLevel ||
    (comparisonNode ? getSpanLogLevel(comparisonNode) >= spanFilterState.minLogLevel : false);
  const nodePassesSpanFilters = nodeMatchesSearch && spanTypeIsDisplayed && spanPassesLogLevel;
  const hasException =
    getSpanExceptionCount(rootNode) > 0 || (comparisonNode ? getSpanExceptionCount(comparisonNode) > 0 : false);
  const nodeShouldBeDisplayed =
    nodePassesSpanFilters ||
    (spanFilterState.showParents && filteredChildren.length > 0) ||
    (spanFilterState.showExceptions && hasException);

  return nodeShouldBeDisplayed
    ? {
        filteredTreeNodes: [{ ...rootNode, children: filteredChildren }],
        matches: [...spanMatches, ...childMatches],
      }
    : { filteredTreeNodes: filteredChildren, matches: childMatches };
};

export const useModelTraceSearch = ({
  treeNodes,
  selectedNode,
  setSelectedNode,
  setSelectedNodeAndTab,
  setExpandedKeys,
  modelTraceInfo,
  comparisonNodeMap = EMPTY_COMPARISON_NODE_MAP,
}: {
  treeNodes: ModelTraceSpanNode[];
  selectedNode: ModelTraceSpanNode | undefined;
  setSelectedNode: (node: ModelTraceSpanNode) => void;
  setSelectedNodeAndTab: (node: ModelTraceSpanNode, tab: ModelTraceExplorerTab) => void;
  setExpandedKeys: React.Dispatch<React.SetStateAction<Set<string | number>>>;
  modelTraceInfo: ModelTrace['info'] | null;
  comparisonNodeMap?: Record<string, ModelTraceSpanNode>;
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
  const [spanFilterState, setSpanFilterState] = useState<SpanFilterState>(() => getDefaultSpanFilterState(treeNodes));
  const [activeMatchIndex, setActiveMatchIndex] = useState(0);
  const treeNodesKeys = useMemo(() => treeNodes.map((n) => n.key).join(','), [treeNodes]);
  const { filteredTreeNodes, matches } = useMemo(() => {
    if (!treeNodes || treeNodes.length === 0) {
      return {
        filteredTreeNodes: [],
        matches: [],
      };
    }

    // Run search over each root and merge results
    const hasComparisonNodes = Object.keys(comparisonNodeMap).length > 0;
    const merged = treeNodes.map((root) =>
      hasComparisonNodes
        ? searchComparisonTree(root, searchFilter, spanFilterState, comparisonNodeMap)
        : searchTree(root, searchFilter, spanFilterState),
    );
    return {
      filteredTreeNodes: merged.flatMap((r) => r.filteredTreeNodes),
      matches: merged.flatMap((r) => r.matches),
    };
    // use the span IDs to determine whether the state should be recomputed.
    // using the whole object seems to cause the state to be reset at
    // unexpected times.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [treeNodesKeys, searchFilter, spanFilterState, modelTraceInfo, comparisonNodeMap]);

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
      setSelectedNodeAndTab(match.span, getTabForMatch(match));
      // Make sure parents are expanded
      const parents = getSpanNodeParentIds(match.span, nodeMap);
      setExpandedKeys((expandedKeys) => {
        // set.union seems to not be available in all environments
        return new Set([...expandedKeys, ...parents]);
      });
    },
    [matches, setSelectedNodeAndTab, nodeMap, setExpandedKeys],
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
        setSelectedNodeAndTab(newSpan, newSpan?.chatMessages ? 'chat' : 'content');
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
    setSelectedNodeAndTab(matches[0].span, getTabForMatch(matches[0]));
    // don't subscribe to selectedNode to prevent infinite loop
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filteredTreeNodes, matches, setSelectedNodeAndTab]);

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
