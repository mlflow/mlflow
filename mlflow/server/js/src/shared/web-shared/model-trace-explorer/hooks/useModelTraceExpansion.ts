import type { Dispatch, SetStateAction } from 'react';
import { useLayoutEffect, useMemo, useRef } from 'react';

import type { ModelTraceSpanNode, SpanFilterState } from '../ModelTrace.types';
import { DEFAULT_EXPAND_DEPTH, getTimelineTreeNodesMap } from '../timeline-tree/TimelineTree.utils';

/** Reconciles topology; changes to trace ID, search text, or span-filter object identity reset expansion. */
export const useModelTraceExpansion = (
  treeNodes: ModelTraceSpanNode[],
  traceId: string | undefined,
  searchFilter: string,
  spanFilterState: SpanFilterState,
  setExpandedKeys: Dispatch<SetStateAction<Set<string | number>>>,
) => {
  const keys = useMemo(
    () =>
      new Set<string | number>(
        Object.values(getTimelineTreeNodesMap(treeNodes, DEFAULT_EXPAND_DEPTH)).map((node) => node.key),
      ),
    [treeNodes],
  );
  const previous = useRef<{
    traceId?: string;
    searchFilter: string;
    spanFilterState: SpanFilterState;
    keys: typeof keys;
  }>();
  useLayoutEffect(() => {
    const before = previous.current;
    previous.current = { traceId, searchFilter, spanFilterState, keys };
    setExpandedKeys((expanded) => {
      // A new trace or deliberate filter change starts from the visible defaults.
      const reset =
        !before ||
        before.traceId !== traceId ||
        before.searchFilter !== searchFilter ||
        before.spanFilterState !== spanFilterState;
      const next = reset ? new Set(keys) : new Set([...expanded].filter((key) => keys.has(key)));
      if (!reset) {
        keys.forEach((key) => {
          if (!before.keys.has(key)) next.add(key);
        });
      }
      return next.size === expanded.size && [...next].every((key) => expanded.has(key)) ? expanded : next;
    });
  }, [keys, traceId, searchFilter, spanFilterState, setExpandedKeys]);
};
