import { defaultRangeExtractor, useVirtualizer } from '@tanstack/react-virtual';
import type { Range } from '@tanstack/react-virtual';
import { useCallback, useMemo } from 'react';

import type { ModelTraceSpanNode } from '../ModelTrace.types';

export const TIMELINE_TREE_VIRTUALIZATION_MIN_ROWS = 100;

export const mergePinnedRowIndex = (range: Range, pinnedIndex: number): number[] => {
  const indexes = defaultRangeExtractor(range);
  if (pinnedIndex < 0 || indexes.includes(pinnedIndex)) return indexes;
  return [...indexes, pinnedIndex].sort((left, right) => left - right);
};

export const useTimelineTreeRowVirtualizer = (
  nodes: ModelTraceSpanNode[],
  scrollElement: HTMLElement | null,
  selectedKey: string | number,
  rowHeight: number,
) => {
  const selectedIndex = useMemo(() => nodes.findIndex((node) => node.key === selectedKey), [nodes, selectedKey]);
  const rangeExtractor = useCallback((range: Range) => mergePinnedRowIndex(range, selectedIndex), [selectedIndex]);
  return useVirtualizer({
    count: nodes.length,
    getScrollElement: () => scrollElement,
    estimateSize: () => rowHeight,
    getItemKey: (index) => nodes[index].key,
    rangeExtractor,
    overscan: 8,
    enabled: nodes.length >= TIMELINE_TREE_VIRTUALIZATION_MIN_ROWS,
  });
};
