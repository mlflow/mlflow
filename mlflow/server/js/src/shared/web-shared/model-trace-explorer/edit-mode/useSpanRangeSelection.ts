import { useState, useCallback, useMemo } from 'react';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { SpanRange, SpanSelector } from '../hooks/useTraceViews';
import { getRangeColor } from './rangeColors';

/**
 * Finds which range (if any) a span belongs to, based on tree-order position
 * between from_selector and to_selector.
 */
const findRangeForSpan = (
  spanId: string,
  nodes: ModelTraceSpanNode[],
  ranges: SpanRange[],
): number | null => {
  for (let rangeIdx = 0; rangeIdx < ranges.length; rangeIdx++) {
    const range = ranges[rangeIdx];
    const fromId = range.from_selector.span_id;
    const toId = range.to_selector?.span_id;

    if (!toId) {
      if (spanId === fromId) return rangeIdx;
    } else {
      let inRange = false;
      for (const node of nodes) {
        const nodeId = String(node.key);
        if (nodeId === fromId) inRange = true;
        if (inRange && nodeId === spanId) return rangeIdx;
        if (nodeId === toId && inRange) break;
      }
    }
  }
  return null;
};

export interface SpanEditState {
  spanId: string;
  rangeIdx: number | undefined;
  inRange: boolean;
  color: ReturnType<typeof getRangeColor> | null;
  isFirstInRange: boolean;
  inDrag: boolean;
  isDimmed: boolean;
}

export interface SpanRangeSelectionResult {
  /** Get the edit state for a node at a given flat index */
  getNodeEditState: (flatIndex: number) => SpanEditState;
  /** Map from span key to flat index in the nodes array */
  spanKeyToFlatIndex: Map<string, number>;
  /** Handle checkbox click on a node */
  handleCheckboxClick: (flatIndex: number) => void;
  /** Handle pointer down for drag-to-select */
  handlePointerDown: (flatIndex: number) => void;
  /** Handle pointer move for drag-to-select */
  handlePointerMove: (flatIndex: number) => void;
  /** Handle pointer up to finalize drag-to-select */
  handlePointerUp: () => void;
  /** Currently expanded span ID for JSON field selector */
  expandedSpanId: string | null;
  /** Toggle the JSON field selector for a span */
  toggleExpandedSpan: (spanId: string) => void;
}

/**
 * Hook that extracts span range selection logic for use in any span-rendering
 * component. Provides selection state, drag-to-select, and per-node edit state
 * without prescribing how spans are rendered.
 */
export const useSpanRangeSelection = (
  nodes: ModelTraceSpanNode[],
  ranges: SpanRange[],
  onAddRange: (from: SpanSelector, to?: SpanSelector) => void,
  onRemoveRange: (index: number) => void,
): SpanRangeSelectionResult => {
  const [expandedSpanId, setExpandedSpanId] = useState<string | null>(null);
  const [dragState, setDragState] = useState<{ startIdx: number; currentIdx: number } | null>(null);

  const spanKeyToFlatIndex = useMemo(() => {
    const map = new Map<string, number>();
    for (let i = 0; i < nodes.length; i++) {
      map.set(String(nodes[i].key), i);
    }
    return map;
  }, [nodes]);

  const spanRangeMap = useMemo(() => {
    const map = new Map<string, number>();
    for (const node of nodes) {
      const rangeIdx = findRangeForSpan(String(node.key), nodes, ranges);
      if (rangeIdx !== null) map.set(String(node.key), rangeIdx);
    }
    return map;
  }, [nodes, ranges]);

  const rangeFirstSpanId = useMemo(() => {
    const result = new Map<number, string>();
    for (let rangeIdx = 0; rangeIdx < ranges.length; rangeIdx++) {
      const fromId = ranges[rangeIdx].from_selector.span_id;
      if (fromId) result.set(rangeIdx, fromId);
    }
    return result;
  }, [ranges]);

  const isDragging = dragState !== null && dragState.startIdx !== dragState.currentIdx;
  const dragMin = dragState ? Math.min(dragState.startIdx, dragState.currentIdx) : -1;
  const dragMax = dragState ? Math.max(dragState.startIdx, dragState.currentIdx) : -1;

  const getNodeEditState = useCallback(
    (flatIndex: number): SpanEditState => {
      const node = nodes[flatIndex];
      const spanId = String(node.key);
      const rangeIdx = spanRangeMap.get(spanId);
      const inRange = rangeIdx !== undefined;
      const color = inRange ? getRangeColor(rangeIdx) : null;
      const isFirstInRange = inRange && rangeFirstSpanId.get(rangeIdx) === spanId;
      const inDrag = isDragging && flatIndex >= dragMin && flatIndex <= dragMax;
      const isDimmed = !inRange && !inDrag;
      return { spanId, rangeIdx, inRange, color, isFirstInRange, inDrag, isDimmed };
    },
    [nodes, spanRangeMap, rangeFirstSpanId, isDragging, dragMin, dragMax],
  );

  const handleCheckboxClick = useCallback(
    (flatIndex: number) => {
      const node = nodes[flatIndex];
      const spanId = String(node.key);
      const existingRange = spanRangeMap.get(spanId);
      if (existingRange !== undefined) {
        onRemoveRange(existingRange);
      } else {
        onAddRange({ span_id: spanId });
      }
    },
    [nodes, spanRangeMap, onAddRange, onRemoveRange],
  );

  const handlePointerDown = useCallback((flatIndex: number) => {
    setDragState({ startIdx: flatIndex, currentIdx: flatIndex });
  }, []);

  const handlePointerMove = useCallback(
    (flatIndex: number) => {
      if (dragState) {
        setDragState((prev) => (prev ? { ...prev, currentIdx: flatIndex } : prev));
      }
    },
    [dragState],
  );

  const handlePointerUp = useCallback(() => {
    if (dragState) {
      const { startIdx, currentIdx } = dragState;
      if (startIdx !== currentIdx) {
        const fromIdx = Math.min(startIdx, currentIdx);
        const toIdx = Math.max(startIdx, currentIdx);
        const fromNode = nodes[fromIdx];
        const toNode = nodes[toIdx];
        onAddRange({ span_id: String(fromNode.key) }, { span_id: String(toNode.key) });
      }
      setDragState(null);
    }
  }, [dragState, nodes, onAddRange]);

  const toggleExpandedSpan = useCallback(
    (spanId: string) => {
      setExpandedSpanId((prev) => (prev === spanId ? null : spanId));
    },
    [],
  );

  return {
    getNodeEditState,
    spanKeyToFlatIndex,
    handleCheckboxClick,
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    expandedSpanId,
    toggleExpandedSpan,
  };
};
