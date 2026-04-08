import { useState, useCallback, useMemo } from 'react';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { SpanRange, SpanSelector } from '../hooks/useTraceViews';
import { isSpanInRange, spanMatchesSelector } from '../hooks/useTraceViewFiltering';
import { getRangeColor } from './rangeColors';

/**
 * Finds which range (if any) a span belongs to, based on tree-order position
 * between from_selector and to_selector.
 */
const findRangeForSpan = (
  node: ModelTraceSpanNode,
  nodes: ModelTraceSpanNode[],
  ranges: SpanRange[],
): number | null => {
  for (let rangeIdx = 0; rangeIdx < ranges.length; rangeIdx++) {
    if (isSpanInRange(node, nodes, ranges[rangeIdx])) return rangeIdx;
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

export interface RangeEditDraft {
  rangeIdx: number;
  from_selector: SpanSelector;
  to_selector: SpanSelector | null;
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
  /** Draft state when editing an existing range's selection, or null */
  editDraft: RangeEditDraft | null;
  /** Start editing a range's from/to selectors */
  startEditingRange: (rangeIdx: number) => void;
  /** Confirm the draft edit and apply it to the range */
  confirmEditDraft: () => void;
  /** Cancel the draft edit and discard changes */
  cancelEditDraft: () => void;
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
  onUpdateRange: (index: number, updates: Partial<SpanRange>) => void,
): SpanRangeSelectionResult => {
  const [expandedSpanId, setExpandedSpanId] = useState<string | null>(null);
  const [dragState, setDragState] = useState<{ startIdx: number; currentIdx: number } | null>(null);
  const [editDraft, setEditDraft] = useState<RangeEditDraft | null>(null);

  const spanKeyToFlatIndex = useMemo(() => {
    const map = new Map<string, number>();
    for (let i = 0; i < nodes.length; i++) {
      map.set(String(nodes[i].key), i);
    }
    return map;
  }, [nodes]);

  // Build a virtual ranges array that overlays the draft on top of the real ranges,
  // so that the tree preview reflects the pending edit.
  const effectiveRanges = useMemo(() => {
    if (!editDraft) return ranges;
    return ranges.map((r, i) =>
      i === editDraft.rangeIdx
        ? { ...r, from_selector: editDraft.from_selector, to_selector: editDraft.to_selector }
        : r,
    );
  }, [ranges, editDraft]);

  const spanRangeMap = useMemo(() => {
    const map = new Map<string, number>();
    for (const node of nodes) {
      const rangeIdx = findRangeForSpan(node, nodes, effectiveRanges);
      if (rangeIdx !== null) map.set(String(node.key), rangeIdx);
    }
    return map;
  }, [nodes, effectiveRanges]);

  const rangeFirstSpanId = useMemo(() => {
    const result = new Map<number, string>();
    for (let rangeIdx = 0; rangeIdx < effectiveRanges.length; rangeIdx++) {
      const fromSelector = effectiveRanges[rangeIdx].from_selector;
      const match = nodes.find((n) => spanMatchesSelector(n, fromSelector));
      if (match) result.set(rangeIdx, String(match.key));
    }
    return result;
  }, [effectiveRanges, nodes]);

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
      if (editDraft) {
        // Find the current draft range bounds in flat index space
        const fromIdx = nodes.findIndex((n) => spanMatchesSelector(n, editDraft.from_selector));
        const toIdx = editDraft.to_selector
          ? nodes.findIndex((n) => spanMatchesSelector(n, editDraft.to_selector))
          : fromIdx;
        const currentMin = Math.min(fromIdx >= 0 ? fromIdx : flatIndex, toIdx >= 0 ? toIdx : flatIndex);
        const currentMax = Math.max(fromIdx >= 0 ? fromIdx : flatIndex, toIdx >= 0 ? toIdx : flatIndex);

        const isInsideRange = flatIndex >= currentMin && flatIndex <= currentMax;

        let newMin: number;
        let newMax: number;

        if (isInsideRange) {
          // Shrink: clicked span is inside range, contract from the nearer end
          const distFromStart = flatIndex - currentMin;
          const distFromEnd = currentMax - flatIndex;
          if (distFromStart <= distFromEnd) {
            // Closer to start: move from forward past the clicked span
            newMin = flatIndex + 1;
            newMax = currentMax;
          } else {
            // Closer to end: move to backward past the clicked span
            newMin = currentMin;
            newMax = flatIndex - 1;
          }
        } else {
          // Expand to include the clicked span
          newMin = Math.min(currentMin, flatIndex);
          newMax = Math.max(currentMax, flatIndex);
        }

        if (newMin > newMax) {
          // Range collapsed entirely — reset to just the clicked span
          setEditDraft((prev) =>
            prev ? { ...prev, from_selector: { span_id: spanId }, to_selector: null } : prev,
          );
        } else if (newMin === newMax) {
          setEditDraft((prev) =>
            prev
              ? { ...prev, from_selector: { span_id: String(nodes[newMin].key) }, to_selector: null }
              : prev,
          );
        } else {
          setEditDraft((prev) =>
            prev
              ? {
                  ...prev,
                  from_selector: { span_id: String(nodes[newMin].key) },
                  to_selector: { span_id: String(nodes[newMax].key) },
                }
              : prev,
          );
        }
      } else {
        const existingRange = spanRangeMap.get(spanId);
        if (existingRange !== undefined) {
          onRemoveRange(existingRange);
        } else {
          onAddRange({ span_id: spanId });
        }
      }
    },
    [nodes, spanRangeMap, onAddRange, onRemoveRange, editDraft],
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
        const fromSelector: SpanSelector = { span_id: String(fromNode.key) };
        const toSelector: SpanSelector = { span_id: String(toNode.key) };
        if (editDraft) {
          // Merge dragged range with existing draft bounds
          const existingFromIdx = nodes.findIndex((n) => spanMatchesSelector(n, editDraft.from_selector));
          const existingToIdx = editDraft.to_selector
            ? nodes.findIndex((n) => spanMatchesSelector(n, editDraft.to_selector))
            : existingFromIdx;
          const newMin = Math.min(fromIdx, existingFromIdx >= 0 ? existingFromIdx : fromIdx);
          const newMax = Math.max(toIdx, existingToIdx >= 0 ? existingToIdx : toIdx);
          setEditDraft((prev) =>
            prev
              ? {
                  ...prev,
                  from_selector: { span_id: String(nodes[newMin].key) },
                  to_selector: newMin === newMax ? null : { span_id: String(nodes[newMax].key) },
                }
              : prev,
          );
        } else {
          onAddRange(fromSelector, toSelector);
        }
      }
      setDragState(null);
    }
  }, [dragState, nodes, onAddRange, editDraft]);

  const startEditingRange = useCallback(
    (rangeIdx: number) => {
      const range = ranges[rangeIdx];
      setEditDraft({
        rangeIdx,
        from_selector: { ...range.from_selector },
        to_selector: range.to_selector ? { ...range.to_selector } : null,
      });
    },
    [ranges],
  );

  const confirmEditDraft = useCallback(() => {
    if (editDraft) {
      onUpdateRange(editDraft.rangeIdx, {
        from_selector: editDraft.from_selector,
        to_selector: editDraft.to_selector,
      });
      setEditDraft(null);
    }
  }, [editDraft, onUpdateRange]);

  const cancelEditDraft = useCallback(() => {
    setEditDraft(null);
  }, []);

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
    editDraft,
    startEditingRange,
    confirmEditDraft,
    cancelEditDraft,
  };
};
