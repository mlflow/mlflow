import { useCallback, useMemo, useState } from 'react';
import type { ModelTraceInfoV3 } from '../../model-trace-explorer/ModelTrace.types';

export interface UseBulkTraceSelectionResult {
  /**
   * Currently-selected traces, keyed on `trace_id`. Stores the full `ModelTraceInfoV3` (not just the
   * id) so bulk actions get their exact expected input for the *entire* cross-page selection — not
   * only the rows that happen to be on the current page. Insertion order = selection order. Persists
   * across pages until explicitly cleared.
   */
  selected: Map<string, ModelTraceInfoV3>;
  /** True iff every visible (current-page) trace is selected. */
  isAllVisibleChecked: boolean;
  /** True iff some — but not all — visible traces are selected (drives the header's indeterminate UI). */
  isSomeVisibleChecked: boolean;
  /** Toggle one trace, or the inclusive visible range from the previous anchor when selectRange is true. */
  toggle: (trace: ModelTraceInfoV3, selectRange?: boolean) => void;
  /** Select every supplied trace unless all are selected, in which case clear all supplied traces. */
  toggleMany: (traces: ModelTraceInfoV3[]) => void;
  /** Select all visible traces when none/some are checked; clear them when all are checked. */
  toggleAll: () => void;
  clear: () => void;
}

/**
 * Bulk-selection state for the traces table, keyed on `trace_id`.
 *
 * Selection is filter-scoped, not page-scoped: it persists across pagination so users can
 * accumulate a selection across pages, and the consumer clears it when the filter/search changes.
 * `isAllVisibleChecked` / `toggleAll` operate only on the currently-visible page's traces.
 *
 * The store holds the full `ModelTraceInfoV3` per selected trace so downstream bulk actions operate
 * on the complete cross-page selection with the input shape they already expect — no id-parsing, no
 * re-fetch.
 */
export const useBulkTraceSelection = (visibleTraces: ModelTraceInfoV3[]): UseBulkTraceSelectionResult => {
  const [{ selected }, setSelection] = useState<{
    selected: Map<string, ModelTraceInfoV3>;
    selectionAnchorTraceId: string | null;
  }>(() => ({ selected: new Map(), selectionAnchorTraceId: null }));

  const toggle = useCallback(
    (trace: ModelTraceInfoV3, selectRange = false) => {
      setSelection((prev) => {
        // Shift-click extends from the last-toggled anchor to the clicked row across the visible page;
        // a plain click toggles just that row (and re-anchors it).
        const anchorIndex = selectRange
          ? visibleTraces.findIndex((visibleTrace) => visibleTrace.trace_id === prev.selectionAnchorTraceId)
          : -1;
        const targetIndex = selectRange
          ? visibleTraces.findIndex((visibleTrace) => visibleTrace.trace_id === trace.trace_id)
          : -1;
        const tracesToToggle =
          anchorIndex >= 0 && targetIndex >= 0
            ? visibleTraces.slice(Math.min(anchorIndex, targetIndex), Math.max(anchorIndex, targetIndex) + 1)
            : [trace];
        const isDeselecting = prev.selected.has(trace.trace_id);
        const next = new Map(prev.selected);
        tracesToToggle.forEach((rangeTrace) => {
          if (isDeselecting) {
            next.delete(rangeTrace.trace_id);
          } else {
            next.set(rangeTrace.trace_id, rangeTrace);
          }
        });
        return {
          selected: next,
          selectionAnchorTraceId: next.size === 0 ? null : trace.trace_id,
        };
      });
    },
    [visibleTraces],
  );

  const toggleMany = useCallback((traces: ModelTraceInfoV3[]) => {
    setSelection((prev) => {
      const allChecked = traces.length > 0 && traces.every((trace) => prev.selected.has(trace.trace_id));
      const next = new Map(prev.selected);
      traces.forEach((trace) => {
        if (allChecked) {
          next.delete(trace.trace_id);
        } else {
          next.set(trace.trace_id, trace);
        }
      });
      return {
        selected: next,
        selectionAnchorTraceId: next.size === 0 ? null : prev.selectionAnchorTraceId,
      };
    });
  }, []);

  const toggleAll = useCallback(() => {
    setSelection((prev) => {
      const allChecked = visibleTraces.length > 0 && visibleTraces.every((trace) => prev.selected.has(trace.trace_id));
      const next = new Map(prev.selected);
      if (allChecked) {
        visibleTraces.forEach((trace) => next.delete(trace.trace_id));
      } else {
        visibleTraces.forEach((trace) => next.set(trace.trace_id, trace));
      }
      return {
        selected: next,
        selectionAnchorTraceId: next.size === 0 ? null : prev.selectionAnchorTraceId,
      };
    });
  }, [visibleTraces]);

  const clear = useCallback(() => {
    setSelection((prev) =>
      prev.selected.size === 0 && prev.selectionAnchorTraceId === null
        ? prev
        : { selected: new Map(), selectionAnchorTraceId: null },
    );
  }, []);

  const { isAllVisibleChecked, isSomeVisibleChecked } = useMemo(() => {
    if (visibleTraces.length === 0) {
      return { isAllVisibleChecked: false, isSomeVisibleChecked: false };
    }
    const checkedCount = visibleTraces.reduce((count, trace) => (selected.has(trace.trace_id) ? count + 1 : count), 0);
    return {
      isAllVisibleChecked: checkedCount === visibleTraces.length,
      isSomeVisibleChecked: checkedCount > 0 && checkedCount < visibleTraces.length,
    };
  }, [visibleTraces, selected]);

  return { selected, isAllVisibleChecked, isSomeVisibleChecked, toggle, toggleMany, toggleAll, clear };
};
