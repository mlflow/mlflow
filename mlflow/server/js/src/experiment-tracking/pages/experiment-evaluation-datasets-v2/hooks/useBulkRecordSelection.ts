import { useCallback, useMemo, useState } from 'react';

interface UseBulkRecordSelectionResult {
  selected: Set<string>;
  /** True iff every id in `visibleIds` is currently selected. */
  isAllVisibleChecked: boolean;
  /** True if some but not all visible rows are checked — used to render the indeterminate header. */
  isSomeVisibleChecked: boolean;
  toggle: (recordId: string) => void;
  /** Adds all visible ids to the selection when none-or-some are checked; clears them when all are checked. */
  toggleAll: () => void;
  clear: () => void;
}

/**
 * Tracks bulk-delete selection for the records table as a `Set<string>` of record ids.
 *
 * Selection is filter-scoped, not page-scoped: callers reset it via `clear()` on search
 * changes (selected records that no longer match the filter would be invisible and easy to
 * mis-delete), but it persists across pagination so users can accumulate selections across
 * pages. `isAllVisibleChecked` / `toggleAll` operate on the currently-visible page only.
 */
export const useBulkRecordSelection = (visibleIds: string[]): UseBulkRecordSelectionResult => {
  const [selected, setSelected] = useState<Set<string>>(() => new Set());

  const toggle = useCallback((recordId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(recordId)) {
        next.delete(recordId);
      } else {
        next.add(recordId);
      }
      return next;
    });
  }, []);

  const toggleAll = useCallback(() => {
    setSelected((prev) => {
      const allChecked = visibleIds.length > 0 && visibleIds.every((id) => prev.has(id));
      const next = new Set(prev);
      if (allChecked) {
        visibleIds.forEach((id) => next.delete(id));
      } else {
        visibleIds.forEach((id) => next.add(id));
      }
      return next;
    });
  }, [visibleIds]);

  const clear = useCallback(() => {
    setSelected((prev) => (prev.size === 0 ? prev : new Set()));
  }, []);

  const { isAllVisibleChecked, isSomeVisibleChecked } = useMemo(() => {
    if (visibleIds.length === 0) {
      return { isAllVisibleChecked: false, isSomeVisibleChecked: false };
    }
    let checkedCount = 0;
    for (const id of visibleIds) {
      if (selected.has(id)) checkedCount += 1;
    }
    return {
      isAllVisibleChecked: checkedCount === visibleIds.length,
      isSomeVisibleChecked: checkedCount > 0 && checkedCount < visibleIds.length,
    };
  }, [visibleIds, selected]);

  return { selected, isAllVisibleChecked, isSomeVisibleChecked, toggle, toggleAll, clear };
};
