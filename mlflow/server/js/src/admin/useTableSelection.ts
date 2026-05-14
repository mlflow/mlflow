import { useCallback, useMemo, useState } from 'react';

/**
 * Selection state for a multi-row table with a "select all" header
 * checkbox. Holds selected ids in a ``Set``, but exposes a
 * ``visibleSelected`` view that drops ids no longer present in
 * ``items`` — the underlying state intentionally retains stale ids so
 * the selection isn't wiped by an unrelated refetch, but call sites
 * (counts, bulk-action handlers) read ``visibleSelected`` to avoid
 * acting on rows that already disappeared.
 *
 * ``isAllSelected`` and ``toggleAll`` are derived from the visible
 * subset for the same reason — comparing raw ``Set.size`` to
 * ``items.length`` would treat a stale-only selection as "fully
 * selected" and fail to clear on click.
 */
export const useTableSelection = <T, K extends keyof T>(items: T[], idKey: K) => {
  type Id = T[K];
  const [selectedIds, setSelectedIds] = useState<Set<Id>>(() => new Set());

  const visibleSelected = useMemo(() => {
    const present = new Set(items.map((item) => item[idKey]));
    return new Set(Array.from(selectedIds).filter((id) => present.has(id)));
  }, [selectedIds, items, idKey]);

  const isAllSelected = items.length > 0 && items.every((item) => visibleSelected.has(item[idKey]));

  const toggleItem = useCallback((id: Id) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const toggleAll = useCallback(() => {
    setSelectedIds(isAllSelected ? new Set() : new Set(items.map((item) => item[idKey])));
  }, [isAllSelected, items, idKey]);

  const clear = useCallback(() => setSelectedIds(new Set()), []);

  return { visibleSelected, isAllSelected, toggleItem, toggleAll, clear };
};
