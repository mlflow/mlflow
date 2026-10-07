import { useCallback, useMemo, type Dispatch, type SetStateAction } from 'react';
import { useLocalStorage } from '../../hooks/useLocalStorage';
import { isTraceColumnId, TRACE_COLUMN_IDS } from '../constants';
import type { TraceColumnId } from '../types';

/** Per-column visibility overrides. A column absent from the map falls back to its computed default. */
type ColumnOverrides = Record<string, boolean>;

/**
 * Coerce a stored value into a complete, deduped column order: keep valid ids in their stored order,
 * then append any canonical columns the stored order is missing (so a column added in a newer release
 * still appears). Invalid/unknown ids are dropped.
 */
const normalizeColumnOrder = (storedOrder: unknown): TraceColumnId[] => {
  const seen = new Set<string>();
  const normalizedOrder: TraceColumnId[] = [];
  const candidateOrder = Array.isArray(storedOrder) ? storedOrder : [];

  for (const id of [...candidateOrder, ...TRACE_COLUMN_IDS]) {
    if (typeof id === 'string' && isTraceColumnId(id) && !seen.has(id)) {
      normalizedOrder.push(id);
      seen.add(id);
    }
  }

  return normalizedOrder;
};

export interface UseTraceColumnVisibilityParams {
  /** localStorage key (the consumer scopes it, e.g. per experiment). */
  storageKey: string;
  /** Bump when the stored schema/default set changes so stale entries reset. */
  version: number;
  /** Computes a column's default visibility (may depend on live data the consumer knows about). */
  getDefaultVisible: (id: TraceColumnId) => boolean;
  /** Recognizes consumer-defined dynamic column ids stored alongside the standard overrides. */
  isDynamicColumnId?: (id: string) => boolean;
}

export interface TraceColumnVisibility {
  /** Every standard column in its current display order, including hidden columns. */
  columnOrder: TraceColumnId[];
  visibleColumns: TraceColumnId[];
  toggleColumn: (column: TraceColumnId) => void;
  /** Move `activeColumn` to `targetColumn`'s position, persisting the new order. */
  reorderColumn: (activeColumn: TraceColumnId, targetColumn: TraceColumnId) => void;
  resetToDefaults: () => void;
  /**
   * Replace the whole visible set with an explicit list, persisted as overrides (every column gets
   * an explicit true/false so the result is independent of the live default). Used when adopting a
   * saved/shared view's columns into the user's own persisted state. An optional `savedColumnOrder`
   * restores the view's column order too; when omitted the order is derived from `columns` (their
   * order first, then any remaining columns in their current order).
   */
  setColumns: (columns: TraceColumnId[], savedColumnOrder?: TraceColumnId[]) => void;
  dynamicVisibilityById: Record<string, boolean>;
  setDynamicVisibility: Dispatch<SetStateAction<Record<string, boolean>>>;
}

/**
 * Persists the traces table column visibility and order in localStorage under `storageKey`.
 *
 * Stores per-column *overrides* on top of a live-computed default (via `getDefaultVisible`) rather
 * than a flat visible list. This lets a data-driven default (e.g. Session shows only when the page
 * has sessions) coexist with a sticky, user-toggleable choice: an explicit toggle writes an override
 * that wins; reset clears overrides and restores canonical order. Dynamic (consumer-defined) column
 * ids are stored in the same overrides map and preserved across reset via `isDynamicColumnId`. Column
 * order lives under a separate `${storageKey}.order` key so reordering never invalidates the
 * visibility entry (and vice versa). Synchronous localStorage reads on mount avoid a column flicker.
 */
export const useTraceColumnVisibility = ({
  storageKey,
  version,
  getDefaultVisible,
  isDynamicColumnId,
}: UseTraceColumnVisibilityParams): TraceColumnVisibility => {
  const [overrides, setOverrides] = useLocalStorage<ColumnOverrides>({
    key: storageKey,
    version,
    initialValue: {},
  });
  const [storedColumnOrder, setStoredColumnOrder] = useLocalStorage<unknown>({
    key: `${storageKey}.order`,
    version,
    initialValue: [...TRACE_COLUMN_IDS],
  });

  const columnOrder = useMemo(() => normalizeColumnOrder(storedColumnOrder), [storedColumnOrder]);

  // Filter the order (not the canonical id list) so visible columns render in the user's chosen order.
  const visibleColumns = useMemo(
    () => columnOrder.filter((id) => overrides[id] ?? getDefaultVisible(id)),
    [columnOrder, overrides, getDefaultVisible],
  );

  const toggleColumn = useCallback(
    (column: TraceColumnId) => {
      const currentlyVisible = overrides[column] ?? getDefaultVisible(column);
      setOverrides((prev) => ({ ...prev, [column]: !currentlyVisible }));
    },
    [overrides, getDefaultVisible, setOverrides],
  );

  const reorderColumn = useCallback(
    (activeColumn: TraceColumnId, targetColumn: TraceColumnId) => {
      if (activeColumn === targetColumn) {
        return;
      }
      setStoredColumnOrder((previousOrder: unknown) => {
        const nextOrder = normalizeColumnOrder(previousOrder);
        const activeIndex = nextOrder.indexOf(activeColumn);
        const targetIndex = nextOrder.indexOf(targetColumn);
        if (activeIndex === -1 || targetIndex === -1) {
          return nextOrder;
        }
        nextOrder.splice(activeIndex, 1);
        nextOrder.splice(targetIndex, 0, activeColumn);
        return nextOrder;
      });
    },
    [setStoredColumnOrder],
  );

  // Reset clears standard overrides + column order; dynamic (consumer) overrides are preserved.
  const resetToDefaults = useCallback(() => {
    setOverrides((current) =>
      isDynamicColumnId ? Object.fromEntries(Object.entries(current).filter(([id]) => isDynamicColumnId(id))) : {},
    );
    setStoredColumnOrder([...TRACE_COLUMN_IDS]);
  }, [isDynamicColumnId, setOverrides, setStoredColumnOrder]);

  // Write an explicit override for every standard column so the adopted set is independent of the live
  // default (a column absent from `columns` is pinned hidden, not left to fall back to its default).
  // Dynamic overrides are preserved. An optional `savedColumnOrder` restores the view's order.
  const setColumns = useCallback(
    (columns: TraceColumnId[], savedColumnOrder?: TraceColumnId[]) => {
      const wanted = new Set(columns);
      setOverrides((current) => {
        const nextOverrides: ColumnOverrides = isDynamicColumnId
          ? Object.fromEntries(Object.entries(current).filter(([id]) => isDynamicColumnId(id)))
          : {};
        for (const id of TRACE_COLUMN_IDS) {
          nextOverrides[id] = wanted.has(id);
        }
        return nextOverrides;
      });
      setStoredColumnOrder(normalizeColumnOrder(savedColumnOrder ?? [...columns, ...columnOrder]));
    },
    [columnOrder, isDynamicColumnId, setOverrides, setStoredColumnOrder],
  );

  const dynamicVisibilityById = useMemo(
    () =>
      isDynamicColumnId ? Object.fromEntries(Object.entries(overrides).filter(([id]) => isDynamicColumnId(id))) : {},
    [isDynamicColumnId, overrides],
  );
  const setDynamicVisibility = useCallback<Dispatch<SetStateAction<Record<string, boolean>>>>(
    (next) =>
      setOverrides((current) => {
        const currentDynamic = isDynamicColumnId
          ? Object.fromEntries(Object.entries(current).filter(([id]) => isDynamicColumnId(id)))
          : {};
        const resolved = typeof next === 'function' ? next(currentDynamic) : next;
        return {
          ...Object.fromEntries(Object.entries(current).filter(([id]) => !isDynamicColumnId?.(id))),
          ...resolved,
        };
      }),
    [isDynamicColumnId, setOverrides],
  );

  return {
    columnOrder,
    visibleColumns,
    toggleColumn,
    reorderColumn,
    resetToDefaults,
    setColumns,
    dynamicVisibilityById,
    setDynamicVisibility,
  };
};
