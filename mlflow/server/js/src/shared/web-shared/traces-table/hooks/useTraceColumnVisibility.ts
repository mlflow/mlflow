import { useCallback, useMemo } from 'react';
import { useLocalStorage } from '../../hooks/useLocalStorage';
import { TRACE_COLUMN_IDS } from '../constants';
import type { TraceColumnId } from '../types';

/** Per-column visibility overrides. A column absent from the map falls back to its computed default. */
type ColumnOverrides = Partial<Record<TraceColumnId, boolean>>;

export interface UseTraceColumnVisibilityParams {
  /** localStorage key (the consumer scopes it, e.g. per experiment). */
  storageKey: string;
  /** Bump when the stored schema/default set changes so stale entries reset. */
  version: number;
  /** Computes a column's default visibility (may depend on live data the consumer knows about). */
  getDefaultVisible: (id: TraceColumnId) => boolean;
}

export interface TraceColumnVisibility {
  visibleColumns: TraceColumnId[];
  toggleColumn: (column: TraceColumnId) => void;
  resetToDefaults: () => void;
  /**
   * Replace the whole visible set with an explicit list, persisted as overrides (every column gets
   * an explicit true/false so the result is independent of the live default). Used when adopting a
   * saved/shared view's columns into the user's own persisted state.
   */
  setColumns: (columns: TraceColumnId[]) => void;
}

/**
 * Persists the traces table column-visibility selection in localStorage under `storageKey`.
 *
 * Stores per-column *overrides* on top of a live-computed default (via `getDefaultVisible`) rather
 * than a flat visible list. This lets a data-driven default (e.g. Session shows only when the page
 * has sessions) coexist with a sticky, user-toggleable choice: an explicit toggle writes an override
 * that wins; reset clears overrides, returning every column to its default. Synchronous localStorage
 * read on mount avoids a column flicker on first paint.
 */
export const useTraceColumnVisibility = ({
  storageKey,
  version,
  getDefaultVisible,
}: UseTraceColumnVisibilityParams): TraceColumnVisibility => {
  const [overrides, setOverrides] = useLocalStorage<ColumnOverrides>({
    key: storageKey,
    version,
    initialValue: {},
  });

  // Iterate the canonical id list so render order is stable regardless of stored key order.
  const visibleColumns = useMemo(
    () => TRACE_COLUMN_IDS.filter((id) => overrides[id] ?? getDefaultVisible(id)),
    [overrides, getDefaultVisible],
  );

  const toggleColumn = useCallback(
    (column: TraceColumnId) => {
      const currentlyVisible = overrides[column] ?? getDefaultVisible(column);
      setOverrides((prev) => ({ ...prev, [column]: !currentlyVisible }));
    },
    [overrides, getDefaultVisible, setOverrides],
  );

  const resetToDefaults = useCallback(() => setOverrides({}), [setOverrides]);

  // Write an explicit override for every column so the adopted set is independent of the live
  // default (a column absent from `columns` is pinned hidden, not left to fall back to its default).
  const setColumns = useCallback(
    (columns: TraceColumnId[]) => {
      const visible = new Set<string>(columns);
      setOverrides(Object.fromEntries(TRACE_COLUMN_IDS.map((id) => [id, visible.has(id)])) as ColumnOverrides);
    },
    [setOverrides],
  );

  return { visibleColumns, toggleColumn, resetToDefaults, setColumns };
};
