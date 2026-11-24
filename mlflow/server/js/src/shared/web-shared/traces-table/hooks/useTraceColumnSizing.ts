import type { ColumnSizingState } from '@tanstack/react-table';
import { useLocalStorage } from '../../hooks/useLocalStorage';

export interface UseTraceColumnSizingParams {
  /** localStorage key (the consumer scopes it, e.g. per experiment). */
  storageKey: string;
  /** Bump when the column set or sizing scheme changes so stale pixel widths reset. */
  version: number;
}

export interface TraceColumnSizing {
  columnSizing: ColumnSizingState;
  setColumnSizing: (value: ColumnSizingState | ((prev: ColumnSizingState) => ColumnSizingState)) => void;
}

/**
 * Persists per-column pixel widths for the resizable traces table in localStorage under `storageKey`.
 * Shape matches TanStack's `ColumnSizingState`. The table keeps sizing *uncontrolled* (seeded once
 * from `columnSizing` via `initialState`, then owned internally for smooth live drag), so
 * `setColumnSizing` here is the settle-persist callback the table calls once on mouseup — not a
 * per-tick controlled setter. `setColumnSizing` is reference-stable (memoized on the storage key),
 * so it's safe to use directly as a resize-settle effect dependency. Empty map means "use each
 * column's seeded default size".
 */
export const useTraceColumnSizing = ({ storageKey, version }: UseTraceColumnSizingParams): TraceColumnSizing => {
  const [columnSizing, setColumnSizing] = useLocalStorage<ColumnSizingState>({
    key: storageKey,
    version,
    initialValue: {},
  });

  return { columnSizing, setColumnSizing };
};
