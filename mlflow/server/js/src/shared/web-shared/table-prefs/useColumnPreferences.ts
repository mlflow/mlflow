import { useCallback, useMemo } from 'react';

import { sanitizeColumnWidths, sanitizePreferences } from './sanitize';
import type { ColumnPreferences } from './types';
import { useLocalStorage } from '../hooks/useLocalStorage';

export interface UseColumnPreferencesParams {
  /** Stable storage scope, e.g. `mlflow.runs.${experimentId}`. */
  storageKey: string;
  /** Bump to invalidate all stored entries (no migration path). */
  version: number;
  /** Canonical render order + universe of known colIds. */
  allColumns: readonly string[];
  /** Columns visible by default when the user hasn't customized. */
  defaultVisible: readonly string[];
}

export interface UseColumnPreferencesResult {
  /** Already sanitized; safe to feed straight into a table adapter. */
  preferences: ColumnPreferences;
  /** True once the user has saved any visibility/order/width customization. */
  isCustomized: boolean;
  setVisibleColumns: (next: string[]) => void;
  toggleColumn: (colId: string) => void;
  setColumnOrder: (next: string[]) => void;
  setColumnWidths: (
    updater: Record<string, number> | ((prev: Record<string, number>) => Record<string, number>),
  ) => void;
  /** Clears visibility, order, and widths back to defaults atomically. */
  resetToDefaults: () => void;
  /** Sanitized snapshot for URL/share, without touching localStorage. */
  serialize: () => ColumnPreferences;
  /** Merge externally-supplied prefs (e.g. from a shared URL) into storage. */
  hydrate: (prefs: Partial<ColumnPreferences>) => void;
}

type StoredPreferences = Partial<ColumnPreferences>;

/**
 * Library-agnostic persistence of table column preferences (visibility, order,
 * widths) in localStorage, scoped by `storageKey`. One storage entry holds all
 * three so "reset to defaults" clears them atomically and future prefs (sort,
 * etc.) can drop in as new fields. Per-library capture/apply lives in adapters.
 *
 * Built on `useLocalStorage` for a synchronous read on mount (no column flicker
 * on first paint).
 */
export const useColumnPreferences = ({
  storageKey,
  version,
  allColumns,
  defaultVisible,
}: UseColumnPreferencesParams): UseColumnPreferencesResult => {
  const [stored, setStored] = useLocalStorage<StoredPreferences>({
    key: storageKey,
    version,
    initialValue: {},
  });

  const preferences = useMemo(
    () => sanitizePreferences(stored, allColumns, defaultVisible),
    [stored, allColumns, defaultVisible],
  );

  const isCustomized = useMemo(
    () =>
      Boolean(
        stored &&
        (stored.visibleColumns !== undefined ||
          (stored.columnOrder?.length ?? 0) > 0 ||
          (stored.columnWidths && Object.keys(stored.columnWidths).length > 0)),
      ),
    [stored],
  );

  const setVisibleColumns = useCallback(
    (next: string[]) => {
      setStored((prev) => ({ ...prev, visibleColumns: next }));
    },
    [setStored],
  );

  const toggleColumn = useCallback(
    (colId: string) => {
      setStored((prev) => {
        const current = sanitizePreferences(prev, allColumns, defaultVisible).visibleColumns;
        const nextVisible = current.includes(colId)
          ? current.filter((id) => id !== colId)
          : // Preserve canonical order so toggling on doesn't shuffle the table.
            allColumns.filter((id) => current.includes(id) || id === colId);
        return { ...prev, visibleColumns: nextVisible };
      });
    },
    [setStored, allColumns, defaultVisible],
  );

  const setColumnOrder = useCallback(
    (next: string[]) => {
      setStored((prev) => ({ ...prev, columnOrder: next }));
    },
    [setStored],
  );

  // Capture handlers may pass a functional updater on every drag tick; sanitize
  // `prev` so corrupt entries already in storage die on the next write.
  const setColumnWidths = useCallback<UseColumnPreferencesResult['setColumnWidths']>(
    (updater) => {
      setStored((prev) => {
        const cleanPrev = sanitizeColumnWidths(prev.columnWidths);
        const next = typeof updater === 'function' ? updater(cleanPrev) : updater;
        return { ...prev, columnWidths: sanitizeColumnWidths(next) };
      });
    },
    [setStored],
  );

  const resetToDefaults = useCallback(() => {
    setStored({});
  }, [setStored]);

  const serialize = useCallback(() => preferences, [preferences]);

  const hydrate = useCallback(
    (prefs: Partial<ColumnPreferences>) => {
      setStored((prev) => ({ ...prev, ...prefs }));
    },
    [setStored],
  );

  return {
    preferences,
    isCustomized,
    setVisibleColumns,
    toggleColumn,
    setColumnOrder,
    setColumnWidths,
    resetToDefaults,
    serialize,
    hydrate,
  };
};
