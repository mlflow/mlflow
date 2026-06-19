import type { ColumnPreferences } from './types';

/**
 * Filters non-positive-finite values so corrupt/old entries (NaN, strings,
 * null, negatives, 0) can't flow into a width and collapse a column.
 */
export const sanitizeColumnWidths = (raw: unknown): Record<string, number> => {
  if (raw === null || typeof raw !== 'object') {
    return {};
  }
  const entries = Object.entries(raw as Record<string, unknown>).filter(
    ([, value]) => typeof value === 'number' && Number.isFinite(value) && value > 0,
  );
  return Object.fromEntries(entries);
};

/**
 * Keeps only known colIds, returned in canonical (`allColumns`) order. Unknown
 * ids (e.g. a metric column that no longer exists) are dropped, and the result
 * never depends on the stored order so toggling can't shuffle the table.
 */
export const sanitizeVisibleColumns = (raw: unknown, allColumns: readonly string[]): string[] => {
  const requested = new Set(Array.isArray(raw) ? raw.filter((id): id is string => typeof id === 'string') : []);
  return allColumns.filter((id) => requested.has(id));
};

/**
 * Keeps the stored display order for known colIds, then appends any known
 * columns missing from storage in their canonical position. Drops unknown ids.
 */
export const sanitizeColumnOrder = (raw: unknown, allColumns: readonly string[]): string[] => {
  const known = new Set(allColumns);
  const storedOrder = Array.isArray(raw)
    ? raw.filter((id): id is string => typeof id === 'string' && known.has(id))
    : [];
  const seen = new Set(storedOrder);
  return [...storedOrder, ...allColumns.filter((id) => !seen.has(id))];
};

export const sanitizePreferences = (
  raw: Partial<ColumnPreferences> | null | undefined,
  allColumns: readonly string[],
  defaultVisible: readonly string[],
): ColumnPreferences => ({
  // `undefined` means "never customized" -> fall back to defaults. An empty
  // array means the user hid everything and must be respected.
  visibleColumns:
    raw?.visibleColumns !== undefined
      ? sanitizeVisibleColumns(raw.visibleColumns, allColumns)
      : sanitizeVisibleColumns(defaultVisible, allColumns),
  columnOrder: sanitizeColumnOrder(raw?.columnOrder, allColumns),
  columnWidths: sanitizeColumnWidths(raw?.columnWidths),
});
