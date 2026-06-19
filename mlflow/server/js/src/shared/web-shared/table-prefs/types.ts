/**
 * Library-agnostic representation of a user's column preferences for a data table.
 *
 * Intentionally free of any table-library types (ag-grid `ColumnState`,
 * TanStack `ColumnSizingState`, etc.) so a single persistence + sanitization
 * core can back tables built on different libraries via thin adapters.
 */
export interface ColumnPreferences {
  /** Visible colIds in canonical order. Source of truth for visibility. */
  visibleColumns: string[];
  /** Full display order including hidden columns. */
  columnOrder: string[];
  /** colId -> width in px. Sanitized to finite, positive values only. */
  columnWidths: Record<string, number>;
}
