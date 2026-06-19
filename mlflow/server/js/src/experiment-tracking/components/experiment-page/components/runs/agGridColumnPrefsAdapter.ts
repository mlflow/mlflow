import type { ColumnState } from '@ag-grid-community/core';

import type { ColumnPreferences } from '@databricks/web-shared/table-prefs';

/**
 * Adapter between the library-agnostic {@link ColumnPreferences} model and
 * ag-grid's `ColumnState`. Keeps ag-grid types out of the shared prefs core:
 * if the runs table ever migrates off ag-grid, only this file changes.
 */

/** Model -> ag-grid. Feed into `columnApi.applyColumnState({ state, applyOrder: true })`. */
export const prefsToColumnState = (preferences: ColumnPreferences): ColumnState[] => {
  const visible = new Set(preferences.visibleColumns);
  return preferences.columnOrder.map((colId) => ({
    colId,
    hide: !visible.has(colId),
    // `undefined` lets ag-grid keep the column's default width.
    width: preferences.columnWidths[colId],
  }));
};

/**
 * ag-grid -> model. Read `columnApi.getColumnState()` on move/resize/visibility.
 * `allColumns` bounds the result to known colIds; the prefs core re-sanitizes,
 * so unknown ids and missing widths are handled defensively downstream.
 */
export const columnStateToPrefs = (state: ColumnState[], allColumns: readonly string[]): ColumnPreferences => {
  const known = new Set(allColumns);
  const relevant = state.filter((column) => typeof column.colId === 'string' && known.has(column.colId));

  const columnWidths: Record<string, number> = {};
  for (const column of relevant) {
    if (typeof column.width === 'number') {
      columnWidths[column.colId] = column.width;
    }
  }

  return {
    columnOrder: relevant.map((column) => column.colId),
    visibleColumns: relevant.filter((column) => !column.hide).map((column) => column.colId),
    columnWidths,
  };
};
