import type { ColumnState } from '@ag-grid-community/core';

/**
 * Adapter between the library-neutral column layout persisted in
 * `ExperimentPageUIState` (`columnOrder` + `columnWidths`) and ag-grid's
 * `ColumnState`. Keeps ag-grid types out of the persisted model: if the runs
 * table ever migrates off ag-grid, only this file changes.
 */

/** Model -> ag-grid. Feed into `columnApi.applyColumnState({ state, applyOrder: true })`. */
export const prefsToColumnState = (columnOrder: string[], columnWidths: Record<string, number>): ColumnState[] =>
  columnOrder.map((colId) => ({
    colId,
    // `undefined` lets ag-grid keep the column's default width.
    width: columnWidths[colId],
  }));

/**
 * ag-grid -> model. Read `columnApi.getColumnState()` on move/resize.
 * `allColumns` bounds the result to known colIds; widths are sanitized to
 * finite, positive numbers so corrupt values never get persisted.
 */
export const columnStateToPrefs = (
  state: ColumnState[],
  allColumns: readonly string[],
): { columnOrder: string[]; columnWidths: Record<string, number> } => {
  const known = new Set(allColumns);
  const relevant = state.filter(
    (column): column is ColumnState & { colId: string } => typeof column.colId === 'string' && known.has(column.colId),
  );

  const columnWidths: Record<string, number> = {};
  for (const column of relevant) {
    if (typeof column.width === 'number' && Number.isFinite(column.width) && column.width > 0) {
      columnWidths[column.colId] = column.width;
    }
  }

  return {
    columnOrder: relevant.map((column) => column.colId),
    columnWidths,
  };
};
