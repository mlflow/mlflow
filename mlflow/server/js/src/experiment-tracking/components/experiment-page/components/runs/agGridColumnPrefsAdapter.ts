import type { ColumnState } from '@ag-grid-community/core';

/**
 * Adapter between the library-neutral column layout persisted in
 * `ExperimentPageUIState` (`columnOrder` + `columnWidths`) and ag-grid's
 * `ColumnState`.
 *
 * GenAI MLflow uses Tanstack, when we do move to TanStack, we can swap out the adapter here
 */

/** Model -> ag-grid. Feed into `columnApi.applyColumnState({ state, applyOrder: true })`. */
export const prefsToColumnState = (columnOrder: string[], columnWidths: Record<string, number>): ColumnState[] =>
  columnOrder.map((colId) => ({
    colId,
    // `undefined` lets ag-grid keep the column's default width.
    width: columnWidths[colId],
  }));

/**
 * Keeps the pinned anchor columns (Run Name, Created) ahead of all other data columns after a
 * drag. ag-grid lets a user drop a column in front of them, but the adapter doesn't persist
 * `pinned`, so that layout reverts on reload. Returns the corrected colId order to re-apply, or
 * `null` if already valid. `dataColIds` (= `allColumns`) excludes the structural checkbox, whose
 * auto-generated colId is left in its current slot.
 */
export const getReorderCorrection = (
  currentOrder: (string | null | undefined)[],
  anchorColIds: string[],
  dataColIds: readonly string[],
): string[] | null => {
  const dataSet = new Set(dataColIds);
  const anchorSet = new Set(anchorColIds);
  const isData = (colId: string | null | undefined): colId is string => typeof colId === 'string' && dataSet.has(colId);
  const isAnchor = (colId: string | null | undefined): colId is string =>
    typeof colId === 'string' && anchorSet.has(colId);

  const dataOrder = currentOrder.filter(isData);
  const anchorsPresent = anchorColIds.filter((id) => dataSet.has(id) && dataOrder.includes(id));
  if (anchorsPresent.length === 0) {
    return null;
  }
  const lastAnchorIdxInData = Math.max(...anchorsPresent.map((id) => dataOrder.indexOf(id)));

  // A non-anchor data column sitting before the last anchor is the violation to correct.
  if (!dataOrder.slice(0, lastAnchorIdxInData).some((colId) => !isAnchor(colId))) {
    return null;
  }

  const desiredData = [...anchorsPresent, ...dataOrder.filter((colId) => !isAnchor(colId))];
  // Re-thread the desired data order through the original slots. Non-data columns (e.g. the
  // checkbox) are re-emitted in their existing position by colId — they are not dropped and not
  // relied on to stay in place implicitly. Only null/undefined colIds are omitted.
  let d = 0;
  return currentOrder.reduce<string[]>((acc, colId) => {
    if (isData(colId)) {
      acc.push(desiredData[d++]);
    } else if (typeof colId === 'string') {
      acc.push(colId);
    }
    return acc;
  }, []);
};

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
