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
 * The leading attribute columns (Run Name, Created) are pinned-left and can't be dragged
 * themselves, but ag-grid still lets a user drop ANOTHER data column in front of them. That layout
 * can't be persisted — the adapter round-trips order + width but not `pinned`, so the pinned
 * anchors re-sort left on reload and the move silently reverts. Rather than persist a layout we
 * can't restore, we detect the violation after a move and snap the order back.
 *
 * Inputs:
 * - `currentOrder`: colIds ag-grid reports after a move, in display order. This includes the
 *   structural checkbox column, whose colId ag-grid auto-generates (e.g. "1") — it is NOT a data
 *   column and must be left wherever ag-grid put it (it is lockPosition-anchored to index 0).
 * - `anchorColIds`: the data columns that must stay ahead of all other DATA columns.
 * - `dataColIds`: every real (user-facing) data colId — i.e. `allColumns`, which already excludes
 *   the checkbox. Only these are reordered; anything else in `currentOrder` is structural and left
 *   in place.
 *
 * Returns `null` if the data-column order is already valid, otherwise the corrected full colId
 * order to re-apply (structural columns kept in their current slots, anchors moved ahead of the
 * other data columns, remaining data columns keeping their relative order).
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

  // Position of each anchor among the DATA columns only (ignoring structural columns like the
  // checkbox). If no anchors are present there is nothing to protect (compact/compare layouts).
  const dataOrder = currentOrder.filter(isData);
  const anchorsPresent = anchorColIds.filter((id) => dataSet.has(id) && dataOrder.includes(id));
  if (anchorsPresent.length === 0) {
    return null;
  }
  const lastAnchorIdxInData = Math.max(...anchorsPresent.map((id) => dataOrder.indexOf(id)));

  // Violation: a non-anchor data column sits before the last anchor among the data columns.
  const violated = dataOrder.slice(0, lastAnchorIdxInData).some((colId) => !isAnchor(colId));
  if (!violated) {
    return null;
  }

  // Desired data-column order: anchors first (configured order), then the rest in current order.
  const rest = dataOrder.filter((colId) => !isAnchor(colId));
  const desiredData = [...anchorsPresent, ...rest];

  // Re-thread the desired data order back through `currentOrder`, leaving every structural
  // (non-data) column exactly where it is. Walk the original slots; data slots get filled from
  // desiredData in sequence, structural slots (e.g. the checkbox) pass through untouched. Slots
  // with no colId are dropped — applyColumnState leaves any unlisted column in place.
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
