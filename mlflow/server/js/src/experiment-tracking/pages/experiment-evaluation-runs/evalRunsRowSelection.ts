import type { Row, Table } from '@tanstack/react-table';
import type { MutableRefObject } from 'react';

/**
 * Toggles one row, or on shift-click every selectable visible row from the anchor (the last clicked
 * row, while still selected) through `row`, to the clicked row's new state.
 *
 * Reads selection from `table.getState()` at click time rather than from render props because
 * rows are memoized and may hold a stale render.
 */
export const toggleEvalRunRowSelection = <TData>({
  table,
  row,
  selectRange,
  anchorRowIdRef,
}: {
  table: Table<TData>;
  row: Row<TData>;
  selectRange: boolean;
  anchorRowIdRef: MutableRefObject<string | undefined>;
}): void => {
  const { rowSelection } = table.getState();
  const anchorRowId = anchorRowIdRef.current;
  anchorRowIdRef.current = row.id;

  const rows = table.getRowModel().rows;
  const anchorIndex =
    selectRange && anchorRowId && rowSelection[anchorRowId] ? rows.findIndex((r) => r.id === anchorRowId) : -1;
  const targetIndex = rows.findIndex((r) => r.id === row.id);
  if (anchorIndex < 0 || targetIndex < 0) {
    row.toggleSelected();
    return;
  }

  const shouldSelect = !row.getIsSelected();
  const next = { ...rowSelection };
  rows
    .slice(Math.min(anchorIndex, targetIndex), Math.max(anchorIndex, targetIndex) + 1)
    // Include collapsed children so a range matches what a plain click on each parent selects.
    .flatMap((rangeRow) => [rangeRow, ...rangeRow.getLeafRows()])
    .filter((rangeRow) => rangeRow.getCanSelect())
    .forEach((rangeRow) => {
      if (shouldSelect) {
        next[rangeRow.id] = true;
      } else {
        delete next[rangeRow.id];
      }
    });
  table.setRowSelection(next);
};
