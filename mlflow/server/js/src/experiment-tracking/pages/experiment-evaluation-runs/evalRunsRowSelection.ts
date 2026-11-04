import type { Row, Table } from '@tanstack/react-table';
import type { MutableRefObject } from 'react';
import type { RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';

export const containsRun = (row: RunEntityOrGroupData, runUuid: string): boolean =>
  ('info' in row && row.info.runUuid === runUuid) ||
  ('subRuns' in row && row.subRuns.some((subRun) => containsRun(subRun, runUuid))) ||
  Boolean('children' in row && row.children?.some((child) => containsRun(child, runUuid)));

// Moves the baseline row (or its group) and its subtree to the top, matching the pinned table order.
export const pinBaselineRows = (rows: Row<RunEntityOrGroupData>[], baselineRunUuid: string) => {
  const baselineIndex = rows.findIndex(
    (row) =>
      row.id === baselineRunUuid || ('groupValues' in row.original && containsRun(row.original, baselineRunUuid)),
  );
  if (baselineIndex <= 0) {
    return rows;
  }
  const baselineDepth = rows[baselineIndex].depth;
  const subtreeEnd = rows.findIndex((row, index) => index > baselineIndex && row.depth <= baselineDepth);
  const end = subtreeEnd === -1 ? rows.length : subtreeEnd;
  return [...rows.slice(baselineIndex, end), ...rows.slice(0, baselineIndex), ...rows.slice(end)];
};

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
  orderRows,
}: {
  table: Table<TData>;
  row: Row<TData>;
  selectRange: boolean;
  anchorRowIdRef: MutableRefObject<string | undefined>;
  /** Reorders the row model to match the rendered order, so a range covers the rows shown between the clicks. */
  orderRows?: (rows: Row<TData>[]) => Row<TData>[];
}): void => {
  const { rowSelection } = table.getState();
  const anchorRowId = anchorRowIdRef.current;
  anchorRowIdRef.current = row.id;

  const modelRows = table.getRowModel().rows;
  const rows = orderRows ? orderRows(modelRows) : modelRows;
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
