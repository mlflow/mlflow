import type { ColumnSizingState, Table } from '@tanstack/react-table';
import { useEffect, useRef } from 'react';

/**
 * Persists column widths on the falling edge of a resize (mouseup): one write per drag, none on
 * mount. Sizing is kept uncontrolled (seeded once via `initialState`) so live drag stays smooth; this
 * hook watches `columnSizingInfo.isResizingColumn` and fires `onColumnSizingSettled` once when a drag
 * ends. The falling-edge ref avoids the redundant mount write a plain `if (!isResizing)` would do.
 *
 * TanStack stores the raw pointer delta in `columnSizing` even when `getSize()` clamps the rendered
 * width to the column's min/max, so the settled state is reset to each column's clamped `getSize()`
 * before persisting; otherwise an oversized first drag becomes the next render's persisted max and a
 * second drag can grow beyond the original ceiling.
 */
export const useColumnSizingPersistence = <T>(
  table: Table<T>,
  onColumnSizingSettled: (sizing: ColumnSizingState) => void,
): void => {
  const isResizingColumn = Boolean(table.getState().columnSizingInfo.isResizingColumn);
  const wasResizing = useRef(false);
  useEffect(() => {
    if (wasResizing.current && !isResizingColumn) {
      const columnSizing = table.getState().columnSizing;
      const clampedSizing = { ...columnSizing };
      for (const column of table.getAllLeafColumns()) {
        if (columnSizing[column.id] !== undefined) {
          clampedSizing[column.id] = column.getSize();
        }
      }
      table.setColumnSizing(clampedSizing);
      onColumnSizingSettled(clampedSizing);
    }
    wasResizing.current = isResizingColumn;
  }, [table, isResizingColumn, onColumnSizingSettled]);
};
