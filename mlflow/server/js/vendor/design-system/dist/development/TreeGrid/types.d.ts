import type React from 'react';
export type TreeGridRowId = string | number;
export interface TreeGridRow {
    /** Unique identifier for the row */
    id: TreeGridRowId;
    /** Allow any other custom properties */
    [key: string]: any;
    /** Child rows */
    children?: TreeGridRow[];
}
export interface TreeGridState {
    /** The expanded state of each row */
    expandedRows: Record<TreeGridRowId, boolean>;
    /** The currently active row index */
    activeRowId: TreeGridRowId | null;
    /** Toggles expansion status of the given row */
    toggleRowExpanded: (rowId: TreeGridRowId) => void;
    /** Sets the active row id */
    setActiveRowId: (rowId: TreeGridRowId | null) => void;
}
export type TreeGridInitialState = Pick<TreeGridState, 'expandedRows'>;
export interface TreeGridColumn {
    /** Unique identifier for the column */
    id: string;
    /** Header text for the column. Not displayed unless `includeHeader` is true. */
    header: string;
    /** Whether this column's cells serve as each row's header. You should only have one row header column per grid */
    isRowHeader?: boolean;
    /** Whether the content of this column's cells is focusable; if so, focus will move to the contents rather than the cell */
    contentFocusable?: boolean;
}
export interface TreeGridRenderCellArgs {
    /** The row data */
    row: TreeGridRow;
    /** The column data */
    column: TreeGridColumn;
    /** The depth of the row. Use this to determine the indentation level of the row, if desired. */
    rowDepth: number;
    /** The index of the row */
    rowIndex: number;
    /** The index of the column */
    colIndex: number;
    /** Whether the row is currently keyboard active */
    rowIsKeyboardActive: boolean;
    /** Whether the row is expanded */
    rowIsExpanded: boolean;
    /** Function to toggle the expanded state of a row */
    toggleRowExpanded: (rowId: string | number) => void;
    /** Props to be applied to the cell element. These must be spread onto a `td` element. */
    cellProps: React.TdHTMLAttributes<HTMLTableCellElement>;
}
export interface TreeGridRenderRowArgs {
    /** The row data */
    row: TreeGridRow;
    /** The index of the row */
    rowIndex: number;
    /** Whether the row is expanded */
    isExpanded: boolean;
    /** Whether the row is currently keyboard active */
    isKeyboardActive: boolean;
    /** Props to be applied to the row element. These must be spread onto a `tr` element. */
    rowProps: React.HTMLAttributes<HTMLTableRowElement>;
    /** The children of the row */
    children: React.ReactNode;
}
export interface TreeGridRenderTableArgs {
    /** Props to be applied to the table element. These must be spread onto a `table` element. */
    tableProps: React.RefAttributes<HTMLTableElement> & React.TableHTMLAttributes<HTMLTableElement>;
    /** The children of the table */
    children: React.ReactNode;
}
export interface TreeGridRenderHeaderArgs {
    /** The columns to be rendered in the header */
    columns: TreeGridColumn[];
    /** Props to be applied to the header element */
    headerProps: React.HTMLAttributes<HTMLTableSectionElement>;
}
//# sourceMappingURL=types.d.ts.map