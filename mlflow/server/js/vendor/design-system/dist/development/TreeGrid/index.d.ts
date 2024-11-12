import React from 'react';
type TreeGridRowId = string | number;
export interface TreeGridRow {
    /** Unique identifier for the row */
    id: TreeGridRowId;
    /** Allow any other custom properties */
    [key: string]: any;
    /** Child rows */
    children?: TreeGridRow[];
}
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
export interface TreeGridState {
    /** The expanded state of each row */
    expandedRows: Record<TreeGridRowId, boolean>;
}
export interface TreeGridProps {
    /** The data to be displayed in the grid */
    data: TreeGridRow[];
    /** The columns to be displayed in the grid */
    columns: TreeGridColumn[];
    /** Function to render the cell content */
    renderCell: (args: TreeGridRenderCellArgs) => React.ReactElement;
    /** Optional function to render the row */
    renderRow?: (args: TreeGridRenderRowArgs) => React.ReactElement;
    /** Optional function to render the entire table */
    renderTable?: (args: TreeGridRenderTableArgs) => React.ReactElement;
    /** Optional function to render the header */
    renderHeader?: (args: TreeGridRenderHeaderArgs) => React.ReactElement;
    /** Callback function when a row is selected via the keyboard */
    onRowKeyboardSelect?: (rowId: string | number) => void;
    /** Callback function when a cell is selected via the keyboard */
    onCellKeyboardSelect?: (rowId: string | number, columnKey: string) => void;
    /** Whether to include a header in the grid */
    includeHeader?: boolean;
    /** Initial state for the grid */
    initialState?: TreeGridState;
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
export declare const TreeGrid: React.FC<TreeGridProps>;
export {};
//# sourceMappingURL=index.d.ts.map