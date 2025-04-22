import React from 'react';
import type { TreeGridRow, TreeGridState, TreeGridColumn, TreeGridRenderCellArgs, TreeGridRenderRowArgs, TreeGridRenderTableArgs, TreeGridRenderHeaderArgs, TreeGridInitialState } from './types';
export interface TreeGridProps {
    /** The data to be displayed in the grid */
    data: TreeGridRow[];
    /** The columns to be displayed in the grid */
    columns: TreeGridColumn[];
    /** Function to render the cell content */
    renderCell: (args: TreeGridRenderCellArgs) => React.ReactElement;
    /** Optional function to render the row */
    renderRow?: (args: TreeGridRenderRowArgs) => React.ReactElement | null;
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
    /**
     * Configures state management for the TreeGrid.
     *
     * When no `state` is provided or when `initialState` is set, the state will be managed within
     * the TreeGrid. Otherwise, the given custom state implementation is used.
     */
    state?: {
        initialState: TreeGridInitialState;
    } | TreeGridState;
}
export declare const TreeGrid: React.FC<TreeGridProps>;
//# sourceMappingURL=TreeGrid.d.ts.map