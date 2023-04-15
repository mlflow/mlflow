import type { CSSProperties } from 'react';
import React from 'react';
interface TableProps {
    size?: 'default' | 'small';
    /** Are any rows currently selected? You must specify this if using `TableRowSelectCell` in your table. */
    someRowsSelected?: boolean;
    /** Style property */
    style?: CSSProperties;
    /** Content slot for providing a pagination component */
    pagination?: React.ReactNode;
    /** Content slot for providing an Empty component */
    empty?: React.ReactNode;
}
export declare const Table: React.FC<TableProps>;
interface TableRowProps {
    isHeader?: boolean;
    /** Vertical alignment of the row's cells. */
    verticalAlignment?: 'top' | 'center';
}
export declare const TableRow: React.FC<TableRowProps>;
interface TableCellProps {
    /** Enables single-line ellipsis truncation */
    ellipsis?: boolean;
    /** Style property */
    style?: CSSProperties;
}
export declare const TableCell: React.FC<TableCellProps>;
interface TableHeaderProps {
    /** Enables single-line ellipsis truncation */
    ellipsis?: boolean;
    /** Is this column sortable? */
    sortable?: boolean;
    /** The current sort direction for this column */
    sortDirection?: 'asc' | 'desc' | 'none';
    /** Callback for when the user requests to toggle `sortDirection` */
    onToggleSort?: (event: unknown) => void;
    /** Style property */
    style?: CSSProperties;
}
export declare const TableHeader: React.FC<TableHeaderProps>;
interface TableHeaderResizeHandleProps {
    /** Style property */
    style?: CSSProperties;
    /** Pass a handler to be called on touchStart */
    resizeHandler?: (event: unknown) => void;
}
export declare const TableHeaderResizeHandle: React.FC<TableHeaderResizeHandleProps>;
type TableRowSelectCellProps = {
    /** Called when the checkbox is clicked */
    onChange?: (event: unknown) => void;
    /** Whether the checkbox is checked */
    checked?: boolean;
    /** Whether the row is indeterminate. Should only be used in header rows. */
    indeterminate?: boolean;
    /** Don't render a checkbox; used for providing spacing in header if you don't want "Select All" functionality */
    noCheckbox?: boolean;
};
export declare const TableRowSelectCell: React.FC<TableRowSelectCellProps>;
interface TableRowMenuContainerProps {
    /** Style property */
    style?: CSSProperties;
}
export declare const TableRowMenuContainer: React.FC<TableRowMenuContainerProps>;
interface TableSkeletonProps {
    /** Number of rows to render */
    lines?: number;
    /** Seed that deterministically arranges the uneven lines, so that they look like ragged text.
     * If you don't provide this (or give each skeleton the same seed) they will all look the same. */
    seed?: string;
    /** Style property */
    style?: CSSProperties;
}
export declare const TableSkeleton: React.FC<TableSkeletonProps>;
export {};
//# sourceMappingURL=index.d.ts.map