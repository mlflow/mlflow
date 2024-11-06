import type { CSSProperties } from 'react';
import React from 'react';
import type { HTMLDataAttributes } from '../types';
export declare const TableContext: React.Context<{
    size: "default" | "small";
    someRowsSelected?: boolean;
    grid?: boolean;
}>;
export interface TableProps extends HTMLDataAttributes {
    size?: 'default' | 'small';
    /** Are any rows currently selected? You must specify this if using `TableRowSelectCell` in your table. */
    someRowsSelected?: boolean;
    /** Style property */
    style?: CSSProperties;
    /** Class name property */
    className?: string;
    /** Content slot for providing a pagination component */
    pagination?: React.ReactNode;
    /** Content slot for providing an Empty component */
    empty?: React.ReactNode;
    /** Child nodes for the table */
    children?: React.ReactNode | React.ReactNode[];
    /** Is this `Table` scrollable? Only use if `Table` is placed within a container of determinate height. */
    scrollable?: boolean;
    /** Adds grid styling to the table (e.g. border around cells and no hover styles) */
    grid?: boolean;
}
export declare const Table: React.ForwardRefExoticComponent<TableProps & React.RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=Table.d.ts.map