import type { CSSProperties } from 'react';
import React from 'react';
import type { HTMLDataAttributes } from '../types';
export declare const TableRowContext: React.Context<{
    isHeader: boolean;
}>;
export interface TableRowProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement> {
    /** Style property */
    style?: CSSProperties;
    /** Class name property */
    className?: string;
    /** Set to true if this row is to be used for a table header row */
    isHeader?: boolean;
    /** @deprecated Vertical alignment of the row's cells. No longer supported (See FEINF-1937) */
    verticalAlignment?: 'top' | 'center';
    /** Child nodes for the table row */
    children?: React.ReactNode | React.ReactNode[];
}
export declare const TableRow: React.ForwardRefExoticComponent<TableRowProps & React.RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=TableRow.d.ts.map