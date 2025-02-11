import type { CSSProperties } from 'react';
import React from 'react';
import type { HTMLDataAttributes } from '../types';
export interface TableRowActionProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement> {
    /** Style property */
    style?: CSSProperties;
    /** Child nodes for the table row. Should contain a single small-sized button. */
    children?: React.ReactNode;
    /** Class name property */
    className?: string;
}
export declare const TableRowAction: React.ForwardRefExoticComponent<TableRowActionProps & React.RefAttributes<HTMLDivElement>>;
/** @deprecated Use `TableRowAction` instead */
export declare const TableRowMenuContainer: React.ForwardRefExoticComponent<TableRowActionProps & React.RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=TableRowAction.d.ts.map