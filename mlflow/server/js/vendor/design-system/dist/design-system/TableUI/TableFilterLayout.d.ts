import type { CSSProperties } from 'react';
import React from 'react';
import type { HTMLDataAttributes } from '../types';
export interface TableFilterLayoutProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement> {
    /** Style property */
    style?: CSSProperties;
    /** Should contain the filter controls to be rendered in this layout. */
    children?: React.ReactNode;
    /** Class name property */
    className?: string;
    /** A container to hold action `Button` elements. */
    actions?: React.ReactNode;
}
export declare const TableFilterLayout: React.ForwardRefExoticComponent<TableFilterLayoutProps & React.RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=TableFilterLayout.d.ts.map