import type { CSSProperties } from 'react';
import React from 'react';
import type { HTMLDataAttributes } from '../types';
export interface TableCellProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement> {
    /** @deprecated Use `multiline` prop instead. This prop will be removed soon. */
    ellipsis?: boolean;
    /** Enables multiline wrapping */
    multiline?: boolean;
    /** How to horizontally align the cell contents */
    align?: 'left' | 'center' | 'right';
    /** Class name property */
    className?: string;
    /** Style property */
    style?: CSSProperties;
    /** Child nodes for the table cell */
    children?: React.ReactNode | React.ReactNode[];
    /** If the content of this cell should be wrapped with Typography. Should only be set to false if
     * content is not a text (e.g. images) or you really need to render custom content. */
    wrapContent?: boolean;
}
export declare const TableCell: React.ForwardRefExoticComponent<TableCellProps & React.RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=TableCell.d.ts.map