import type { SerializedStyles } from '@emotion/react';
import type { CSSProperties } from 'react';
import React from 'react';
import type { HTMLDataAttributes } from '../types';
export interface TableRowMultiActionProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement> {
    /** Style property */
    style?: CSSProperties;
    css?: SerializedStyles;
    /** Child nodes for the table row. Should contain a single small-sized button. */
    children?: React.ReactNode;
    /** Class name property */
    className?: string;
}
/**
 * A component that renders a multi-action row in a table. Similar to TableRowAction, but with a gap between the actions.
 * TableRowMultiAction also allows for individual buttons to apply the `skipHideIconButtonClassName` classname and be
 * always-visible when necessary.
 *
 * Child buttons can also apply the `skipHideIconButtonClassName` classname and be always-visible when necessary.
 *
 * @param children - The child nodes for the table row. Should contain one or more small-sized buttons.
 * @param style - The style property.
 * @param className - The class name property.
 * @param css - The CSS property.
 * @param rest - The rest of the props.
 */
export declare const TableRowMultiAction: React.ForwardRefExoticComponent<TableRowMultiActionProps & React.RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=TableRowMultiAction.d.ts.map