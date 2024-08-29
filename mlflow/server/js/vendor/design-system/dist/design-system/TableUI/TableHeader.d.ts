import type { CSSProperties } from 'react';
import React from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { AnalyticsEventOptionalProps, HTMLDataAttributes } from '../types';
export interface TableHeaderProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement>, AnalyticsEventOptionalProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
    /** @deprecated Use `multiline` prop instead. This prop will be removed soon. */
    ellipsis?: boolean;
    /** Enables multiline wrapping */
    multiline?: boolean;
    /** Is this column sortable? */
    sortable?: boolean;
    /** The current sort direction for this column */
    sortDirection?: 'asc' | 'desc' | 'none';
    /** Callback for when the user requests to toggle `sortDirection` */
    onToggleSort?: (event: unknown) => void;
    /** Style property */
    style?: CSSProperties;
    /** Class name property */
    className?: string;
    /** Child nodes for the table header */
    children?: React.ReactNode | React.ReactNode[];
    /** Whether the table header should include a resize handler */
    resizable?: boolean;
    /** Event handler to be passed down to <TableHeaderResizeHandle /> */
    resizeHandler?: React.PointerEventHandler<HTMLDivElement>;
    /** Whether the header is currently being resized */
    isResizing?: boolean;
    /** How to horizontally align the cell contents */
    align?: 'left' | 'center' | 'right';
    /** If the content of this header should be wrapped with Typography. Should only be set to false if
     * content is not a text (e.g. images) or you really need to render custom content. */
    wrapContent?: boolean;
}
export declare const TableHeader: React.ForwardRefExoticComponent<TableHeaderProps & React.RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=TableHeader.d.ts.map