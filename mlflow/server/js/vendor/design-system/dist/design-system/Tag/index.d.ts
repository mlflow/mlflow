import type { HTMLAttributes } from 'react';
import React from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { AnalyticsEventProps, HTMLDataAttributes } from '../types';
export interface TagProps extends HTMLDataAttributes, HTMLAttributes<HTMLSpanElement>, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnView | DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
    /**
     * The color of the tag.
     */
    color?: TagColors;
    /**
     * Text to be rendered inside the tag.
     */
    children: React.ReactNode;
    /**
     * Whether or not the tag should be closable.
     */
    closable?: boolean;
    /**
     * Function called when the close button is clicked.
     */
    onClose?: () => void;
    closeButtonProps?: Omit<React.HTMLAttributes<HTMLButtonElement>, 'children' | 'onClick' | 'onMouseDown'>;
    icon?: React.ReactNode;
}
export type TagColors = 'default' | 'brown' | 'coral' | 'charcoal' | 'indigo' | 'lemon' | 'lime' | 'pink' | 'purple' | 'teal' | 'turquoise';
export declare const Tag: React.ForwardRefExoticComponent<TagProps & React.RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=index.d.ts.map