import type { HTMLAttributes } from 'react';
import React from 'react';
import type { SecondaryColorToken, TagColorToken } from '../../theme/colors';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { AnalyticsEventProps, HTMLDataAttributes } from '../types';
export interface TagProps extends HTMLDataAttributes, HTMLAttributes<HTMLSpanElement>, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnView> {
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
}
export type TagColors = keyof typeof colorMap;
declare const colorMap: Record<SecondaryColorToken | 'default' | 'charcoal', TagColorToken | 'grey600'>;
export declare const Tag: React.ForwardRefExoticComponent<TagProps & React.RefAttributes<HTMLDivElement>>;
export {};
//# sourceMappingURL=index.d.ts.map