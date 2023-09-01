import type { HTMLAttributes } from 'react';
import React from 'react';
import type { SecondaryColorToken, TagColorToken } from '../../theme/colorList';
import type { HTMLDataAttributes } from '../types';
export interface TagProps extends HTMLDataAttributes, HTMLAttributes<HTMLSpanElement> {
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
}
export type TagColors = keyof typeof colorMap;
declare const colorMap: Record<SecondaryColorToken | 'default', TagColorToken>;
export declare function Tag(props: TagProps): JSX.Element;
export {};
//# sourceMappingURL=index.d.ts.map