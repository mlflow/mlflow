import { type ReactElement } from 'react';
import type { HTMLDataAttributes } from '../types';
export interface GraphicProps extends Omit<React.HTMLAttributes<HTMLSpanElement>, 'style' | 'className'>, HTMLDataAttributes {
    /**
     * The SVG component to render. Generated graphics will provide both light and dark
     * variants and automatically select the appropriate one based on theme.
     */
    component?: (props: React.SVGProps<SVGSVGElement>) => ReactElement | null;
    /**
     * Width of the graphic. Can be a number (pixels) or a string with units.
     * @default 128
     */
    width?: number | string;
    /**
     * Height of the graphic. Can be a number (pixels) or a string with units.
     * If not provided, the graphic will scale proportionally based on width and viewBox.
     */
    height?: number | string;
}
/**
 * @description FOR INTERNAL USE ONLY. DO NOT USE THIS COMPONENT DIRECTLY.
 */
export declare const Graphic: import("react").ForwardRefExoticComponent<GraphicProps & import("react").RefAttributes<HTMLSpanElement>>;
//# sourceMappingURL=Graphic.d.ts.map