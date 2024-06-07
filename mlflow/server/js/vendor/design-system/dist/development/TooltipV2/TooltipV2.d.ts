import type { TooltipContentProps as RadixTooltipContentProps, TooltipProps as RadixTooltipRootProps } from '@radix-ui/react-tooltip';
import type { HTMLAttributes } from 'react';
import React from 'react';
import type { HTMLDataAttributes } from '../../design-system/types';
export interface TooltipV2Props extends HTMLDataAttributes, Pick<RadixTooltipContentProps, 'side' | 'sideOffset' | 'align' | 'alignOffset' | 'avoidCollisions' | 'collisionPadding' | 'sticky' | 'hideWhenDetached'>, Pick<RadixTooltipRootProps, 'defaultOpen' | 'delayDuration'> {
    /**
     * The element with which the tooltip should be associated.
     */
    children: React.ReactNode;
    /**
     * Content or text that will appear within the tooltip.
     */
    content: React.ReactNode;
}
/**
 * If the tooltip is not displaying for you, it might be because the child does not accept the onMouseEnter, onMouseLeave, onPointerEnter,
 * onPointerLeave, onFocus, and onClick props. You can add these props to your child component, or wrap it in a `<span>` tag.
 *
 * See go/dubois.
 */
export declare const TooltipV2: React.FC<TooltipV2Props> & HTMLAttributes<HTMLDivElement>;
//# sourceMappingURL=TooltipV2.d.ts.map