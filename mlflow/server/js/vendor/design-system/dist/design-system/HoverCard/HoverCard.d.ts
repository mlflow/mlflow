import type { CSSObject } from '@emotion/react';
import type { HoverCardContentProps as RadixHoverCardContentProps, HoverCardProps as RadixHoverCardRootProps } from '@radix-ui/react-hover-card';
import React from 'react';
/**
 * Props for the HoverCard component.
 * This interface extends specific props from Radix HoverCard components to maintain flexibility and alignment
 * with Radix's capabilities while providing a simpler API for end-users
 */
export interface HoverCardProps extends Pick<RadixHoverCardContentProps, 'side' | 'sideOffset' | 'align' | 'alignOffset' | 'avoidCollisions' | 'collisionPadding' | 'sticky'>, Pick<RadixHoverCardRootProps, 'defaultOpen' | 'open' | 'onOpenChange' | 'openDelay' | 'closeDelay'> {
    trigger: React.ReactNode;
    content: React.ReactNode;
    minWidth?: CSSObject['minWidth'];
    maxWidth?: CSSObject['maxWidth'];
}
/**
 * The HoverCard component combines Radix's HoverCard primitives into a single, easy-to-use component.
 * It handles the setup of the trigger, content, and arrow elements, as well as applying custom styles
 * using Emotion CSS
 */
export declare const HoverCard: React.FC<HoverCardProps>;
//# sourceMappingURL=HoverCard.d.ts.map