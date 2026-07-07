import type { HTMLAttributes } from 'react';
import React from 'react';
import type { TooltipProps } from './Tooltip';
export interface TruncationTooltipProps extends Omit<TooltipProps, 'open' | 'onOpenChange' | 'content'> {
}
/**
 * A variant of Tooltip that only shows when the child's text content is truncated.
 * The tooltip content is automatically derived from the child element's textContent,
 * so consumers don't need to pass the text.
 *
 * Truncation is checked lazily only on mouse enter, not during render.
 * This ensures negligible performance impact even with many TruncationTooltips on the page.
 *
 * Detects both:
 * - Horizontal ellipsis (text-overflow: ellipsis or overflow: hidden)
 * - Vertical line-clamp (line-clamp or -webkit-line-clamp)
 *
 * See go/dubois.
 */
export declare const TruncationTooltip: React.FC<React.PropsWithChildren<TruncationTooltipProps>> & HTMLAttributes<HTMLDivElement>;
//# sourceMappingURL=TruncationTooltip.d.ts.map