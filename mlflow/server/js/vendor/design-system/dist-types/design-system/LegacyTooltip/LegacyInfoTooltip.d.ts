import type { LegacyTooltipProps } from './LegacyTooltip';
/**
 * `LegacyInfoTooltip` is deprecated in favor of the new `InfoTooltip` component
 * @deprecated
 */
export interface LegacyInfoTooltipProps extends Omit<React.HTMLAttributes<HTMLSpanElement>, 'title'> {
    title: React.ReactNode;
    tooltipProps?: Omit<LegacyTooltipProps, 'children' | 'title'>;
    iconTitle?: string;
    isKeyboardFocusable?: boolean;
}
/**
 * `LegacyInfoTooltip` is deprecated in favor of the new `InfoTooltip` component
 * @deprecated
 */
export declare const LegacyInfoTooltip: ({ title, tooltipProps, iconTitle, isKeyboardFocusable, ...iconProps }: LegacyInfoTooltipProps) => JSX.Element;
//# sourceMappingURL=LegacyInfoTooltip.d.ts.map