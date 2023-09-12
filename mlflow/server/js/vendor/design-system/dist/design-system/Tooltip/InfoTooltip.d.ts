/// <reference types="react" />
import type { TooltipProps } from './Tooltip';
export interface InfoTooltipProps extends Omit<React.HTMLAttributes<HTMLSpanElement>, 'title'> {
    title: React.ReactNode;
    tooltipProps?: Omit<TooltipProps, 'children' | 'title'>;
    iconTitle?: string;
}
export declare const InfoTooltip: ({ title, tooltipProps, iconTitle, ...iconProps }: InfoTooltipProps) => JSX.Element;
//# sourceMappingURL=InfoTooltip.d.ts.map