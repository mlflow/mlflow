import { type TooltipProps } from './Tooltip';
export interface InfoTooltipProps extends Omit<TooltipProps, 'children'> {
    iconTitle?: string;
}
export declare const InfoTooltip: ({ content, iconTitle, ...props }: InfoTooltipProps) => JSX.Element;
//# sourceMappingURL=InfoTooltip.d.ts.map