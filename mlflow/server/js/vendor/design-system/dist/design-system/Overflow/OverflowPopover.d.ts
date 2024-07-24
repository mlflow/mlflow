import type { PopoverProps } from '../Popover/Popover';
interface OverflowPopoverProps extends PopoverProps {
    items: React.ReactNode[];
    renderLabel?: (label: string) => React.ReactNode;
    tooltipText?: string;
}
export declare const OverflowPopover: ({ items, renderLabel, tooltipText, ...props }: OverflowPopoverProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=OverflowPopover.d.ts.map