import type { PopoverProps } from '../Popover/Popover';
export interface OverflowProps extends PopoverProps {
    /** Used for components like Tag which have already have margins */
    noMargin?: boolean;
    children: React.ReactNode;
}
export declare const Overflow: ({ children, noMargin, ...props }: OverflowProps) => React.ReactElement;
//# sourceMappingURL=Overflow.d.ts.map