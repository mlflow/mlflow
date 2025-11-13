import type { PopoverProps } from '../Popover/Popover';
export interface OverflowProps extends PopoverProps {
    /** Used for components like Tag which have already have margins */
    noMargin?: boolean;
    /**
     * Number of items to show outside the overflow menu
     *
     * @default 1
     */
    visibleItemsCount?: number;
    children: React.ReactNode;
}
export declare const Overflow: ({ children, noMargin, visibleItemsCount, ...props }: OverflowProps) => React.ReactElement;
//# sourceMappingURL=Overflow.d.ts.map