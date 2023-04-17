/// <reference types="react" />
import * as Popover from '@radix-ui/react-popover';
import type { HTMLDataAttributes } from '../../design-system/types';
export interface FilterContentProps extends Popover.PopoverContentProps, HTMLDataAttributes {
    width?: number;
    loading?: boolean;
    maxHeight?: number;
    maxWidth?: number;
    minHeight?: number;
    minWidth?: number;
    side?: 'top' | 'bottom';
}
export declare const FilterContent: import("react").ForwardRefExoticComponent<FilterContentProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=FilterContent.d.ts.map