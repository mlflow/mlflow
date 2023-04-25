/// <reference types="react" />
import * as Popover from '@radix-ui/react-popover';
import type { HTMLDataAttributes } from '../types';
export interface DialogComboboxContentProps extends Popover.PopoverContentProps, HTMLDataAttributes {
    width?: number;
    loading?: boolean;
    maxHeight?: number;
    maxWidth?: number;
    minHeight?: number;
    minWidth?: number;
    side?: 'top' | 'bottom';
}
export declare const DialogComboboxContent: import("react").ForwardRefExoticComponent<DialogComboboxContentProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=DialogComboboxContent.d.ts.map