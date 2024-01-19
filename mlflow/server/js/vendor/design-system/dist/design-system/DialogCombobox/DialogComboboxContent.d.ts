/// <reference types="react" />
import * as Popover from '@radix-ui/react-popover';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import type { HTMLDataAttributes } from '../types';
export interface DialogComboboxContentProps extends Popover.PopoverContentProps, HTMLDataAttributes, WithLoadingState {
    width?: number | string;
    loading?: boolean;
    maxHeight?: number;
    maxWidth?: number;
    minHeight?: number;
    minWidth?: number;
    side?: 'top' | 'bottom';
    matchTriggerWidth?: boolean;
    textOverflowMode?: 'ellipsis' | 'multiline';
}
export declare const DialogComboboxContent: import("react").ForwardRefExoticComponent<DialogComboboxContentProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=DialogComboboxContent.d.ts.map