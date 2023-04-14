import type { SerializedStyles } from '@emotion/react';
import * as Popover from '@radix-ui/react-popover';
import React from 'react';
import type { HTMLDataAttributes } from '../types';
export interface DialogComboboxTriggerProps extends Popover.PopoverTriggerProps, HTMLDataAttributes {
    maxWidth?: number;
    removable?: boolean;
    onRemove?: () => void;
    allowClear?: boolean;
    onClear?: () => void;
    showTagAfterValueCount?: number;
    controlled?: boolean;
    wrapperProps?: {
        css?: SerializedStyles;
    } & React.HTMLAttributes<HTMLDivElement>;
    width?: number | string;
    withChevronIcon?: boolean;
}
export declare const DialogComboboxTrigger: React.ForwardRefExoticComponent<DialogComboboxTriggerProps & React.RefAttributes<HTMLButtonElement>>;
//# sourceMappingURL=DialogComboboxTrigger.d.ts.map