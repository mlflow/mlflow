import type { SerializedStyles } from '@emotion/react';
import * as Popover from '@radix-ui/react-popover';
import React from 'react';
import type { FormElementValidationState, HTMLDataAttributes } from '../types';
export interface DialogComboboxTriggerProps extends Popover.PopoverTriggerProps, FormElementValidationState, HTMLDataAttributes {
    minWidth?: number | string;
    maxWidth?: number | string;
    width?: number | string;
    removable?: boolean;
    onRemove?: () => void;
    allowClear?: boolean;
    onClear?: () => void;
    showTagAfterValueCount?: number;
    controlled?: boolean;
    wrapperProps?: {
        css?: SerializedStyles;
    } & React.HTMLAttributes<HTMLDivElement>;
    withChevronIcon?: boolean;
    withInlineLabel?: boolean;
    isBare?: boolean;
}
export declare const DialogComboboxTrigger: React.ForwardRefExoticComponent<DialogComboboxTriggerProps & React.RefAttributes<HTMLButtonElement>>;
//# sourceMappingURL=DialogComboboxTrigger.d.ts.map