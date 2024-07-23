import type { SerializedStyles } from '@emotion/react';
import * as Popover from '@radix-ui/react-popover';
import type { ReactNode } from 'react';
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
    formatDisplayedValue?: (value: string) => string | ReactNode;
}
export declare const DialogComboboxTrigger: React.ForwardRefExoticComponent<DialogComboboxTriggerProps & React.RefAttributes<HTMLButtonElement>>;
interface DialogComboboxIconButtonTriggerProps extends Popover.PopoverTriggerProps {
}
/**
 * A custom button trigger that can be wrapped around any button.
 */
export declare const DialogComboboxCustomButtonTriggerWrapper: ({ children }: DialogComboboxIconButtonTriggerProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=DialogComboboxTrigger.d.ts.map