/// <reference types="react" />
import * as Popover from '@radix-ui/react-popover';
import type { HTMLDataAttributes } from '../../design-system/types';
export interface DialogComboboxProps extends Popover.PopoverProps, HTMLDataAttributes {
    label: string | React.ReactNode;
    value?: string[];
    stayOpenOnSelection?: boolean;
    multiSelect?: boolean;
    emptyText?: string;
    scrollToSelectedElement?: boolean;
    rememberLastScrollPosition?: boolean;
}
export declare const DialogCombobox: ({ children, label, value, open, emptyText, scrollToSelectedElement, rememberLastScrollPosition, ...props }: DialogComboboxProps) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=DialogCombobox.d.ts.map