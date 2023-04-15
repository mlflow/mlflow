/// <reference types="react" />
import type { HTMLDataAttributes } from '../types';
export interface DialogComboboxOptionListCheckboxItemProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement> {
    value: string;
    checked?: boolean;
    disabled?: boolean;
    disabledReason?: React.ReactNode;
    indeterminate?: boolean;
    children?: React.ReactNode;
    onChange?: (...args: any[]) => any;
}
export declare const DialogComboboxOptionListCheckboxItem: import("react").ForwardRefExoticComponent<DialogComboboxOptionListCheckboxItemProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=DialogComboboxOptionListCheckboxItem.d.ts.map