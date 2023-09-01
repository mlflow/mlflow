/// <reference types="react" />
import type { HTMLDataAttributes } from '../types';
export interface DialogComboboxOptionListSelectItemProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement> {
    value: string;
    checked?: boolean;
    disabled?: boolean;
    disabledReason?: React.ReactNode;
    children?: React.ReactNode;
    onChange?: (...args: any[]) => any;
    _TYPE?: string;
}
export declare const DialogComboboxOptionListSelectItem: import("react").ForwardRefExoticComponent<DialogComboboxOptionListSelectItemProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=DialogComboboxOptionListSelectItem.d.ts.map