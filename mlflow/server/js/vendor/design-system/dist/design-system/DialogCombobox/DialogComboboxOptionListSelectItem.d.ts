/// <reference types="react" />
import type { HTMLDataAttributes } from '../types';
export interface DialogComboboxOptionListSelectItemProps extends HTMLDataAttributes, Omit<React.HTMLAttributes<HTMLDivElement>, 'onChange'> {
    value: string;
    checked?: boolean;
    disabled?: boolean;
    disabledReason?: React.ReactNode;
    children?: React.ReactNode;
    onChange?: (value: any, event?: React.MouseEvent<HTMLDivElement> | React.KeyboardEvent<HTMLDivElement>) => void;
    _TYPE?: string;
}
export declare const DialogComboboxOptionListSelectItem: import("react").ForwardRefExoticComponent<DialogComboboxOptionListSelectItemProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=DialogComboboxOptionListSelectItem.d.ts.map