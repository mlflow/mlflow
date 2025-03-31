import type { HTMLDataAttributes } from '../types';
export interface DialogComboboxOptionListCheckboxItemProps extends HTMLDataAttributes, Omit<React.HTMLAttributes<HTMLDivElement>, 'onChange'> {
    value: string;
    checked?: boolean;
    disabled?: boolean;
    disabledReason?: React.ReactNode;
    indeterminate?: boolean;
    children?: React.ReactNode;
    onChange?: (value: any, event?: React.MouseEvent<HTMLDivElement> | React.KeyboardEvent<HTMLDivElement>) => void;
    _TYPE?: string;
}
export declare const DialogComboboxOptionListCheckboxItem: import("react").ForwardRefExoticComponent<DialogComboboxOptionListCheckboxItemProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=DialogComboboxOptionListCheckboxItem.d.ts.map