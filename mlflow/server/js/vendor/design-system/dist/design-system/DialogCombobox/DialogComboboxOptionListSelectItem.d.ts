/// <reference types="react" />
import type { HTMLDataAttributes } from '../types';
export interface DialogComboboxOptionListSelectItemProps extends HTMLDataAttributes, Omit<React.HTMLAttributes<HTMLDivElement>, 'onChange'> {
    value: string;
    checked?: boolean;
    disabled?: boolean;
    disabledReason?: React.ReactNode;
    children?: React.ReactNode;
    onChange?: (value: any, event?: React.MouseEvent<HTMLDivElement> | React.KeyboardEvent<HTMLDivElement>) => void;
    hintColumn?: React.ReactNode;
    hintColumnWidthPercent?: number;
    _TYPE?: string;
    icon?: React.ReactNode;
    /**
     * In certain very custom instances you may wish to hide the check; this is not recommended.
     * If the check is hidden, the user will not be able to tell which item is selected.
     */
    dangerouslyHideCheck?: boolean;
}
export declare const DialogComboboxOptionListSelectItem: import("react").ForwardRefExoticComponent<DialogComboboxOptionListSelectItemProps & import("react").RefAttributes<HTMLDivElement>>;
export { getComboboxOptionItemWrapperStyles, getComboboxOptionLabelStyles } from '../_shared_/Combobox';
//# sourceMappingURL=DialogComboboxOptionListSelectItem.d.ts.map