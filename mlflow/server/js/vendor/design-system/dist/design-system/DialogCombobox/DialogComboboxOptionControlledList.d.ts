/// <reference types="react" />
import type { DialogComboboxOptionListProps } from './DialogComboboxOptionList';
export interface DialogComboboxOptionControlledListProps extends Omit<DialogComboboxOptionListProps, 'children'> {
    withSearch?: boolean;
    showSelectAndClearAll?: boolean;
    options: string[];
    onChange?: (...args: any[]) => any;
}
export declare const DialogComboboxOptionControlledList: import("react").ForwardRefExoticComponent<DialogComboboxOptionControlledListProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=DialogComboboxOptionControlledList.d.ts.map