import type { DialogComboboxOptionListProps } from './DialogComboboxOptionList';
import type { WithLoadingState } from '../LoadingState/LoadingState';
export interface DialogComboboxOptionControlledListProps extends Omit<DialogComboboxOptionListProps, 'children'>, WithLoadingState {
    withSearch?: boolean;
    showAllOption?: boolean;
    allOptionLabel?: string;
    options: string[];
    onChange?: (...args: any[]) => any;
}
export declare const DialogComboboxOptionControlledList: import("react").ForwardRefExoticComponent<DialogComboboxOptionControlledListProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=DialogComboboxOptionControlledList.d.ts.map