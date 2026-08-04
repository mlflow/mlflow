import type { UseMultipleSelectionReturnValue } from 'downshift';
import type { TypeaheadComboboxInputProps } from './TypeaheadComboboxInput';
import type { ComboboxStateAnalyticsReturnValue } from './hooks';
export interface TypeaheadComboboxMultiSelectInputProps<T> extends TypeaheadComboboxInputProps<T> {
    comboboxState: ComboboxStateAnalyticsReturnValue<T>;
    multipleSelectionState: UseMultipleSelectionReturnValue<T>;
    selectedItems: T[];
    setSelectedItems: React.Dispatch<React.SetStateAction<T[]>>;
    getSelectedItemLabel: (item: T) => React.ReactNode;
    allowClear?: boolean;
    showTagAfterValueCount?: number;
    width?: string | number;
    maxHeight?: string | number;
    disableTooltip?: boolean;
}
export declare const TypeaheadComboboxMultiSelectInput: import("react").ForwardRefExoticComponent<TypeaheadComboboxMultiSelectInputProps<any> & import("react").RefAttributes<HTMLInputElement>>;
//# sourceMappingURL=TypeaheadComboboxMultiSelectInput.d.ts.map