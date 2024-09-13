import type { UseComboboxReturnValue, UseMultipleSelectionReturnValue } from 'downshift';
import type { TypeaheadComboboxInputProps } from './TypeaheadComboboxInput';
export interface TypeaheadComboboxMultiSelectInputProps<T> extends Omit<TypeaheadComboboxInputProps<T>, 'componentId' | 'analyticsEvents'> {
    comboboxState: UseComboboxReturnValue<T>;
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
export declare const TypeaheadComboboxMultiSelectInput: React.FC<TypeaheadComboboxMultiSelectInputProps<any>>;
//# sourceMappingURL=TypeaheadComboboxMultiSelectInput.d.ts.map