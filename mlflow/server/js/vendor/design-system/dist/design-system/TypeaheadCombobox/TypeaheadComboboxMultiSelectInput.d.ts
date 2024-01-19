/// <reference types="react" />
import type { UseComboboxReturnValue, UseMultipleSelectionReturnValue } from 'downshift';
import type { TypeaheadComboboxInputProps } from './TypeaheadComboboxInput';
export interface TypeaheadComboboxMultiSelectInputProps<T> extends TypeaheadComboboxInputProps<T> {
    comboboxState: UseComboboxReturnValue<T>;
    multipleSelectionState: UseMultipleSelectionReturnValue<T>;
    selectedItems: T[];
    setSelectedItems: React.Dispatch<React.SetStateAction<T[]>>;
    getSelectedItemLabel: (item: T) => React.ReactNode;
    allowClear?: boolean;
    showTagAfterValueCount?: number;
    width?: string | number;
    maxHeight?: string | number;
}
export declare const TypeaheadComboboxMultiSelectInput: React.FC<TypeaheadComboboxMultiSelectInputProps<any>>;
//# sourceMappingURL=TypeaheadComboboxMultiSelectInput.d.ts.map