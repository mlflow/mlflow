import type { UseComboboxReturnValue } from 'downshift';
import { type TypeaheadComboboxMenuItemProps } from './TypeaheadComboboxMenuItem';
export interface TypeaheadComboboxCheckboxItemProps<T> extends TypeaheadComboboxMenuItemProps<T> {
    comboboxState: UseComboboxReturnValue<T>;
    selectedItems: T[];
    selectedMatcher?: (a: T, b: T) => boolean;
    _type?: string;
}
export declare const TypeaheadComboboxCheckboxItem: import("react").ForwardRefExoticComponent<TypeaheadComboboxCheckboxItemProps<any> & import("react").RefAttributes<HTMLLIElement>>;
export default TypeaheadComboboxCheckboxItem;
//# sourceMappingURL=TypeaheadComboboxCheckboxItem.d.ts.map