/// <reference types="react" />
import { type UseComboboxReturnValue, type UseMultipleSelectionReturnValue, type UseComboboxStateChange } from 'downshift';
interface SingleSelectProps<T> extends CommonComboboxStateProps<T> {
    multiSelect?: false;
    setItems: React.Dispatch<React.SetStateAction<T[]>>;
}
interface MultiSelectProps<T> extends CommonComboboxStateProps<T> {
    multiSelect: true;
    setInputValue: React.Dispatch<React.SetStateAction<string>>;
    selectedItems: T[];
    setSelectedItems: React.Dispatch<React.SetStateAction<T[]>>;
}
interface CommonComboboxStateProps<T> {
    allItems: T[];
    items: T[];
    matcher: (item: T, searchQuery: string) => boolean;
    itemToString?: (item: T) => string;
    onIsOpenChange?: (changes: UseComboboxStateChange<T>) => void;
}
export type UseComboboxStateProps<T> = SingleSelectProps<T> | MultiSelectProps<T>;
export declare function useComboboxState<T>({ allItems, items, itemToString, onIsOpenChange, ...props }: UseComboboxStateProps<T>): UseComboboxReturnValue<T>;
export declare function useMultipleSelectionState<T>(selectedItems: T[], setSelectedItems: React.Dispatch<React.SetStateAction<T[]>>): UseMultipleSelectionReturnValue<T>;
export {};
//# sourceMappingURL=downshiftHookWrappers.d.ts.map