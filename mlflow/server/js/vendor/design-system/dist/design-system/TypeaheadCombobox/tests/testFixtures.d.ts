import type { TypeaheadComboboxInputProps } from '../TypeaheadComboboxInput';
import type { TypeaheadComboboxMultiSelectInputProps } from '../TypeaheadComboboxMultiSelectInput';
export interface Book {
    author: string;
    title: string;
}
export interface renderMultiSelectTypeaheadProps {
    preSelectedItems?: Book[];
    multiSelectInputProps?: Partial<TypeaheadComboboxMultiSelectInputProps<Book>>;
}
export interface renderSingleSelectTypeaheadProps {
    initialInputValue?: string;
    initialSelectedItem?: Book;
    selectInputProps?: Partial<TypeaheadComboboxInputProps<Book>>;
}
export declare const books: Book[];
export declare const matcher: (book: Book, query: string) => boolean;
export declare const getFilteredBooks: (inputValue: string) => Book[];
export declare const renderSingleSelectTypeahead: ({ initialSelectedItem, initialInputValue, selectInputProps, }?: renderSingleSelectTypeaheadProps) => void;
export declare const renderMultiSelectTypeahead: ({ preSelectedItems, multiSelectInputProps, }?: renderMultiSelectTypeaheadProps) => void;
export declare const renderSingleSelectTypeaheadWithLabel: ({ initialSelectedItem, initialInputValue, selectInputProps, }?: renderSingleSelectTypeaheadProps) => void;
export declare const openMenuWithButton: () => Promise<void>;
export declare const closeMenu: () => Promise<void>;
export declare const selectItemByText: (text: string) => Promise<void>;
export declare const clickClearButton: () => Promise<void>;
//# sourceMappingURL=testFixtures.d.ts.map