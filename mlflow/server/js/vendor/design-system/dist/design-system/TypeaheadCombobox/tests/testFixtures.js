import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';
import { TypeaheadComboboxCheckboxItem, TypeaheadComboboxInput, TypeaheadComboboxMenu, TypeaheadComboboxMenuItem, TypeaheadComboboxMultiSelectInput, TypeaheadComboboxRoot, } from '..';
import { DesignSystemProvider } from '../../DesignSystemProvider';
import { FormUI } from '../../FormV2';
import { useComboboxState, useMultipleSelectionState } from '../hooks';
export const books = [
    { author: 'Harper Lee', title: 'To Kill a Mockingbird' },
    { author: 'Lev Tolstoy', title: 'War and Peace' },
    { author: 'Fyodor Dostoyevsy', title: 'The Idiot' },
    { author: 'Oscar Wilde', title: 'A Picture of Dorian Gray' },
    { author: 'George Orwell', title: '1984' },
    { author: 'Jane Austen', title: 'Pride and Prejudice' },
    { author: 'Marcus Aurelius', title: 'Meditations' },
    { author: 'Fyodor Dostoevsky', title: 'The Brothers Karamazov' },
    { author: 'Lev Tolstoy', title: 'Anna Karenina' },
    { author: 'Fyodor Dostoevsky', title: 'Crime and Punishment' },
    { author: 'F. Scott Fitzgerald', title: 'The Great Gatsby' },
    { author: 'Emily Bronte', title: 'Wuthering Heights' },
    { author: 'J.R.R. Tolkien', title: 'The Lord of the Rings' },
    { author: 'Bram Stoker', title: 'Dracula' },
    { author: 'Charles Dickens', title: 'Great Expectations' },
    { author: 'Herman Melville', title: 'Moby Dick' },
    { author: 'J.D. Salinger', title: 'The Catcher in the Rye' },
    { author: 'Louisa May Alcott', title: 'Little Women' },
    { author: 'Mark Twain', title: 'Adventures of Huckleberry Finn' },
    { author: 'William Golding', title: 'Lord of the Flies' },
    { author: 'George Orwell', title: 'Animal Farm' },
    { author: 'Charles Dickens', title: 'A Tale of Two Cities' },
    { author: 'George Eliot', title: 'Middlemarch' },
    { author: 'Miguel de Cervantes', title: 'Don Quixote' },
    { author: 'Aldous Huxley', title: 'Brave New World' },
    { author: 'Nathaniel Hawthorne', title: 'The Scarlet Letter' },
    { author: 'Ray Bradbury', title: 'Fahrenheit 451' },
];
export const matcher = (book, query) => book.title.toLowerCase().includes(query) || book.author.toLowerCase().includes(query);
export const getFilteredBooks = (inputValue) => {
    const lowerCasedInputValue = inputValue.toLowerCase();
    return books.filter((book) => book.title.toLowerCase().includes(lowerCasedInputValue) ||
        book.author.toLowerCase().includes(lowerCasedInputValue));
};
export const renderSingleSelectTypeahead = ({ initialSelectedItem, initialInputValue, selectInputProps = {}, } = {}) => {
    const SingleSelectTypeahead = () => {
        const [items, setItems] = React.useState(books);
        const comboboxState = useComboboxState({
            componentId: 'codegen_design-system_src_design-system_typeaheadcombobox_tests_testfixtures.tsx_96',
            allItems: books,
            items,
            setItems,
            itemToString: (item) => item.title,
            matcher,
            initialInputValue,
            initialSelectedItem,
        });
        return (_jsx(DesignSystemProvider, { children: _jsxs(TypeaheadComboboxRoot, { id: "book", comboboxState: comboboxState, children: [_jsx(TypeaheadComboboxInput, { placeholder: "Choose an option", comboboxState: comboboxState, ...selectInputProps }), _jsx(TypeaheadComboboxMenu, { comboboxState: comboboxState, children: items.map((item, index) => (_jsx(TypeaheadComboboxMenuItem, { item: item, index: index, comboboxState: comboboxState, children: item.title }, `book-${item.title}`))) })] }) }));
    };
    render(_jsx(SingleSelectTypeahead, {}));
};
export const renderMultiSelectTypeahead = ({ preSelectedItems = [], multiSelectInputProps = {}, } = {}) => {
    const MultiSelectTypeahead = () => {
        const [inputValue, setInputValue] = React.useState('');
        const [selectedItems, setSelectedItems] = React.useState(preSelectedItems);
        const items = React.useMemo(() => getFilteredBooks(inputValue), [inputValue]);
        const comboboxState = useComboboxState({
            componentId: 'codegen_design-system_src_design-system_typeaheadcombobox_tests_testfixtures.tsx_146',
            allItems: books,
            items,
            setInputValue,
            matcher,
            itemToString: (item) => item.title,
            multiSelect: true,
            selectedItems,
            setSelectedItems,
        });
        const multipleSelectionState = useMultipleSelectionState(selectedItems, setSelectedItems, comboboxState);
        return (_jsx(DesignSystemProvider, { children: _jsxs(TypeaheadComboboxRoot, { id: "books", multiSelect: true, comboboxState: comboboxState, children: [_jsx(TypeaheadComboboxMultiSelectInput, { placeholder: selectedItems.length ? '' : 'Choose books', comboboxState: comboboxState, multipleSelectionState: multipleSelectionState, selectedItems: selectedItems, setSelectedItems: setSelectedItems, getSelectedItemLabel: (item) => item.title, ...multiSelectInputProps }), _jsx(TypeaheadComboboxMenu, { comboboxState: comboboxState, width: 300, children: items.map((item, index) => (_jsx(TypeaheadComboboxCheckboxItem, { item: item, index: index, comboboxState: comboboxState, selectedItems: selectedItems, textOverflowMode: "ellipsis", children: item.title }, `book-${item.title}`))) })] }) }));
    };
    render(_jsx(MultiSelectTypeahead, {}));
};
export const renderSingleSelectTypeaheadWithLabel = ({ initialSelectedItem, initialInputValue, selectInputProps = {}, } = {}) => {
    const SingleSelectTypeahead = () => {
        const [items, setItems] = React.useState(books);
        const comboboxState = useComboboxState({
            componentId: 'codegen_design-system_src_design-system_typeaheadcombobox_tests_testfixtures.tsx_96',
            allItems: books,
            items,
            setItems,
            itemToString: (item) => item.title,
            matcher,
            initialInputValue,
            initialSelectedItem,
        });
        return (_jsxs(DesignSystemProvider, { children: [_jsx(FormUI.Label, { htmlFor: "book", ...comboboxState.getLabelProps(), children: "Favorite book" }), _jsxs(TypeaheadComboboxRoot, { id: "book", comboboxState: comboboxState, children: [_jsx(TypeaheadComboboxInput, { placeholder: "Choose an option", comboboxState: comboboxState, ...selectInputProps }), _jsx(TypeaheadComboboxMenu, { comboboxState: comboboxState, children: items.map((item, index) => (_jsx(TypeaheadComboboxMenuItem, { item: item, index: index, comboboxState: comboboxState, children: item.title }, `book-${item.title}`))) })] })] }));
    };
    render(_jsx(SingleSelectTypeahead, {}));
};
export const openMenuWithButton = async () => {
    const toggleButton = screen.getByRole('button', { name: 'toggle menu' });
    await userEvent.click(toggleButton);
    expect(screen.getByRole('listbox')).toBeVisible();
};
export const closeMenu = async () => {
    const toggleButton = screen.getByRole('button');
    await userEvent.click(toggleButton);
    expect(screen.queryByRole('listbox')).toBeNull();
};
export const selectItemByText = async (text) => {
    await userEvent.click(screen.getByRole('option', { name: text }));
};
export const clickClearButton = async () => {
    const clearButton = screen.getByRole('button', { name: 'Clear selection' });
    await userEvent.click(clearButton);
};
//# sourceMappingURL=testFixtures.js.map