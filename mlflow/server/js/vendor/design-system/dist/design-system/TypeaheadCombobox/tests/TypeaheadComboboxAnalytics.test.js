import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { TypeaheadComboboxRoot, TypeaheadComboboxMultiSelectInput, TypeaheadComboboxMenu, TypeaheadComboboxCheckboxItem, useComboboxState, useMultipleSelectionState, TypeaheadComboboxInput, TypeaheadComboboxMenuItem, DesignSystemEventProviderAnalyticsEventTypes, TypeaheadComboboxFooter, TypeaheadComboboxAddButton, setupDesignSystemEventProviderForTesting, } from '@databricks/design-system';
import { beforeEach, describe, jest, it, expect } from '@jest/globals';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';
import { books, clickClearButton, selectItemByText } from './testFixtures';
import { setupSafexTesting } from '../../utils/safex';
const matcher = (book, query) => book.title.toLowerCase().includes(query) || book.author.toLowerCase().includes(query);
function getFilteredBooks(inputValue) {
    const lowerCasedInputValue = inputValue.toLowerCase();
    return books.filter((book) => book.title.toLowerCase().includes(lowerCasedInputValue) ||
        book.author.toLowerCase().includes(lowerCasedInputValue));
}
function MultiSelectComboBox() {
    const [inputValue, setInputValue] = React.useState('');
    const [selectedItems, setSelectedItems] = React.useState(books.slice(0, 2));
    const items = React.useMemo(() => getFilteredBooks(inputValue), [inputValue]);
    const itemToString = (item) => item.title;
    const multipleSelectionState = useMultipleSelectionState(selectedItems, setSelectedItems, {
        componentId: 'book-test',
        analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
        valueHasNoPii: true,
        itemToString,
    });
    const comboboxState = useComboboxState({
        allItems: books,
        items,
        setInputValue,
        matcher,
        itemToString,
        multiSelect: true,
        selectedItems,
        setSelectedItems,
        componentId: 'book-test',
        analyticsEvents: [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView,
        ],
        valueHasNoPii: true,
    });
    return (_jsxs(TypeaheadComboboxRoot, { id: "books", multiSelect: true, comboboxState: comboboxState, children: [_jsx(TypeaheadComboboxMultiSelectInput, { placeholder: "Choose books", comboboxState: comboboxState, multipleSelectionState: multipleSelectionState, selectedItems: selectedItems, setSelectedItems: setSelectedItems, getSelectedItemLabel: (item) => item.title }), _jsx(TypeaheadComboboxMenu, { comboboxState: comboboxState, children: items.map((item, index) => (_jsx(TypeaheadComboboxCheckboxItem, { item: item, index: index, comboboxState: comboboxState, selectedItems: selectedItems, children: item.title }, `book-${item.title}`))) })] }));
}
function SingleSelectCombobox({ valueHasNoPii }) {
    const [items, setItems] = React.useState(books);
    const handleAdd = () => {
        setItems([...items, { author: `New author ${items.length + 1}`, title: `New book ${items.length + 1}` }]);
    };
    const comboboxState = useComboboxState({
        allItems: books,
        items,
        setItems,
        itemToString: (item) => item.title,
        matcher,
        componentId: 'book-test',
        valueHasNoPii,
    });
    return (_jsxs(TypeaheadComboboxRoot, { id: "book", comboboxState: comboboxState, children: [_jsx(TypeaheadComboboxInput, { comboboxState: comboboxState }), _jsxs(TypeaheadComboboxMenu, { comboboxState: comboboxState, children: [items.map((item, index) => (_jsx(TypeaheadComboboxMenuItem, { item: item, index: index, comboboxState: comboboxState, children: item.title }, `book-${item.title}`))), _jsx(TypeaheadComboboxFooter, { children: _jsx(TypeaheadComboboxAddButton, { componentId: "add_book", onClick: handleAdd, children: "Add new book" }) })] })] }));
}
describe('TypeaheadComboboxAnalytics', () => {
    const { setSafex } = setupSafexTesting();
    const eventCallback = jest.fn();
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    beforeEach(() => {
        setSafex({
            'databricks.fe.observability.defaultComponentView.typeaheadCombobox': true,
            'databricks.fe.observability.defaultButtonComponentView': false,
            'databricks.fe.observability.defaultComponentView.input': false,
            'databricks.fe.observability.defaultComponentView.checkbox': false,
        });
        // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
        window.IntersectionObserver = undefined;
    });
    it('emits TypeAheadCombobox Analytics single select events', async () => {
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(SingleSelectCombobox, { valueHasNoPii: true }) }));
        await waitFor(() => {
            expect(screen.getByRole('textbox')).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentId: 'book-test',
            componentType: 'typeahead_combobox',
            shouldStartInteraction: false,
            value: '',
        });
        // Type in the first option and select it.
        const singleSelectCombobox = screen.getByRole('textbox');
        await userEvent.click(singleSelectCombobox);
        await userEvent.type(singleSelectCombobox, 'mockingbird');
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'book-test.input',
            componentType: 'input',
            shouldStartInteraction: false,
            value: undefined,
        });
        await selectItemByText('To Kill a Mockingbird');
        expect(eventCallback).toHaveBeenCalledTimes(3);
        expect(eventCallback).toHaveBeenNthCalledWith(3, {
            eventType: 'onValueChange',
            componentId: 'book-test',
            componentType: 'typeahead_combobox',
            shouldStartInteraction: false,
            value: 'To Kill a Mockingbird',
        });
        // Clear the selection.
        await clickClearButton();
        const combobox = screen.getByRole('combobox');
        await waitFor(() => expect(within(combobox).queryByText('To Kill a Mockingbird')).toBeNull());
        expect(eventCallback).toHaveBeenCalledTimes(4);
        expect(eventCallback).toHaveBeenNthCalledWith(4, {
            eventType: 'onValueChange',
            componentId: 'book-test',
            componentType: 'typeahead_combobox',
            shouldStartInteraction: false,
            value: '',
        });
        // Add a new item and select it.
        await userEvent.click(singleSelectCombobox);
        const addButton = screen.getByRole('button', { name: 'Add new book' });
        await userEvent.click(addButton);
        expect(eventCallback).toHaveBeenCalledTimes(5);
        expect(eventCallback).toHaveBeenNthCalledWith(5, {
            eventType: 'onClick',
            componentId: 'book-test.add_option',
            componentType: 'button',
            shouldStartInteraction: true,
            isInteractionSubject: true,
            event: expect.anything(),
            value: undefined,
        });
        await selectItemByText(`New book ${books.length + 1}`);
        expect(eventCallback).toHaveBeenCalledTimes(6);
        expect(eventCallback).toHaveBeenNthCalledWith(6, {
            eventType: 'onValueChange',
            componentId: 'book-test',
            componentType: 'typeahead_combobox',
            shouldStartInteraction: false,
            value: `New book ${books.length + 1}`,
        });
    });
    it('emits TypeAheadCombobox Analytics multi select events', async () => {
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(MultiSelectComboBox, {}) }));
        await waitFor(() => {
            expect(screen.getByRole('textbox')).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentId: 'book-test',
            componentType: 'typeahead_combobox',
            shouldStartInteraction: false,
            value: '["To Kill a Mockingbird","War and Peace"]',
        });
        // Initially, both options are selected. Clear the selections.
        await clickClearButton();
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'book-test',
            componentType: 'typeahead_combobox',
            shouldStartInteraction: false,
            value: '[]',
        });
        // Select both options.
        await selectItemByText('To Kill a Mockingbird');
        expect(eventCallback).toHaveBeenCalledTimes(3);
        expect(eventCallback).toHaveBeenNthCalledWith(3, {
            eventType: 'onValueChange',
            componentId: 'book-test',
            componentType: 'typeahead_combobox',
            shouldStartInteraction: false,
            value: '["To Kill a Mockingbird"]',
        });
        await selectItemByText('War and Peace');
        expect(eventCallback).toHaveBeenCalledTimes(4);
        expect(eventCallback).toHaveBeenNthCalledWith(4, {
            eventType: 'onValueChange',
            componentId: 'book-test',
            componentType: 'typeahead_combobox',
            shouldStartInteraction: false,
            value: '["To Kill a Mockingbird","War and Peace"]',
        });
        // Deselect option with keyboard.
        await userEvent.click(screen.getByRole('textbox'));
        await userEvent.keyboard('{backspace}');
        expect(eventCallback).toHaveBeenCalledTimes(5);
        expect(eventCallback).toHaveBeenNthCalledWith(5, {
            eventType: 'onValueChange',
            componentId: 'book-test',
            componentType: 'typeahead_combobox',
            shouldStartInteraction: false,
            value: '["To Kill a Mockingbird"]',
        });
        // Deselect option with mouse.
        await selectItemByText('To Kill a Mockingbird');
        expect(eventCallback).toHaveBeenCalledTimes(6);
        expect(eventCallback).toHaveBeenNthCalledWith(6, {
            eventType: 'onValueChange',
            componentId: 'book-test',
            componentType: 'typeahead_combobox',
            shouldStartInteraction: false,
            value: '[]',
        });
        // Select an option by typing. multiSelect does not use an input component, so the input event is not fired.
        const multiSelectCombobox = screen.getByRole('textbox');
        await userEvent.click(multiSelectCombobox);
        await userEvent.type(multiSelectCombobox, 'mockingbird');
        await selectItemByText('To Kill a Mockingbird');
        expect(eventCallback).toHaveBeenCalledTimes(7);
        expect(eventCallback).toHaveBeenNthCalledWith(7, {
            eventType: 'onValueChange',
            componentId: 'book-test',
            componentType: 'typeahead_combobox',
            shouldStartInteraction: false,
            value: '["To Kill a Mockingbird"]',
        });
    });
});
//# sourceMappingURL=TypeaheadComboboxAnalytics.test.js.map