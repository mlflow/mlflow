import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, expect, jest } from '@jest/globals';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { TypeaheadComboboxCheckboxItem, TypeaheadComboboxMenuItem } from '../../TypeaheadCombobox';
import { RHFControlledComponents } from '../RHFAdapters';
const books = [
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
];
const matcher = (book, query) => book.title.toLowerCase().includes(query) || book.author.toLowerCase().includes(query);
const RHFAdapterTypeaheadComboboxAsync = () => {
    const { control } = useForm();
    const [items, setItems] = useState([books[0]]);
    setTimeout(() => {
        setItems(books);
    }, 1000);
    return (_jsx(RHFControlledComponents.TypeaheadCombobox, { componentId: "codegen_design-system_src_design-system_formv2_tests_rhftypeaheadcombobox.test.tsx_39", control: control, name: "book", id: "book", rules: { required: true }, itemToString: (item) => item.title, allItems: items, matcher: matcher, children: ({ comboboxState, items }) => items.map((item, index) => (_jsx(TypeaheadComboboxMenuItem, { item: item, index: index, comboboxState: comboboxState, children: item.title }, `book-${item.title}`))) }));
};
const RHFAdapterTypeaheadComboboxMultiselectAsync = () => {
    const { control } = useForm();
    const [items, setItems] = useState([books[0]]);
    setTimeout(() => {
        setItems(books);
    }, 1000);
    return (_jsx(RHFControlledComponents.MultiSelectTypeaheadCombobox, { componentId: "codegen_design-system_src_design-system_formv2_tests_rhftypeaheadcombobox.test.tsx_68", control: control, name: "books", id: "books", rules: { required: true }, itemToString: (item) => item.title, allItems: items, matcher: matcher, inputProps: {
            width: 300,
            maxHeight: 160,
        }, menuProps: {
            width: 300,
            maxHeight: 500,
        }, children: ({ comboboxState, items, selectedItems }) => items.map((item, index) => (_jsx(TypeaheadComboboxCheckboxItem, { item: item, index: index, comboboxState: comboboxState, selectedItems: selectedItems, textOverflowMode: "ellipsis", children: item.title }, `book-${item.title}`))) }));
};
const allRegions = ['global', 'europe', 'america', 'asia', 'pacific'];
const SetValueMultiselectExample = () => {
    const { control, setValue, watch } = useForm();
    const selectedRegions = watch('region');
    // If user selects "global", remove all other selections
    if (selectedRegions?.length > 1 && selectedRegions.includes('global')) {
        setValue('region', ['global']);
    }
    return (_jsx(RHFControlledComponents.MultiSelectTypeaheadCombobox, { componentId: "codegen_design-system_src_design-system_formv2_tests_rhftypeaheadcombobox.test.tsx_118", control: control, name: "region", itemToString: (item) => item, allItems: allRegions, matcher: (item, query) => item.includes(query), children: ({ comboboxState, items, selectedItems }) => items.map((item, index) => (_jsx(TypeaheadComboboxCheckboxItem, { item: item, index: index, comboboxState: comboboxState, selectedItems: selectedItems, children: item }, `region-${item}`))) }));
};
const SetValueSingleSelectExample = () => {
    const { control, setValue } = useForm();
    const handleSetRegionToGlobal = () => {
        setValue('region', 'global');
    };
    return (_jsxs("div", { children: [_jsx(RHFControlledComponents.TypeaheadCombobox, { componentId: "codegen_design-system_src_design-system_formv2_tests_rhftypeaheadcombobox.test.tsx_153", control: control, name: "region", itemToString: (item) => item, allItems: allRegions, matcher: (item, query) => item.includes(query), children: ({ comboboxState, items }) => items.map((item, index) => (_jsx(TypeaheadComboboxMenuItem, { item: item, index: index, comboboxState: comboboxState, children: item }, item))) }), _jsx("button", { onClick: handleSetRegionToGlobal, children: "Set Region to Global" })] }));
};
const InputChangeSingleSelectExample = ({ onInputChange }) => {
    const { control } = useForm();
    return (_jsx(RHFControlledComponents.TypeaheadCombobox, { componentId: "codegen_design-system_src_design-system_formv2_tests_rhftypeaheadcombobox.test.tsx_189", control: control, name: "region", itemToString: (item) => item, allItems: allRegions, matcher: (item, query) => item.includes(query), onInputChange: onInputChange, children: ({ comboboxState, items }) => items.map((item, index) => (_jsx(TypeaheadComboboxMenuItem, { item: item, index: index, comboboxState: comboboxState, children: item }, item))) }));
};
const InputChangeMultiSelectExample = ({ onInputChange }) => {
    const { control } = useForm();
    return (_jsx(RHFControlledComponents.MultiSelectTypeaheadCombobox, { componentId: "codegen_design-system_src_design-system_formv2_tests_rhftypeaheadcombobox.test.tsx_219", control: control, name: "region", itemToString: (item) => item, allItems: allRegions, matcher: (item, query) => item.includes(query), onInputChange: onInputChange, children: ({ comboboxState, items, selectedItems }) => items.map((item, index) => (_jsx(TypeaheadComboboxCheckboxItem, { item: item, index: index, comboboxState: comboboxState, selectedItems: selectedItems, children: item }, `region-${item}`))) }));
};
describe('RHF Adapter TypeaheadCombobox', () => {
    it('loads async items into options list', async () => {
        render(_jsx(RHFAdapterTypeaheadComboboxAsync, {}));
        const input = screen.getByRole('textbox');
        await userEvent.click(input);
        const options = screen.getAllByRole('option');
        expect(options).toHaveLength(1);
        await new Promise((r) => setTimeout(r, 1000));
        const optionsAfter = screen.getAllByRole('option');
        expect(optionsAfter).toHaveLength(books.length);
    });
    it('loads async items into multiselect options list', async () => {
        render(_jsx(RHFAdapterTypeaheadComboboxMultiselectAsync, {}));
        const input = screen.getByRole('textbox');
        await userEvent.click(input);
        const options = screen.getAllByRole('option');
        expect(options).toHaveLength(1);
        await new Promise((r) => setTimeout(r, 1000));
        const optionsAfter = screen.getAllByRole('option');
        expect(optionsAfter).toHaveLength(books.length);
    });
    it('Works correctly with `setValue` (single select)', async () => {
        render(_jsx(SetValueSingleSelectExample, {}));
        const input = screen.getByRole('textbox');
        await userEvent.click(input);
        const options = screen.getAllByRole('option');
        expect(options).toHaveLength(allRegions.length);
        // select "europe"
        await userEvent.click(screen.getByRole('option', { name: 'europe' }));
        // click button, which will call setValue.
        const setRegionButton = screen.getByRole('button', { name: /Set Region to Global/i });
        await userEvent.click(setRegionButton);
        const globalRegionInput = screen.getByRole('textbox');
        expect(globalRegionInput).toHaveValue('global');
    });
    it('Works correctly with `setValue` (multiselect)', async () => {
        render(_jsx(SetValueMultiselectExample, {}));
        const input = screen.getByRole('textbox');
        await userEvent.click(input);
        const options = screen.getAllByRole('option');
        expect(options).toHaveLength(allRegions.length);
        // Select "europe" and "america"
        await userEvent.click(screen.getByRole('option', { name: 'europe' }));
        await userEvent.click(screen.getByRole('option', { name: 'america' }));
        // Close the menu so that the input measure node is not shown (otherwise the
        // last selected value will be used and the text will show up twice)
        await userEvent.click(input);
        const combobox = screen.getByRole('combobox');
        expect(within(combobox).getByText('europe')).toBeInTheDocument();
        expect(within(combobox).getByText('america')).toBeInTheDocument();
        // Select "global"
        await userEvent.click(screen.getByRole('option', { name: 'global' }));
        // Verify that only "global" is selected
        expect(within(combobox).getByText('global')).toBeInTheDocument();
        expect(within(combobox).queryByText('europe')).not.toBeInTheDocument();
        expect(within(combobox).queryByText('america')).not.toBeInTheDocument();
    });
    it('emits onInputChange event (single select)', async () => {
        const handleInputChange = jest.fn();
        render(_jsx(InputChangeSingleSelectExample, { onInputChange: handleInputChange }));
        const input = screen.getByRole('textbox');
        let inputsLength = 1; // Emmited once on initial render
        expect(handleInputChange).toHaveBeenCalledTimes(inputsLength);
        expect(handleInputChange).toHaveBeenCalledWith('');
        expect(input).toHaveValue('');
        const toBeTyped = 'ab';
        await userEvent.type(input, toBeTyped);
        inputsLength += toBeTyped.length;
        expect(handleInputChange).toHaveBeenCalledTimes(inputsLength);
        expect(handleInputChange).toHaveBeenCalledWith(toBeTyped);
        expect(input).toHaveValue(toBeTyped);
        const toBeTyped2 = 'new test value';
        await userEvent.type(input, toBeTyped2);
        inputsLength += toBeTyped2.length + 1; // +1 for the event clearing the input
        expect(handleInputChange).toHaveBeenCalledTimes(inputsLength);
        expect(handleInputChange).toHaveBeenCalledWith(toBeTyped2);
        expect(input).toHaveValue(toBeTyped2);
    });
    it('emits onInputChange event (multiselect)', async () => {
        const handleInputChange = jest.fn();
        render(_jsx(InputChangeMultiSelectExample, { onInputChange: handleInputChange }));
        const input = screen.getByRole('textbox');
        let inputsLength = 1; // Emmited once on initial render
        expect(handleInputChange).toHaveBeenCalledTimes(inputsLength);
        expect(handleInputChange).toHaveBeenCalledWith('');
        expect(input).toHaveValue('');
        const toBeTyped = 'ab';
        await userEvent.type(input, toBeTyped);
        inputsLength += toBeTyped.length;
        expect(handleInputChange).toHaveBeenCalledTimes(inputsLength);
        expect(handleInputChange).toHaveBeenCalledWith(toBeTyped);
        expect(input).toHaveValue(toBeTyped);
        const toBeTyped2 = 'c';
        await userEvent.type(input, toBeTyped2);
        inputsLength += toBeTyped2.length;
        expect(handleInputChange).toHaveBeenCalledTimes(inputsLength);
        expect(handleInputChange).toHaveBeenCalledWith(toBeTyped + toBeTyped2);
        expect(input).toHaveValue(toBeTyped + toBeTyped2);
    });
});
//# sourceMappingURL=RHFTypeaheadCombobox.test.js.map