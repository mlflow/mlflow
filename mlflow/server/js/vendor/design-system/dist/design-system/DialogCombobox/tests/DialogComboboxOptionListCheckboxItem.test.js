import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, afterEach, jest, it, expect } from '@jest/globals';
import { fireEvent, render, waitFor } from '@testing-library/react';
import { DialogCombobox } from '../DialogCombobox';
import { DialogComboboxContent } from '../DialogComboboxContent';
import { DialogComboboxOptionList } from '../DialogComboboxOptionList';
import { DialogComboboxOptionListCheckboxItem } from '../DialogComboboxOptionListCheckboxItem';
import { DialogComboboxTrigger } from '../DialogComboboxTrigger';
describe('Dialog Combobox - CheckboxItem', () => {
    afterEach(() => {
        jest.clearAllMocks();
    });
    it('renders', async () => {
        const options = ['value 1', 'value 2', 'value 3'];
        const { getByRole, queryAllByRole } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistcheckboxitem.test.tsx_18", label: "example filter", open: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionList, { children: options.map((option) => (_jsx(DialogComboboxOptionListCheckboxItem, { value: option, children: option }, option))) }) })] }));
        await waitFor(() => {
            expect(getByRole('listbox')).toBeVisible();
        });
        expect(getByRole('listbox')).toBeVisible();
        expect(queryAllByRole('option')).toHaveLength(3);
    });
    it('does not render outside of DialogComboboxOptionList', async () => {
        jest.spyOn(console, 'error').mockImplementation(() => { });
        expect(() => render(_jsx(DialogComboboxOptionListCheckboxItem, { value: "value 1", children: "value 1" }))).toThrowError('`DialogComboboxOptionListCheckboxItem` must be used within `DialogComboboxOptionList`');
    });
    it('renders with checked state', async () => {
        const options = ['value 1', 'value 2', 'value 3'];
        const { getByRole, queryAllByRole } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistcheckboxitem.test.tsx_52", label: "example filter", open: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionList, { children: options.map((option) => (_jsx(DialogComboboxOptionListCheckboxItem, { value: option, checked: option === 'value 2', children: option }, option))) }) })] }));
        await waitFor(() => {
            expect(getByRole('listbox')).toBeVisible();
        });
        expect(getByRole('listbox')).toBeVisible();
        const optionsElements = queryAllByRole('option');
        expect(optionsElements).toHaveLength(3);
        expect(optionsElements[0]).toHaveAttribute('aria-selected', 'false');
        expect(optionsElements[1]).toHaveAttribute('aria-selected', 'true');
        expect(optionsElements[2]).toHaveAttribute('aria-selected', 'false');
    });
    it('propagates change on selection', async () => {
        const onChange = jest.fn();
        const options = ['value 1', 'value 2', 'value 3'];
        const { getByRole, queryAllByRole } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistcheckboxitem.test.tsx_83", label: "example filter", open: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionList, { children: options.map((option) => (_jsx(DialogComboboxOptionListCheckboxItem, { value: option, onChange: onChange, children: option }, option))) }) })] }));
        await waitFor(() => {
            expect(getByRole('listbox')).toBeVisible();
        });
        expect(getByRole('listbox')).toBeVisible();
        const optionsElements = queryAllByRole('option');
        expect(optionsElements).toHaveLength(3);
        fireEvent.click(optionsElements[0]);
        await waitFor(() => {
            expect(onChange).toHaveBeenCalledWith('value 1', expect.anything());
        });
        fireEvent.click(optionsElements[1]);
        await waitFor(() => {
            expect(onChange).toHaveBeenCalledWith('value 2', expect.anything());
        });
        fireEvent.click(optionsElements[2]);
        await waitFor(() => {
            expect(onChange).toHaveBeenCalledWith('value 3', expect.anything());
        });
    });
    it('renders indeterminate state', async () => {
        const options = ['value 1', 'value 2', 'value 3'];
        const { getByRole, queryAllByRole } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistcheckboxitem.test.tsx_125", label: "example filter", open: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionList, { children: options.map((option) => (_jsx(DialogComboboxOptionListCheckboxItem, { value: option, indeterminate: option === 'value 2', children: option }, option))) }) })] }));
        await waitFor(() => {
            expect(getByRole('listbox')).toBeVisible();
        });
        expect(getByRole('listbox')).toBeVisible();
        const optionsElements = queryAllByRole('option');
        expect(optionsElements).toHaveLength(3);
        expect(optionsElements[1].querySelector('input')).toHaveAttribute('aria-checked', 'mixed');
    });
    it('should not trigger `onChange` when disabled and help icon is clicked', async () => {
        const onChange = jest.fn();
        const { getByRole, queryAllByRole } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistcheckboxitem.test.tsx_153", label: "example filter", open: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionList, { children: _jsx(DialogComboboxOptionListCheckboxItem, { value: "value 1", onChange: onChange, disabled: true, children: "value 1" }, "value 1") }) })] }));
        await waitFor(() => {
            expect(getByRole('listbox')).toBeVisible();
        });
        expect(getByRole('listbox')).toBeVisible();
        expect(queryAllByRole('option')).toHaveLength(1);
        fireEvent.click(queryAllByRole('option')[0]);
        expect(onChange).not.toHaveBeenCalled();
    });
});
//# sourceMappingURL=DialogComboboxOptionListCheckboxItem.test.js.map