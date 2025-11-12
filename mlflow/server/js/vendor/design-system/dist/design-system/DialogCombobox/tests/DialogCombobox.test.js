import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, expect, jest } from '@jest/globals';
import { fireEvent, render, waitFor } from '@testing-library/react';
import { DialogCombobox } from '../DialogCombobox';
import { DialogComboboxContent } from '../DialogComboboxContent';
import { DialogComboboxOptionList } from '../DialogComboboxOptionList';
import { DialogComboboxOptionListCheckboxItem } from '../DialogComboboxOptionListCheckboxItem';
import { DialogComboboxOptionListSelectItem } from '../DialogComboboxOptionListSelectItem';
import { DialogComboboxTrigger } from '../DialogComboboxTrigger';
describe('Dialog Combobox', () => {
    it('should render', () => {
        const { getByRole } = render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcombobox.test.tsx_13", label: "render test", children: _jsx(DialogComboboxTrigger, {}) }));
        expect(getByRole('combobox')).toBeVisible();
    });
    it('should render once on value change from within', async () => {
        const onChange = jest.fn();
        const { getByRole, getByText } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcombobox.test.tsx_24", label: "render test", value: [], open: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsxs(DialogComboboxOptionList, { children: [_jsx(DialogComboboxOptionListSelectItem, { value: "one", onChange: onChange, children: "one" }), _jsx(DialogComboboxOptionListSelectItem, { value: "two", onChange: onChange, children: "two" })] }) })] }));
        expect(getByRole('combobox')).toBeVisible();
        fireEvent.click(getByText('one'));
        await waitFor(() => {
            expect(onChange).toHaveBeenCalledWith('one', expect.anything());
            expect(onChange).toHaveBeenCalledTimes(1);
        });
    });
    it('should render once on value change from within multiselect', async () => {
        const onChange = jest.fn();
        const { queryAllByRole, getByRole } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcombobox.test.tsx_50", label: "render test", multiSelect: true, value: [], open: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsxs(DialogComboboxOptionList, { children: [_jsx(DialogComboboxOptionListCheckboxItem, { value: "one", onChange: onChange, children: "one" }), _jsx(DialogComboboxOptionListCheckboxItem, { value: "two", onChange: onChange, children: "two" })] }) })] }));
        await waitFor(() => expect(getByRole('listbox')).toBeVisible());
        const optionsElements = queryAllByRole('option');
        expect(optionsElements).toHaveLength(2);
        fireEvent.click(optionsElements[0]);
        await waitFor(() => {
            expect(onChange).toHaveBeenCalledWith('one', expect.anything());
            expect(onChange).toHaveBeenCalledTimes(1);
        });
        fireEvent.click(optionsElements[1]);
        await waitFor(() => {
            expect(onChange).toHaveBeenCalledWith('two', expect.anything());
            expect(onChange).toHaveBeenCalledTimes(2);
        });
    });
});
//# sourceMappingURL=DialogCombobox.test.js.map