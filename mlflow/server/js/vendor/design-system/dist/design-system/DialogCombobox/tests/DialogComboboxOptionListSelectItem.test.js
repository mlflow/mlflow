import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, afterEach, jest, it, expect } from '@jest/globals';
import { fireEvent, render, waitFor } from '@testing-library/react';
import { DialogCombobox } from '../DialogCombobox';
import { DialogComboboxContent } from '../DialogComboboxContent';
import { DialogComboboxOptionList } from '../DialogComboboxOptionList';
import { DialogComboboxOptionListSelectItem } from '../DialogComboboxOptionListSelectItem';
import { DialogComboboxTrigger } from '../DialogComboboxTrigger';
describe('Dialog Combobox - SelectItem', () => {
    afterEach(() => {
        jest.clearAllMocks();
    });
    function renderDialogCombobox(options, listItemProps = {}) {
        const { getByRole, queryAllByRole } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistselectitem.test.tsx_18", label: "example filter", open: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionList, { children: options.map((option) => (_jsx(DialogComboboxOptionListSelectItem, { value: option, ...listItemProps, children: option }, option))) }) })] }));
        return { getByRole, queryAllByRole };
    }
    it('renders', async () => {
        const options = ['value 1', 'value 2', 'value 3'];
        const { getByRole, queryAllByRole } = renderDialogCombobox(options);
        await waitFor(() => {
            expect(getByRole('combobox')).toBeVisible();
        });
        expect(getByRole('combobox')).toBeVisible();
        expect(queryAllByRole('option')).toHaveLength(3);
    });
    it('does not render outside of DialogComboboxOptionList', async () => {
        jest.spyOn(console, 'error').mockImplementation(() => { });
        expect(() => render(_jsx(DialogComboboxOptionListSelectItem, { value: "value 1", children: "value 1" }))).toThrowError('`DialogComboboxOptionListSelectItem` must be used within `DialogComboboxOptionList`');
    });
    it('renders with selected state', async () => {
        const options = ['value 1', 'value 2', 'value 3'];
        const { getByRole, queryAllByRole } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistselectitem.test.tsx_52", label: "example filter", open: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionList, { children: options.map((option) => (_jsx(DialogComboboxOptionListSelectItem, { value: option, checked: option === 'value 2', children: option }, option))) }) })] }));
        await waitFor(() => {
            expect(getByRole('combobox')).toBeVisible();
        });
        expect(getByRole('combobox')).toBeVisible();
        expect(queryAllByRole('option')).toHaveLength(3);
        expect(queryAllByRole('option')[1]).toHaveAttribute('aria-selected', 'true');
    });
    it('propagates changes', async () => {
        const options = ['value 1', 'value 2', 'value 3'];
        const onChange = jest.fn();
        const { getByRole, queryAllByRole } = renderDialogCombobox(options, { onChange });
        await waitFor(() => {
            expect(getByRole('combobox')).toBeVisible();
        });
        expect(getByRole('combobox')).toBeVisible();
        expect(queryAllByRole('option')).toHaveLength(3);
        fireEvent.click(queryAllByRole('option')[1]);
        await waitFor(() => {
            expect(onChange).toHaveBeenCalledWith('value 2', expect.anything());
        });
    });
    it('navigates with keyboard', async () => {
        const options = ['value 1', 'value 2', 'value 3'];
        const { getByRole, queryAllByRole } = renderDialogCombobox(options);
        await waitFor(() => {
            expect(getByRole('combobox')).toBeVisible();
        });
        expect(getByRole('combobox')).toBeVisible();
        expect(queryAllByRole('option')).toHaveLength(3);
        fireEvent.keyDown(getByRole('combobox'), { key: 'ArrowDown' });
        const optionElements = queryAllByRole('option');
        await waitFor(() => {
            expect(optionElements[0]).toHaveFocus();
        });
        fireEvent.keyDown(optionElements[0], { key: 'ArrowDown' });
        await waitFor(() => {
            expect(optionElements[1]).toHaveFocus();
        });
        fireEvent.keyDown(optionElements[1], { key: 'ArrowUp' });
        await waitFor(() => {
            expect(optionElements[0]).toHaveFocus();
        });
    });
    it('should disable mouseover during keyboard navigation', async () => {
        const options = ['value 1', 'value 2', 'value 3'];
        const { getByRole, queryAllByRole } = renderDialogCombobox(options);
        await waitFor(() => {
            expect(getByRole('combobox')).toBeVisible();
        });
        fireEvent.keyDown(getByRole('combobox'), { key: 'ArrowDown' });
        const optionElements = queryAllByRole('option');
        await waitFor(() => {
            expect(optionElements[0]).toHaveFocus();
        });
        fireEvent.keyDown(optionElements[0], { key: 'ArrowDown' });
        expect(optionElements[1]).toHaveFocus();
        fireEvent.mouseOver(optionElements[2]);
        // didn't change focus because mouseOver is disabled by the keyDown
        expect(optionElements[1]).toHaveFocus();
        // mouseMove reenabled mouseOver
        fireEvent.mouseMove(optionElements[2]);
        fireEvent.mouseOver(optionElements[2]);
        await waitFor(() => {
            expect(optionElements[2]).toHaveFocus();
        });
    });
    it('should not trigger `onChange` when disabled and help icon is clicked', async () => {
        const options = ['value 1'];
        const onChange = jest.fn();
        const { getByRole, queryAllByRole } = renderDialogCombobox(options, { onChange, disabled: true });
        await waitFor(() => {
            expect(getByRole('combobox')).toBeVisible();
        });
        expect(getByRole('combobox')).toBeVisible();
        expect(queryAllByRole('option')).toHaveLength(1);
        fireEvent.click(queryAllByRole('option')[0]);
        expect(onChange).not.toHaveBeenCalled();
    });
});
//# sourceMappingURL=DialogComboboxOptionListSelectItem.test.js.map