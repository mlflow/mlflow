import { Fragment as _Fragment, jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, afterEach, jest, it, expect } from '@jest/globals';
import { fireEvent, render, waitFor, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { setupDesignSystemEventProviderForTesting } from '../../DesignSystemEventProvider';
import { DialogCombobox } from '../DialogCombobox';
import { DialogComboboxContent } from '../DialogComboboxContent';
import { DialogComboboxOptionList } from '../DialogComboboxOptionList';
import { DialogComboboxOptionListCheckboxItem } from '../DialogComboboxOptionListCheckboxItem';
import { DialogComboboxOptionListSelectItem } from '../DialogComboboxOptionListSelectItem';
import { DialogComboboxTrigger } from '../DialogComboboxTrigger';
describe('Dialog Combobox - OptionList', () => {
    afterEach(() => {
        jest.clearAllMocks();
    });
    it('does not render outside DialogCombobox', () => {
        const renderOutsideDialogCombobox = () => {
            render(_jsx(DialogComboboxOptionList, { children: _jsx(_Fragment, {}) }));
        };
        const renderInsideDialogCombobox = () => {
            render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlist.test.tsx_28", label: "example filter", children: _jsx(DialogComboboxOptionList, { children: _jsx(_Fragment, {}) }) }));
        };
        jest.spyOn(console, 'error').mockImplementation(() => { });
        expect(() => renderOutsideDialogCombobox()).toThrowError();
        expect(() => renderInsideDialogCombobox()).not.toThrowError();
    });
    it('opens on trigger click', async () => {
        const { getByRole } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlist.test.tsx_43", label: "example filter", children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionList, { children: _jsx(_Fragment, {}) }) })] }));
        fireEvent.click(getByRole('combobox'));
        await waitFor(() => {
            expect(getByRole('listbox')).toBeVisible();
        });
    });
    it('closes on escape', async () => {
        const onOpenChange = jest.fn();
        const { getByRole } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlist.test.tsx_63", label: "example filter", open: true, onOpenChange: onOpenChange, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionList, { children: _jsx(_Fragment, {}) }) })] }));
        await waitFor(() => {
            expect(getByRole('listbox')).toBeVisible();
        });
        onOpenChange.mockReset();
        fireEvent.keyDown(getByRole('listbox'), { key: 'Escape' });
        expect(onOpenChange).toHaveBeenCalledWith(false);
    });
    it('shows loading state', async () => {
        const options = ['value 1', 'value 2', 'value 3'];
        const { getByRole, getByLabelText, queryByRole } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlist.test.tsx_84", label: "example filter", open: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionList, { loading: true, children: options.map((option) => (_jsx(DialogComboboxOptionListSelectItem, { value: option, children: option }, option))) }) })] }));
        await waitFor(() => {
            expect(getByRole('combobox')).toBeVisible();
        });
        expect(getByRole('listbox')).toBeVisible();
        expect(getByLabelText('Loading')).toBeVisible();
        expect(queryByRole('option')).not.toBeInTheDocument();
    });
    it('shows loading state with progressive loading', async () => {
        const options = ['value 1', 'value 2', 'value 3'];
        const { getByRole, getByLabelText, queryAllByRole } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlist.test.tsx_110", label: "example filter", open: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionList, { loading: true, withProgressiveLoading: true, children: options.map((option) => (_jsx(DialogComboboxOptionListSelectItem, { value: option, children: option }, option))) }) })] }));
        await waitFor(() => {
            expect(getByRole('combobox')).toBeVisible();
        });
        expect(getByRole('listbox')).toBeVisible();
        expect(getByLabelText('Loading')).toBeVisible();
        expect(queryAllByRole('option')).toHaveLength(3);
    });
    describe('Analytics Events', () => {
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        const SingleSelectComponent = ({ valueHasNoPii, defaultValue, }) => {
            const [value, setValue] = useState(defaultValue ?? '');
            const options = ['one', 'two', 'three'];
            return (_jsx(DesignSystemEventProviderForTest, { children: _jsxs(DialogCombobox, { label: "example filter", componentId: "dialog_combobox_test", valueHasNoPii: valueHasNoPii, value: value ? [value] : [], children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionList, { children: options.map((option) => (_jsx(DialogComboboxOptionListSelectItem, { value: option, onChange: () => setValue(option), checked: option === value }, option))) }) })] }) }));
        };
        const MultiselectComponent = ({ valueHasNoPii, defaultValue, }) => {
            const [values, setValues] = useState(defaultValue ?? []);
            const options = ['one', 'two', 'three'];
            const handleChange = (option) => {
                if (values.includes(option)) {
                    setValues(values.filter((value) => value !== option));
                }
                else {
                    setValues([...values, option]);
                }
            };
            return (_jsx(DesignSystemEventProviderForTest, { children: _jsxs(DialogCombobox, { label: "example filter", componentId: "dialog_combobox_test", valueHasNoPii: valueHasNoPii, value: values, multiSelect: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionList, { children: options.map((option) => (_jsx(DialogComboboxOptionListCheckboxItem, { value: option, onChange: handleChange, checked: values.includes(option) }, option))) }) })] }) }));
        };
        describe('Single Select List', () => {
            it('emits value change events with value', async () => {
                render(_jsx(SingleSelectComponent, { valueHasNoPii: true }));
                expect(eventCallback).not.toHaveBeenCalled();
                const combobox = screen.getByRole('combobox', { name: /example filter/ });
                await userEvent.click(combobox);
                expect(eventCallback).not.toHaveBeenCalled();
                await userEvent.click(screen.getByRole('option', { name: 'one' }));
                expect(eventCallback).toHaveBeenCalledTimes(1);
                expect(eventCallback).toHaveBeenCalledWith({
                    eventType: 'onValueChange',
                    componentId: 'dialog_combobox_test',
                    componentType: 'dialog_combobox',
                    shouldStartInteraction: false,
                    value: '["one"]',
                });
                await userEvent.click(combobox);
                await userEvent.click(screen.getByRole('option', { name: 'two' }));
                expect(eventCallback).toHaveBeenCalledTimes(2);
                expect(eventCallback).toHaveBeenCalledWith({
                    eventType: 'onValueChange',
                    componentId: 'dialog_combobox_test',
                    componentType: 'dialog_combobox',
                    shouldStartInteraction: false,
                    value: '["two"]',
                });
            });
            it('emits value changes events without value when valueHasNoPii is not set', async () => {
                render(_jsx(SingleSelectComponent, {}));
                expect(eventCallback).not.toHaveBeenCalled();
                const combobox = screen.getByRole('combobox', { name: /example filter/ });
                await userEvent.click(combobox);
                expect(eventCallback).not.toHaveBeenCalled();
                await userEvent.click(screen.getByRole('option', { name: 'one' }));
                expect(eventCallback).toHaveBeenCalledTimes(1);
                expect(eventCallback).toHaveBeenCalledWith({
                    eventType: 'onValueChange',
                    componentId: 'dialog_combobox_test',
                    componentType: 'dialog_combobox',
                    shouldStartInteraction: false,
                    value: undefined,
                });
            });
            it('does not emit a value change event for the default value', async () => {
                render(_jsx(SingleSelectComponent, { defaultValue: "one" }));
                expect(eventCallback).not.toHaveBeenCalled();
            });
        });
        describe('Multiselect List', () => {
            it('emits value change events with value', async () => {
                render(_jsx(MultiselectComponent, { valueHasNoPii: true }));
                expect(eventCallback).not.toHaveBeenCalled();
                const combobox = screen.getByRole('combobox', { name: /example filter/ });
                await userEvent.click(combobox);
                expect(eventCallback).not.toHaveBeenCalled();
                await userEvent.click(screen.getByRole('option', { name: 'one' }));
                expect(eventCallback).toHaveBeenCalledTimes(1);
                expect(eventCallback).toHaveBeenCalledWith({
                    eventType: 'onValueChange',
                    componentId: 'dialog_combobox_test',
                    componentType: 'dialog_combobox',
                    shouldStartInteraction: false,
                    value: '["one"]',
                });
                await userEvent.click(screen.getByRole('option', { name: 'two' }));
                expect(eventCallback).toHaveBeenCalledTimes(2);
                expect(eventCallback).toHaveBeenCalledWith({
                    eventType: 'onValueChange',
                    componentId: 'dialog_combobox_test',
                    componentType: 'dialog_combobox',
                    shouldStartInteraction: false,
                    value: '["one","two"]',
                });
            });
            it('emits value changes events without value when valueHasNoPii is not set', async () => {
                render(_jsx(MultiselectComponent, {}));
                expect(eventCallback).not.toHaveBeenCalled();
                const combobox = screen.getByRole('combobox', { name: /example filter/ });
                await userEvent.click(combobox);
                expect(eventCallback).not.toHaveBeenCalled();
                await userEvent.click(screen.getByRole('option', { name: 'one' }));
                expect(eventCallback).toHaveBeenCalledTimes(1);
                expect(eventCallback).toHaveBeenCalledWith({
                    eventType: 'onValueChange',
                    componentId: 'dialog_combobox_test',
                    componentType: 'dialog_combobox',
                    shouldStartInteraction: false,
                    value: undefined,
                });
            });
            it('does not emit a value change event for the default value', async () => {
                render(_jsx(MultiselectComponent, { defaultValue: ['one', 'two'] }));
                expect(eventCallback).not.toHaveBeenCalled();
            });
        });
    });
});
//# sourceMappingURL=DialogComboboxOptionList.test.js.map