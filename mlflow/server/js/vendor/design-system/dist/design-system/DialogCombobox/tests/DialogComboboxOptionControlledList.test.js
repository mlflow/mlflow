import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, jest, afterEach, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { setupDesignSystemEventProviderForTesting } from '../../DesignSystemEventProvider';
import { DialogCombobox } from '../DialogCombobox';
import { DialogComboboxContent } from '../DialogComboboxContent';
import { DialogComboboxOptionControlledList } from '../DialogComboboxOptionControlledList';
import { DialogComboboxTrigger } from '../DialogComboboxTrigger';
describe('Dialog Combobox - OptionControlledList', () => {
    const eventCallback = jest.fn();
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    const Component = ({ valueHasNoPii, multiSelect, value, }) => (_jsx(DesignSystemEventProviderForTest, { children: _jsxs(DialogCombobox, { label: "example filter", componentId: "dialog_combobox_test", valueHasNoPii: valueHasNoPii, multiSelect: multiSelect, value: value, children: [_jsx(DialogComboboxTrigger, { controlled: true }), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionControlledList, { options: ['one', 'two', 'three'] }) })] }) }));
    afterEach(() => {
        jest.clearAllMocks();
    });
    it('does not render outside DialogCombobox', () => {
        const renderOutsideDialogCombobox = () => {
            render(_jsx(DialogComboboxOptionControlledList, { options: [] }));
        };
        const renderInsideDialogCombobox = () => {
            render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptioncontrolledlist.test.tsx_48", label: "example filter", children: _jsx(DialogComboboxOptionControlledList, { options: [] }) }));
        };
        jest.spyOn(console, 'error').mockImplementation(() => { });
        expect(() => renderOutsideDialogCombobox()).toThrowError();
        expect(() => renderInsideDialogCombobox()).not.toThrowError();
    });
    it('emits value change events for single select list', async () => {
        render(_jsx(Component, { valueHasNoPii: true }));
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
    it('emits value change events for multiselect list', async () => {
        render(_jsx(Component, { valueHasNoPii: true, multiSelect: true }));
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
    it('emits value change events without value when valueHasNoPii is not set', async () => {
        render(_jsx(Component, {}));
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
        render(_jsx(Component, { valueHasNoPii: true, value: ['one'] }));
        expect(eventCallback).not.toHaveBeenCalled();
    });
});
//# sourceMappingURL=DialogComboboxOptionControlledList.test.js.map