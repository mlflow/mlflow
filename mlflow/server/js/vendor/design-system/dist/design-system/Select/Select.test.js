import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, jest, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { Select } from './Select';
import { SelectContent } from './SelectContent';
import { SelectOption } from './SelectOption';
import { SelectTrigger } from './SelectTrigger';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
describe('Select', () => {
    const eventCallback = jest.fn();
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    const Component = ({ valueHasNoPii, defaultValue }) => {
        const [value, setValue] = useState(defaultValue ?? '');
        const options = ['one', 'two', 'three'];
        return (_jsx(DesignSystemEventProviderForTest, { children: _jsxs(Select, { label: "example filter", componentId: "select_test", valueHasNoPii: valueHasNoPii, value: value, children: [_jsx(SelectTrigger, {}), _jsx(SelectContent, { children: options.map((option) => (_jsx(SelectOption, { value: option, onChange: () => setValue(option), children: option }, option))) })] }) }));
    };
    it('emits value change event with value through underlying dialog combobox', async () => {
        render(_jsx(Component, { valueHasNoPii: true }));
        expect(eventCallback).not.toHaveBeenCalled();
        const select = screen.getByRole('combobox', { name: /example filter/ });
        await userEvent.click(select);
        expect(eventCallback).not.toHaveBeenCalled();
        await userEvent.click(screen.getByRole('option', { name: 'one' }));
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'select_test',
            componentType: 'dialog_combobox',
            shouldStartInteraction: false,
            value: '["one"]',
        });
        await userEvent.click(select);
        await userEvent.click(screen.getByRole('option', { name: 'two' }));
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'select_test',
            componentType: 'dialog_combobox',
            shouldStartInteraction: false,
            value: '["two"]',
        });
    });
    it('emits value change event without value when valueHasNoPii is not set', async () => {
        render(_jsx(Component, {}));
        expect(eventCallback).not.toHaveBeenCalled();
        const select = screen.getByRole('combobox', { name: /example filter/ });
        await userEvent.click(select);
        expect(eventCallback).not.toHaveBeenCalled();
        await userEvent.click(screen.getByRole('option', { name: 'one' }));
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'select_test',
            componentType: 'dialog_combobox',
            shouldStartInteraction: false,
            value: undefined,
        });
    });
    it('does not emit a value change event for the default value', async () => {
        render(_jsx(Component, { defaultValue: "one" }));
        expect(eventCallback).not.toHaveBeenCalled();
    });
});
//# sourceMappingURL=Select.test.js.map