import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect } from '@jest/globals';
import { cleanup, render } from '@testing-library/react';
import { DialogCombobox } from '../DialogCombobox';
import { DialogComboboxTrigger } from '../DialogComboboxTrigger';
describe('Dialog Combobox - Trigger', () => {
    it('does not render outside DialogCombobox', () => {
        const renderWrongTrigger = () => {
            render(_jsx(DialogComboboxTrigger, {}));
        };
        const renderCorrectTrigger = () => {
            render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxtrigger.test.tsx_13", label: "example filter", children: _jsx(DialogComboboxTrigger, {}) }));
        };
        jest.spyOn(console, 'error').mockImplementation(() => { });
        expect(() => renderWrongTrigger()).toThrowError();
        expect(() => renderCorrectTrigger()).not.toThrowError();
    });
    it('should render with value visible', () => {
        const { getByRole } = render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxtrigger.test.tsx_26", label: "label", value: ['value'], children: _jsx(DialogComboboxTrigger, {}) }));
        const listbox = getByRole('combobox');
        expect(listbox).toBeVisible();
        expect(listbox.textContent).toEqual('label:value');
    });
    it('should render with multiple values', () => {
        const values = ['value1', 'value2', 'value3'];
        const { getByRole } = render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxtrigger.test.tsx_38", label: "label", value: values, children: _jsx(DialogComboboxTrigger, {}) }));
        const listbox = getByRole('combobox');
        expect(listbox).toBeVisible();
        expect(listbox.textContent).toEqual(`label:${values.join(', ')}`);
    });
    it('should render label with formatLabel', () => {
        const values = ['value1', 'value2', 'value3'];
        const format = (val) => `${val} - extra formatting`;
        const { getByRole } = render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxtrigger.test.tsx_51", label: "label", value: values, children: _jsx(DialogComboboxTrigger, { renderDisplayedValue: format }) }));
        const listbox = getByRole('combobox');
        expect(listbox).toBeVisible();
        expect(listbox.textContent).toEqual(`label:${values.map(format).join(', ')}`);
    });
    it('should render with formatDisplayedValue using React nodes', () => {
        const values = ['value1', 'value2', 'value3'];
        const nodeFormat = (val) => _jsxs("span", { "data-testid": `formatted-${val}`, children: [val, " - custom node"] });
        const { getByRole, getAllByTestId } = render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxtrigger.test.tsx_65", label: "label", value: values, children: _jsx(DialogComboboxTrigger, { renderDisplayedValue: nodeFormat }) }));
        const listbox = getByRole('combobox');
        expect(listbox).toBeVisible();
        const formattedNodes = getAllByTestId(/^formatted-/);
        expect(formattedNodes).toHaveLength(values.length);
        formattedNodes.forEach((node, index) => {
            expect(node.textContent).toEqual(`${values[index]} - custom node`);
        });
    });
    it('should render badge with more than 3 values', () => {
        const values = ['value1', 'value2', 'value3', 'value4'];
        const { getByRole } = render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxtrigger.test.tsx_83", label: "label", value: values, children: _jsx(DialogComboboxTrigger, {}) }));
        const listbox = getByRole('combobox');
        expect(listbox).toBeVisible();
        expect(listbox.textContent).toEqual(`label:${values.slice(0, 3).join(', ')}+${values.length - 3}`);
        expect(getByRole('status').textContent).toEqual(`+${values.length - 3}`);
    });
    it('should render badge after 1 value with showTagAfterValueCount', () => {
        const values = ['value1', 'value2', 'value3', 'value4'];
        const checkForOneValue = () => {
            const { getByRole } = render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxtrigger.test.tsx_97", label: "label", value: values.slice(0, 1), children: _jsx(DialogComboboxTrigger, { showTagAfterValueCount: 1 }) }));
            const listbox = getByRole('combobox');
            expect(listbox).toBeVisible();
            expect(listbox.textContent).toEqual(`label:${values[0]}`);
            expect(() => getByRole('status')).toThrowError();
        };
        const checkForBadge = () => {
            const { getByRole } = render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxtrigger.test.tsx_109", label: "label", value: values.slice(0, 2), children: _jsx(DialogComboboxTrigger, { showTagAfterValueCount: 1 }) }));
            const listbox = getByRole('combobox');
            expect(listbox).toBeVisible();
            expect(listbox.textContent).toEqual(`label:${values[0]}+1`);
            expect(getByRole('status').textContent).toEqual('+1');
        };
        checkForOneValue();
        cleanup();
        checkForBadge();
    });
    it('should render with clear button', () => {
        const { getByRole } = render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxtrigger.test.tsx_127", label: "label", value: ['value'], children: _jsx(DialogComboboxTrigger, {}) }));
        expect(getByRole('combobox')).toBeVisible();
        const button = getByRole('button', { hidden: true });
        expect(button).toBeVisible();
        expect(button).toHaveAttribute('aria-label', 'Clear selection');
    });
    it('should calculate aria-label automatically when id is not provided and label is', () => {
        const { getByRole } = render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxtrigger.test.tsx_139", label: "My label", value: ['my-value'], children: _jsx(DialogComboboxTrigger, {}) }));
        expect(getByRole('combobox')).toHaveAttribute('aria-label', 'My label, selected option: my-value');
    });
    it('should maintain string aria-label when using React nodes for formatting', () => {
        const nodeFormat = (val) => _jsxs("span", { children: [val, " - custom"] });
        const { getByRole } = render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxtrigger.test.tsx_149", label: "My label", value: ['my-value'], children: _jsx(DialogComboboxTrigger, { renderDisplayedValue: nodeFormat }) }));
        expect(getByRole('combobox')).toHaveAttribute('aria-label', 'My label, selected option: my-value');
    });
    it('should not calculate aria-label automatically when id is provided', () => {
        const { getByRole } = render(_jsx(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxtrigger.test.tsx_158", label: "My label", id: "my-id", value: ['my-value'], children: _jsx(DialogComboboxTrigger, {}) }));
        expect(getByRole('combobox')).not.toHaveAttribute('aria-label');
        expect(getByRole('combobox')).toHaveAttribute('id', 'my-id');
    });
});
//# sourceMappingURL=DialogComboboxTrigger.test.js.map