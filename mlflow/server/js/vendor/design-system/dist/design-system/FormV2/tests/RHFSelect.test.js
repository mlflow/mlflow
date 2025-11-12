import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useForm } from 'react-hook-form';
import { SelectOption } from '../../Select';
import { RHFControlledComponents } from '../RHFAdapters';
const Fixture = ({ options }) => {
    const { control, watch } = useForm({
        defaultValues: {
            data: {
                type: options[1].value,
            },
        },
    });
    const value = watch('data.type');
    return (_jsxs(_Fragment, { children: [_jsx(RHFControlledComponents.Select, { componentId: "codegen_design-system_src_design-system_formv2_tests_rhfselect.test.tsx_23", control: control, name: "data.type", label: "Type", width: "100%", children: ({ onChange }) => options.map((option) => (_jsx(SelectOption, { value: option.value, onChange: onChange, children: option.label }, option.value))) }), _jsx("div", { "data-testid": "actual-value", children: value })] }));
};
const FixtureWrappedOptions = ({ options }) => {
    const { control } = useForm({
        defaultValues: {
            data: {
                type: options[1].value,
            },
        },
    });
    return (_jsx(_Fragment, { children: _jsx(RHFControlledComponents.Select, { componentId: "codegen_design-system_src_design-system_formv2_tests_rhfselect.test.tsx_48", control: control, name: "data.type", label: "Type", width: "100%", children: ({ onChange }) => (_jsx(_Fragment, { children: options.map((option) => (_jsx(SelectOption, { value: option.value, onChange: onChange, children: option.label }, option.value))) })) }) }));
};
describe('RHF Adapter Select', () => {
    it('updates value correctly when label = value', async () => {
        const opts = [
            {
                label: 'openai',
                value: 'openai',
            },
            {
                label: 'azure',
                value: 'azure',
            },
            {
                label: 'azuread',
                value: 'azuread',
            },
        ];
        render(_jsx(Fixture, { options: opts }));
        const trigger = screen.getByRole('combobox');
        await userEvent.click(trigger);
        let options = screen.getAllByRole('option');
        expect(options).toHaveLength(3);
        const actualValue = screen.getByTestId('actual-value');
        // Trigger value matches the label of the default option (azure -- option[1])
        expect(actualValue.textContent).toBe(trigger.textContent);
        expect(actualValue.textContent).toBe(options[1].textContent);
        await userEvent.click(options[0]);
        await userEvent.click(trigger);
        options = screen.getAllByRole('option');
        // Label and value match selected value (openai -- option[0])
        expect(actualValue.textContent).toBe(options[0].textContent);
        expect(actualValue.textContent).toBe(trigger.textContent);
    });
    it('updates value correctly when label != value', async () => {
        const opts = [
            {
                label: 'Open AI',
                value: 'oai',
            },
            {
                label: 'Azure',
                value: 'azr',
            },
            {
                label: 'Azure AD',
                value: 'azad',
            },
        ];
        render(_jsx(Fixture, { options: opts }));
        const trigger = screen.getByRole('combobox');
        await userEvent.click(trigger);
        let options = screen.getAllByRole('option');
        expect(options).toHaveLength(3);
        const actualValue = screen.getByTestId('actual-value');
        expect(trigger.textContent).toBe(opts[1].label);
        expect(actualValue.textContent).toBe(opts[1].value);
        await userEvent.click(options[0]);
        await userEvent.click(trigger);
        options = screen.getAllByRole('option');
        expect(actualValue.textContent).toBe(opts[0].value);
        expect(trigger.textContent).toBe(opts[0].label);
    });
    it('works when options are mapped without fragment wrapper', async () => {
        const opts = [
            {
                label: 'Open AI',
                value: 'oai',
            },
            {
                label: 'Azure',
                value: 'azr',
            },
            {
                label: 'Azure AD',
                value: 'azad',
            },
        ];
        render(_jsx(Fixture, { options: opts }));
        const trigger = screen.getByRole('combobox');
        await userEvent.click(trigger);
        const options = screen.getAllByRole('option');
        expect(options).toHaveLength(3);
    });
    it('works when options are wrapped in fragment', async () => {
        const opts = [
            {
                label: 'Open AI',
                value: 'oai',
            },
            {
                label: 'Azure',
                value: 'azr',
            },
            {
                label: 'Azure AD',
                value: 'azad',
            },
        ];
        render(_jsx(FixtureWrappedOptions, { options: opts }));
        const trigger = screen.getByRole('combobox');
        await userEvent.click(trigger);
        const options = screen.getAllByRole('option');
        expect(options).toHaveLength(3);
    });
});
//# sourceMappingURL=RHFSelect.test.js.map