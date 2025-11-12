import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useRef, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { RHFControlledComponents } from '../RHFAdapters';
import { FormUI } from '../index';
const Fixture = ({ initialValue = '', required = false, autoFocus = false, }) => {
    const { control, watch, formState: { errors }, } = useForm({
        mode: 'all',
        defaultValues: {
            data: {
                value: initialValue,
            },
        },
    });
    const value = watch('data.value');
    const inputRef = useRef(null);
    useEffect(() => {
        if (autoFocus) {
            inputRef.current?.focus();
        }
    }, [autoFocus]);
    return (_jsxs(_Fragment, { children: [_jsx(FormUI.Label, { htmlFor: "test-input", children: "Value" }), _jsx(RHFControlledComponents.Input, { id: "test-input", componentId: "test-rhf-input", control: control, name: "data.value", rules: { required }, inputRef: inputRef }), errors?.data?.value?.type === 'required' && _jsx(FormUI.Message, { type: "error", message: "Missing required value" }), _jsx("div", { "data-testid": "actual-value", children: value })] }));
};
describe('RHF Adapter Input', () => {
    it('initializes from initial value', async () => {
        render(_jsx(Fixture, { initialValue: "initial value" }));
        expect(await screen.findByLabelText('Value')).toHaveValue('initial value');
    });
    it('updates value correctly', async () => {
        render(_jsx(Fixture, {}));
        await userEvent.type(await screen.findByLabelText('Value'), 'new value');
        expect(await screen.findByLabelText('Value')).toHaveValue('new value');
    });
    it('applies validation correctly', async () => {
        render(_jsx(Fixture, { required: true, initialValue: "value" }));
        await userEvent.clear(await screen.findByLabelText('Value'));
        expect(await screen.findByText('Missing required value')).toBeInTheDocument();
    });
    it('passes inputRef correctly', async () => {
        render(_jsx(Fixture, { autoFocus: true }));
        expect(await screen.findByLabelText('Value')).toHaveFocus();
    });
});
//# sourceMappingURL=RHFInput.test.js.map