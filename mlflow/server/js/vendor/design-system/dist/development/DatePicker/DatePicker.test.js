import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, it, expect, jest } from '@jest/globals';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Controller, useForm } from 'react-hook-form';
import { DatePicker, RangePicker } from './DatePicker';
import { DesignSystemProvider } from '../../design-system';
const DatePickerExample = (props) => {
    return (_jsx(DesignSystemProvider, { children: _jsx(DatePicker, { ...props }) }));
};
const RangePickerExample = (props) => {
    return (_jsx(DesignSystemProvider, { children: _jsx(RangePicker, { ...props }) }));
};
const DatePickerRHFExample = (props) => {
    const { onUpdate, ...restProps } = props;
    const { register, setValue } = useForm();
    return (_jsx(DesignSystemProvider, { children: _jsx(DatePicker, { ...register('date', {
                onChange: (e) => {
                    onUpdate(e);
                    setValue('date', e);
                },
            }), ...restProps }) }));
};
const DatePickerRHFControlExample = (props) => {
    const { onUpdate } = props;
    const { control } = useForm();
    return (_jsx(DesignSystemProvider, { children: _jsx(Controller, { name: "date", control: control, render: ({ field }) => (_jsx(DatePicker, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_62", onChange: (e) => {
                    onUpdate(e.target.value);
                    field.onChange(e.target.value);
                }, value: field.value })) }) }));
};
const RangePickerRHFExample = (props) => {
    const { onUpdate } = props;
    const { register, setValue } = useForm();
    return (_jsx(DesignSystemProvider, { children: _jsx(RangePicker, { startDatePickerProps: {
                componentId: 'YOUR_TRACKING_ID',
                ...register('from', {
                    onChange: (e) => {
                        onUpdate(e);
                        setValue('from', e);
                    },
                }),
            }, endDatePickerProps: {
                componentId: 'YOUR_TRACKING_ID',
                ...register('to', {
                    onChange: (e) => {
                        onUpdate(e);
                        setValue('to', e);
                    },
                }),
            } }) }));
};
const RangePickerRHFControlExample = (props) => {
    const { onUpdate } = props;
    const { control } = useForm();
    return (_jsx(DesignSystemProvider, { children: _jsx(Controller, { name: "range", control: control, render: ({ field }) => (_jsx(RangePicker, { startDatePickerProps: {
                    componentId: 'YOUR_TRACKING_ID',
                    onChange: (e) => {
                        onUpdate(e.target.value);
                        field.onChange({ ...field.value, from: e.target.value });
                    },
                    value: field.value?.from,
                }, endDatePickerProps: {
                    componentId: 'YOUR_TRACKING_ID',
                    onChange: (e) => {
                        onUpdate(e.target.value);
                        field.onChange({ ...field.value, to: e.target.value });
                    },
                    value: field.value?.to,
                } })) }) }));
};
const getDatePickerInput = (name = 'Select Date') => screen.getByRole('textbox', { name });
const getDatePickerDialog = () => screen.getByRole('dialog');
const getDatePickerDateCell = (date) => screen.getByRole('button', { name: date });
const getClearButton = () => screen.getByRole('button', { name: 'close-circle' });
const getTimeInput = () => screen.getByRole('textbox', { name: 'Time' });
describe('DatePicker', () => {
    it('should render DatePicker', () => {
        render(_jsx(DatePickerExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_155" }));
        expect(getDatePickerInput()).toBeInTheDocument();
    });
    it('should update date on input value change', async () => {
        render(_jsx(DatePickerExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_161" }));
        const input = getDatePickerInput();
        fireEvent.focus(input);
        fireEvent.change(input, { target: { value: '2024-05-05' } });
        expect(input).toHaveValue('2024-05-05');
    });
    it('should update date and time on input value change', async () => {
        render(_jsx(DatePickerExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_176", includeTime: true }));
        const input = getDatePickerInput('Select Date and Time');
        fireEvent.focus(input);
        fireEvent.change(input, { target: { value: '2024-05-05 22:00' } });
        expect(input).toHaveValue('2024-05-05T22:00');
    });
    it('should display seconds when using includeSeconds', async () => {
        render(_jsx(DatePickerExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_176", includeTime: true, includeSeconds: true }));
        const input = getDatePickerInput('Select Date and Time');
        fireEvent.focus(input);
        fireEvent.change(input, { target: { value: '2024-05-05 22:00:59' } });
        // when accessing input value, it includes milliseconds by default
        expect(input).toHaveValue('2024-05-05T22:00:59.000');
    });
    it('should ignore includeSeconds if includeTime is not present', async () => {
        render(_jsx(DatePickerExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_176", includeSeconds: true }));
        const input = getDatePickerInput('Select Date');
        fireEvent.focus(input);
        fireEvent.change(input, { target: { value: '2024-05-05' } });
        expect(input).toHaveValue('2024-05-05');
    });
    it('should open datepicker dialog on input click', async () => {
        render(_jsx(DatePickerExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_190" }));
        const input = getDatePickerInput();
        await userEvent.click(input);
        const dialog = getDatePickerDialog();
        expect(dialog).toBeInTheDocument();
    });
    it('should select date on date click', async () => {
        render(_jsx(DatePickerExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_201" }));
        const input = getDatePickerInput();
        await userEvent.click(input);
        const dialog = getDatePickerDialog();
        expect(dialog).toBeInTheDocument();
        const dateCell = getDatePickerDateCell('Thursday, January 15, 1970');
        await userEvent.click(dateCell);
        expect(input).toHaveValue('1970-01-15');
    });
    it('should select date on date enter key press', async () => {
        render(_jsx(DatePickerExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_219" }));
        const input = getDatePickerInput();
        await userEvent.click(input);
        const dialog = getDatePickerDialog();
        await waitFor(async () => {
            expect(dialog).toBeInTheDocument();
        });
        const dateCell = getDatePickerDateCell('Thursday, January 15, 1970');
        await userEvent.type(dateCell, '{enter}');
        await waitFor(async () => {
            expect(dialog).not.toBeInTheDocument();
        });
        expect(input).toHaveValue('1970-01-15');
    });
    it('should clear date on clear button click', async () => {
        render(_jsx(DatePickerExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_237", allowClear: true }));
        const input = getDatePickerInput();
        fireEvent.focus(input);
        fireEvent.change(input, { target: { value: '2024-05-05' } });
        expect(input).toHaveValue('2024-05-05');
        const clearButton = getClearButton();
        await userEvent.click(clearButton);
        expect(input).toHaveValue('');
    });
    it('should select date and update time on date click and time change', async () => {
        render(_jsx(DatePickerExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_258", includeTime: true }));
        const input = getDatePickerInput('Select Date and Time');
        await userEvent.click(input);
        const dialog = getDatePickerDialog();
        expect(dialog).toBeInTheDocument();
        const dateCell = getDatePickerDateCell('Thursday, January 15, 1970');
        await userEvent.click(dateCell);
        expect(input).toHaveValue('1970-01-15T00:00');
        const timeInput = getTimeInput();
        fireEvent.focus(timeInput);
        fireEvent.change(timeInput, { target: { value: '22:00' } });
        expect(input).toHaveValue('1970-01-15T22:00');
    });
    it('should select date and update time on date click and time change (including seconds)', async () => {
        render(_jsx(DatePickerExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_258", includeTime: true, includeSeconds: true }));
        const input = getDatePickerInput('Select Date and Time');
        await userEvent.click(input);
        const dialog = getDatePickerDialog();
        expect(dialog).toBeInTheDocument();
        const dateCell = getDatePickerDateCell('Thursday, January 15, 1970');
        await userEvent.click(dateCell);
        expect(input).toHaveValue('1970-01-15T00:00');
        const timeInput = getTimeInput();
        fireEvent.focus(timeInput);
        fireEvent.change(timeInput, { target: { value: '22:45:30' } });
        // when accessing input value, it includes milliseconds by default
        expect(input).toHaveValue('1970-01-15T22:45:30.000');
        const dateCell2 = getDatePickerDateCell('Saturday, January 10, 1970');
        await userEvent.click(dateCell2);
        // when accessing input value, it includes milliseconds by default
        expect(input).toHaveValue('1970-01-10T22:45:30.000');
    });
});
const getRangePickerInputs = (name = 'Select Date') => screen.getAllByRole('textbox', { name });
const getRangePickerClearButtons = () => screen.getAllByRole('button', { name: 'close-circle' });
describe('RangePicker', () => {
    it('should render RangePicker', () => {
        render(_jsx(RangePickerExample, {}));
        const rangePickerInputs = getRangePickerInputs();
        expect(rangePickerInputs).toHaveLength(2);
        expect(rangePickerInputs[0]).toBeInTheDocument();
        expect(rangePickerInputs[1]).toBeInTheDocument();
    });
    it('should update dates on input value change', async () => {
        render(_jsx(RangePickerExample, {}));
        const rangePickerInputs = getRangePickerInputs();
        const fromInput = rangePickerInputs[0];
        fireEvent.focus(fromInput);
        fireEvent.change(fromInput, { target: { value: '2024-05-05' } });
        expect(fromInput).toHaveValue('2024-05-05');
        const toInput = rangePickerInputs[1];
        fireEvent.focus(toInput);
        fireEvent.change(toInput, { target: { value: '2025-05-05' } });
        expect(toInput).toHaveValue('2025-05-05');
    });
    it('should update dates and times on input value change', async () => {
        render(_jsx(RangePickerExample, { includeTime: true }));
        const rangePickerInputs = getRangePickerInputs('Select Date and Time');
        const fromInput = rangePickerInputs[0];
        fireEvent.focus(fromInput);
        fireEvent.change(fromInput, { target: { value: '2024-05-05 08:00' } });
        expect(fromInput).toHaveValue('2024-05-05T08:00');
        const toInput = rangePickerInputs[1];
        fireEvent.focus(toInput);
        fireEvent.change(toInput, { target: { value: '2025-05-05 22:00' } });
        expect(toInput).toHaveValue('2025-05-05T22:00');
    });
    it('should update dates and times with seconds on input value change', async () => {
        render(_jsx(RangePickerExample, { includeTime: true, includeSeconds: true }));
        const rangePickerInputs = getRangePickerInputs('Select Date and Time');
        const fromInput = rangePickerInputs[0];
        fireEvent.focus(fromInput);
        fireEvent.change(fromInput, { target: { value: '2024-05-05 08:00:30' } });
        expect(fromInput).toHaveValue('2024-05-05T08:00:30.000');
        const toInput = rangePickerInputs[1];
        fireEvent.focus(toInput);
        fireEvent.change(toInput, { target: { value: '2025-05-05 22:00:59' } });
        expect(toInput).toHaveValue('2025-05-05T22:00:59.000');
    });
    it('should ignore includeSeconds if includeTime is not present', async () => {
        render(_jsx(RangePickerExample, { includeSeconds: true }));
        const rangePickerInputs = getRangePickerInputs('Select Date');
        const fromInput = rangePickerInputs[0];
        fireEvent.focus(fromInput);
        fireEvent.change(fromInput, { target: { value: '2024-05-05' } });
        expect(fromInput).toHaveValue('2024-05-05');
        const toInput = rangePickerInputs[1];
        fireEvent.focus(toInput);
        fireEvent.change(toInput, { target: { value: '2025-05-10' } });
        expect(toInput).toHaveValue('2025-05-10');
    });
    it('should open datepicker dialog on input click', async () => {
        render(_jsx(RangePickerExample, {}));
        const rangePickerInputs = getRangePickerInputs();
        const fromInput = rangePickerInputs[0];
        await userEvent.click(fromInput);
        expect(getDatePickerDialog()).toBeInTheDocument();
        const toInput = rangePickerInputs[1];
        await userEvent.click(toInput);
        expect(getDatePickerDialog()).toBeInTheDocument();
    });
    it('should select date on date click', async () => {
        render(_jsx(RangePickerExample, {}));
        const rangePickerInputs = getRangePickerInputs();
        const fromInput = rangePickerInputs[0];
        await userEvent.click(fromInput);
        expect(getDatePickerDialog()).toBeInTheDocument();
        const fromDateCell = getDatePickerDateCell('Thursday, January 15, 1970');
        await userEvent.click(fromDateCell);
        expect(fromInput).toHaveValue('1970-01-15');
        const toInput = rangePickerInputs[1];
        expect(getDatePickerDialog()).toBeInTheDocument();
        const toDateCell = getDatePickerDateCell('Tuesday, January 20, 1970');
        await userEvent.click(toDateCell);
        expect(toInput).toHaveValue('1970-01-20');
    });
    it('should have starting date selected in the second datepicker dialog', async () => {
        render(_jsx(RangePickerExample, {}));
        const rangePickerInputs = getRangePickerInputs();
        const fromInput = rangePickerInputs[0];
        await userEvent.click(fromInput);
        expect(getDatePickerDialog()).toBeInTheDocument();
        const fromDateCell = getDatePickerDateCell('Thursday, January 15, 1970');
        await userEvent.click(fromDateCell);
        expect(fromInput).toHaveValue('1970-01-15');
        expect(getDatePickerDialog()).toBeInTheDocument();
        expect(getDatePickerDateCell('Thursday, January 15, 1970')).toHaveAttribute('aria-selected', 'true');
    });
    it('should have ending date select in the first datepicker dialog', async () => {
        render(_jsx(RangePickerExample, {}));
        const rangePickerInputs = getRangePickerInputs();
        const toInput = rangePickerInputs[1];
        await userEvent.click(toInput);
        expect(getDatePickerDialog()).toBeInTheDocument();
        const toDateCell = getDatePickerDateCell('Thursday, January 15, 1970');
        await userEvent.click(toDateCell);
        expect(toInput).toHaveValue('1970-01-15');
        const fromInput = rangePickerInputs[0];
        await userEvent.click(fromInput);
        expect(getDatePickerDialog()).toBeInTheDocument();
        expect(getDatePickerDateCell('Thursday, January 15, 1970')).toHaveAttribute('aria-selected', 'true');
    });
    it('should clear starting date on clear button click', async () => {
        render(_jsx(RangePickerExample, { allowClear: true }));
        const rangePickerInputs = getRangePickerInputs();
        const fromInput = rangePickerInputs[0];
        fireEvent.focus(fromInput);
        fireEvent.change(fromInput, { target: { value: '2024-05-05' } });
        expect(fromInput).toHaveValue('2024-05-05');
        const clearButton = getRangePickerClearButtons()[0];
        await userEvent.click(clearButton);
        expect(fromInput).toHaveValue('');
    });
    it('should clear ending date on clear button click', async () => {
        render(_jsx(RangePickerExample, { allowClear: true }));
        const rangePickerInputs = getRangePickerInputs();
        const toInput = rangePickerInputs[1];
        fireEvent.focus(toInput);
        fireEvent.change(toInput, { target: { value: '2024-05-05' } });
        expect(toInput).toHaveValue('2024-05-05');
        const clearButton = getRangePickerClearButtons()[1];
        await userEvent.click(clearButton);
        expect(toInput).toHaveValue('');
    });
});
describe('DatePicker with React Hook Form', () => {
    it('should update date on input value change', async () => {
        const onUpdate = jest.fn();
        render(_jsx(DatePickerRHFExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_503", onUpdate: onUpdate }));
        expect(onUpdate).toHaveBeenCalledTimes(0);
        const input = getDatePickerInput();
        fireEvent.focus(input);
        fireEvent.change(input, { target: { value: '2024-05-05' } });
        expect(onUpdate).toHaveBeenCalledTimes(1);
        expect(input).toHaveValue('2024-05-05');
        expect(onUpdate).toHaveBeenCalledWith({
            target: {
                name: 'date',
                value: new Date('2024-05-05'),
            },
            type: 'change',
            updateLocation: 'input',
        });
    });
    it('should update date on calendar day click', async () => {
        const onUpdate = jest.fn();
        render(_jsx(DatePickerRHFExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_528", onUpdate: onUpdate }));
        expect(onUpdate).toHaveBeenCalledTimes(0);
        const input = getDatePickerInput();
        await userEvent.click(input);
        expect(getDatePickerDialog()).toBeInTheDocument();
        const dateCell = getDatePickerDateCell('Thursday, January 15, 1970');
        await userEvent.click(dateCell);
        expect(onUpdate).toHaveBeenCalledTimes(1);
        expect(input).toHaveValue('1970-01-15');
        expect(onUpdate).toHaveBeenCalledWith({
            target: {
                name: 'date',
                value: new Date('1970-01-15'),
            },
            type: 'change',
            updateLocation: 'calendar',
        });
    });
    it('should update date on input value change - with control', async () => {
        const onUpdate = jest.fn();
        render(_jsx(DatePickerRHFControlExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_557", onUpdate: onUpdate }));
        expect(onUpdate).toHaveBeenCalledTimes(0);
        const input = getDatePickerInput();
        fireEvent.focus(input);
        fireEvent.change(input, { target: { value: '2024-05-05' } });
        expect(onUpdate).toHaveBeenCalledTimes(1);
        expect(input).toHaveValue('2024-05-05');
        expect(onUpdate).toHaveBeenCalledWith(new Date('2024-05-05'));
    });
    it('should update date on calendar day click - with control', async () => {
        const onUpdate = jest.fn();
        render(_jsx(DatePickerRHFControlExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_575", onUpdate: onUpdate }));
        expect(onUpdate).toHaveBeenCalledTimes(0);
        const input = getDatePickerInput();
        await userEvent.click(input);
        expect(getDatePickerDialog()).toBeInTheDocument();
        const dateCell = getDatePickerDateCell('Thursday, January 15, 1970');
        await userEvent.click(dateCell);
        expect(onUpdate).toHaveBeenCalledTimes(1);
        expect(input).toHaveValue('1970-01-15');
        expect(onUpdate).toHaveBeenCalledWith(new Date('1970-01-15'));
    });
});
describe('RangePicker with React Hook Form', () => {
    it('should update dates on input value change', async () => {
        const onUpdate = jest.fn();
        render(_jsx(RangePickerRHFExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_599", onUpdate: onUpdate }));
        expect(onUpdate).toHaveBeenCalledTimes(0);
        const rangePickerInputs = getRangePickerInputs();
        const fromInput = rangePickerInputs[0];
        fireEvent.focus(fromInput);
        fireEvent.change(fromInput, { target: { value: '2024-05-05' } });
        expect(onUpdate).toHaveBeenCalledTimes(1);
        expect(fromInput).toHaveValue('2024-05-05');
        expect(onUpdate).toHaveBeenCalledWith({
            target: {
                name: 'from',
                value: new Date('2024-05-05'),
            },
            type: 'change',
            updateLocation: 'input',
        });
        const toInput = rangePickerInputs[1];
        fireEvent.focus(toInput);
        fireEvent.change(toInput, { target: { value: '2025-05-05' } });
        expect(onUpdate).toHaveBeenCalledTimes(2);
        expect(toInput).toHaveValue('2025-05-05');
        expect(onUpdate).toHaveBeenCalledWith({
            target: {
                name: 'to',
                value: new Date('2025-05-05'),
            },
            type: 'change',
            updateLocation: 'input',
        });
    });
    it('should update dates on calendar day click', async () => {
        const onUpdate = jest.fn();
        render(_jsx(RangePickerRHFExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_645", onUpdate: onUpdate }));
        expect(onUpdate).toHaveBeenCalledTimes(0);
        const rangePickerInputs = getRangePickerInputs();
        const fromInput = rangePickerInputs[0];
        await userEvent.click(fromInput);
        expect(getDatePickerDialog()).toBeInTheDocument();
        const fromDateCell = getDatePickerDateCell('Thursday, January 15, 1970');
        await userEvent.click(fromDateCell);
        expect(onUpdate).toHaveBeenCalledTimes(1);
        expect(fromInput).toHaveValue('1970-01-15');
        expect(onUpdate).toHaveBeenCalledWith({
            target: {
                name: 'from',
                value: new Date('1970-01-15'),
            },
            type: 'change',
            updateLocation: 'calendar',
        });
        const toInput = rangePickerInputs[1];
        expect(getDatePickerDialog()).toBeInTheDocument();
        const toDateCell = getDatePickerDateCell('Tuesday, January 20, 1970');
        await userEvent.click(toDateCell);
        expect(onUpdate).toHaveBeenCalledTimes(2);
        expect(toInput).toHaveValue('1970-01-20');
        expect(onUpdate).toHaveBeenCalledWith({
            target: {
                name: 'to',
                value: new Date('1970-01-20'),
            },
            type: 'change',
            updateLocation: 'calendar',
        });
    });
    it('should update dates on input value change - with control', async () => {
        const onUpdate = jest.fn();
        render(_jsx(RangePickerRHFControlExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_697", onUpdate: onUpdate }));
        expect(onUpdate).toHaveBeenCalledTimes(0);
        const rangePickerInputs = getRangePickerInputs();
        const fromInput = rangePickerInputs[0];
        fireEvent.focus(fromInput);
        fireEvent.change(fromInput, { target: { value: '2024-05-05' } });
        expect(onUpdate).toHaveBeenCalledTimes(1);
        expect(fromInput).toHaveValue('2024-05-05');
        expect(onUpdate).toHaveBeenCalledWith(new Date('2024-05-05'));
        const toInput = rangePickerInputs[1];
        fireEvent.focus(toInput);
        fireEvent.change(toInput, { target: { value: '2025-05-05' } });
        expect(onUpdate).toHaveBeenCalledTimes(2);
        expect(toInput).toHaveValue('2025-05-05');
        expect(onUpdate).toHaveBeenCalledWith(new Date('2025-05-05'));
    });
    it('should update dates on calendar day click - with control', async () => {
        const onUpdate = jest.fn();
        render(_jsx(RangePickerRHFControlExample, { componentId: "codegen_design-system_src_development_datepicker_datepicker.test.tsx_728", onUpdate: onUpdate }));
        expect(onUpdate).toHaveBeenCalledTimes(0);
        const rangePickerInputs = getRangePickerInputs();
        const fromInput = rangePickerInputs[0];
        await userEvent.click(fromInput);
        expect(getDatePickerDialog()).toBeInTheDocument();
        const fromDateCell = getDatePickerDateCell('Thursday, January 15, 1970');
        await userEvent.click(fromDateCell);
        expect(onUpdate).toHaveBeenCalledTimes(1);
        expect(fromInput).toHaveValue('1970-01-15');
        expect(onUpdate).toHaveBeenCalledWith(new Date('1970-01-15'));
        const toInput = rangePickerInputs[1];
        expect(getDatePickerDialog()).toBeInTheDocument();
        const toDateCell = getDatePickerDateCell('Tuesday, January 20, 1970');
        await userEvent.click(toDateCell);
        expect(onUpdate).toHaveBeenCalledTimes(2);
        expect(toInput).toHaveValue('1970-01-20');
        expect(onUpdate).toHaveBeenCalledWith(new Date('1970-01-20'));
    });
});
//# sourceMappingURL=DatePicker.test.js.map