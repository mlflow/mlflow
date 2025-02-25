import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Controller, useForm } from 'react-hook-form';

import type { DatePickerChangeEventType, DatePickerProps, RangePickerProps } from './DatePicker';
import { DatePicker, RangePicker } from './DatePicker';
import { DesignSystemProvider } from '../../design-system';

const DatePickerExample = (props: DatePickerProps) => {
  return (
    <DesignSystemProvider>
      <DatePicker {...props} />
    </DesignSystemProvider>
  );
};

const RangePickerExample = (props: RangePickerProps) => {
  return (
    <DesignSystemProvider>
      <RangePicker {...props} />
    </DesignSystemProvider>
  );
};

interface DatePickerPropsForRHFExample extends DatePickerProps {
  onUpdate: (formData: any) => void;
}

const DatePickerRHFExample = (props: DatePickerPropsForRHFExample) => {
  const { onUpdate, ...restProps } = props;
  const { register, setValue } = useForm<{
    date?: Date;
  }>();

  return (
    <DesignSystemProvider>
      <DatePicker
        {...register('date', {
          onChange: (e) => {
            onUpdate(e);
            setValue('date', e);
          },
        })}
        {...restProps}
      />
    </DesignSystemProvider>
  );
};

const DatePickerRHFControlExample = (props: DatePickerPropsForRHFExample) => {
  const { onUpdate } = props;
  const { control } = useForm<{
    date?: Date;
  }>();

  return (
    <DesignSystemProvider>
      <Controller
        name="date"
        control={control}
        render={({ field }) => (
          <DatePicker
            componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_62"
            onChange={(e: DatePickerChangeEventType) => {
              onUpdate(e.target.value);
              field.onChange(e.target.value);
            }}
            value={field.value}
          />
        )}
      />
    </DesignSystemProvider>
  );
};

interface RangePickerPropsForRHFExample extends DatePickerProps {
  onUpdate: (formData: any) => void;
}

const RangePickerRHFExample = (props: RangePickerPropsForRHFExample) => {
  const { onUpdate } = props;
  const { register, setValue } = useForm<{
    from?: Date;
    to?: Date;
  }>();

  return (
    <DesignSystemProvider>
      <RangePicker
        startDatePickerProps={{
          componentId: 'YOUR_TRACKING_ID',
          ...register('from', {
            onChange: (e) => {
              onUpdate(e);
              setValue('from', e);
            },
          }),
        }}
        endDatePickerProps={{
          componentId: 'YOUR_TRACKING_ID',
          ...register('to', {
            onChange: (e) => {
              onUpdate(e);
              setValue('to', e);
            },
          }),
        }}
      />
    </DesignSystemProvider>
  );
};

const RangePickerRHFControlExample = (props: RangePickerPropsForRHFExample) => {
  const { onUpdate } = props;
  const { control } = useForm<{
    range: {
      from?: Date;
      to?: Date;
    };
  }>();

  return (
    <DesignSystemProvider>
      <Controller
        name="range"
        control={control}
        render={({ field }) => (
          <RangePicker
            startDatePickerProps={{
              componentId: 'YOUR_TRACKING_ID',
              onChange: (e: DatePickerChangeEventType) => {
                onUpdate(e.target.value);
                field.onChange({ ...field.value, from: e.target.value });
              },
              value: field.value?.from,
            }}
            endDatePickerProps={{
              componentId: 'YOUR_TRACKING_ID',
              onChange: (e: DatePickerChangeEventType) => {
                onUpdate(e.target.value);
                field.onChange({ ...field.value, to: e.target.value });
              },
              value: field.value?.to,
            }}
          />
        )}
      />
    </DesignSystemProvider>
  );
};

const getDatePickerInput = (name = 'Select Date') => screen.getByRole('textbox', { name });
const getDatePickerDialog = () => screen.getByRole('dialog');
const getDatePickerDateCell = (date: string) => screen.getByRole('button', { name: date });
const getClearButton = () => screen.getByRole('button', { name: 'close-circle' });
const getTimeInput = () => screen.getByRole('textbox', { name: 'Time' });

describe('DatePicker', () => {
  it('should render DatePicker', () => {
    render(
      <DatePickerExample componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_155" />,
    );

    expect(getDatePickerInput()).toBeInTheDocument();
  });

  it('should update date on input value change', async () => {
    render(
      <DatePickerExample componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_161" />,
    );

    const input = getDatePickerInput();
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: '2024-05-05' } });

    expect(input).toHaveValue('2024-05-05');
  });

  it('should update date and time on input value change', async () => {
    render(
      <DatePickerExample
        componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_176"
        includeTime
      />,
    );

    const input = getDatePickerInput('Select Date and Time');
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: '2024-05-05 22:00' } });

    expect(input).toHaveValue('2024-05-05T22:00');
  });

  it('should open datepicker dialog on input click', async () => {
    render(
      <DatePickerExample componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_190" />,
    );

    const input = getDatePickerInput();
    await userEvent.click(input);

    const dialog = getDatePickerDialog();
    expect(dialog).toBeInTheDocument();
  });

  it('should select date on date click', async () => {
    render(
      <DatePickerExample componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_201" />,
    );

    const input = getDatePickerInput();
    await userEvent.click(input);

    const dialog = getDatePickerDialog();
    expect(dialog).toBeInTheDocument();

    const dateCell = getDatePickerDateCell('15');
    await userEvent.click(dateCell);

    expect(input).toHaveValue('1970-01-15');
  });

  it('should select date on date enter key press', async () => {
    render(
      <DatePickerExample componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_219" />,
    );

    const input = getDatePickerInput();
    await userEvent.click(input);

    const dialog = getDatePickerDialog();
    await waitFor(async () => {
      expect(dialog).toBeInTheDocument();
    });

    const dateCell = getDatePickerDateCell('15');
    await userEvent.type(dateCell, '{enter}');

    await waitFor(async () => {
      expect(dialog).not.toBeInTheDocument();
    });

    expect(input).toHaveValue('1970-01-15');
  });

  it('should clear date on clear button click', async () => {
    render(
      <DatePickerExample
        componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_237"
        allowClear
      />,
    );

    const input = getDatePickerInput();
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: '2024-05-05' } });

    expect(input).toHaveValue('2024-05-05');

    const clearButton = getClearButton();
    await userEvent.click(clearButton);

    expect(input).toHaveValue('');
  });

  it('should select date and update time on date click and time change', async () => {
    render(
      <DatePickerExample
        componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_258"
        includeTime
      />,
    );

    const input = getDatePickerInput('Select Date and Time');

    await userEvent.click(input);

    const dialog = getDatePickerDialog();
    expect(dialog).toBeInTheDocument();

    const dateCell = getDatePickerDateCell('15');
    await userEvent.click(dateCell);

    expect(input).toHaveValue('1970-01-15T00:00');

    const timeInput = getTimeInput();

    fireEvent.focus(timeInput);
    fireEvent.change(timeInput, { target: { value: '22:00' } });

    expect(input).toHaveValue('1970-01-15T22:00');
  });
});

const getRangePickerInputs = (name = 'Select Date') => screen.getAllByRole('textbox', { name });
const getRangePickerClearButtons = () => screen.getAllByRole('button', { name: 'close-circle' });

describe('RangePicker', () => {
  it('should render RangePicker', () => {
    render(<RangePickerExample />);

    const rangePickerInputs = getRangePickerInputs();
    expect(rangePickerInputs).toHaveLength(2);

    expect(rangePickerInputs[0]).toBeInTheDocument();
    expect(rangePickerInputs[1]).toBeInTheDocument();
  });

  it('should update dates on input value change', async () => {
    render(<RangePickerExample />);

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
    render(<RangePickerExample includeTime />);

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

  it('should open datepicker dialog on input click', async () => {
    render(<RangePickerExample />);

    const rangePickerInputs = getRangePickerInputs();
    const fromInput = rangePickerInputs[0];
    await userEvent.click(fromInput);

    expect(getDatePickerDialog()).toBeInTheDocument();

    const toInput = rangePickerInputs[1];
    await userEvent.click(toInput);

    expect(getDatePickerDialog()).toBeInTheDocument();
  });

  it('should select date on date click', async () => {
    render(<RangePickerExample />);

    const rangePickerInputs = getRangePickerInputs();
    const fromInput = rangePickerInputs[0];
    await userEvent.click(fromInput);

    expect(getDatePickerDialog()).toBeInTheDocument();

    const fromDateCell = getDatePickerDateCell('15');
    await userEvent.click(fromDateCell);

    expect(fromInput).toHaveValue('1970-01-15');

    const toInput = rangePickerInputs[1];
    expect(getDatePickerDialog()).toBeInTheDocument();

    const toDateCell = getDatePickerDateCell('20');
    await userEvent.click(toDateCell);

    expect(toInput).toHaveValue('1970-01-20');
  });

  it('should have starting date selected in the second datepicker dialog', async () => {
    render(<RangePickerExample />);

    const rangePickerInputs = getRangePickerInputs();
    const fromInput = rangePickerInputs[0];
    await userEvent.click(fromInput);

    expect(getDatePickerDialog()).toBeInTheDocument();

    const fromDateCell = getDatePickerDateCell('15');
    await userEvent.click(fromDateCell);

    expect(fromInput).toHaveValue('1970-01-15');

    expect(getDatePickerDialog()).toBeInTheDocument();
    expect(getDatePickerDateCell('15')).toHaveAttribute('aria-selected', 'true');
  });

  it('should have ending date select in the first datepicker dialog', async () => {
    render(<RangePickerExample />);

    const rangePickerInputs = getRangePickerInputs();
    const toInput = rangePickerInputs[1];
    await userEvent.click(toInput);

    expect(getDatePickerDialog()).toBeInTheDocument();

    const toDateCell = getDatePickerDateCell('15');
    await userEvent.click(toDateCell);

    expect(toInput).toHaveValue('1970-01-15');

    const fromInput = rangePickerInputs[0];
    await userEvent.click(fromInput);

    expect(getDatePickerDialog()).toBeInTheDocument();
    expect(getDatePickerDateCell('15')).toHaveAttribute('aria-selected', 'true');
  });

  it('should clear starting date on clear button click', async () => {
    render(<RangePickerExample allowClear />);

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
    render(<RangePickerExample allowClear />);

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
    render(
      <DatePickerRHFExample
        componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_503"
        onUpdate={onUpdate}
      />,
    );

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
    render(
      <DatePickerRHFExample
        componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_528"
        onUpdate={onUpdate}
      />,
    );

    expect(onUpdate).toHaveBeenCalledTimes(0);
    const input = getDatePickerInput();
    await userEvent.click(input);

    expect(getDatePickerDialog()).toBeInTheDocument();

    const dateCell = getDatePickerDateCell('15');
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
    render(
      <DatePickerRHFControlExample
        componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_557"
        onUpdate={onUpdate}
      />,
    );

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
    render(
      <DatePickerRHFControlExample
        componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_575"
        onUpdate={onUpdate}
      />,
    );

    expect(onUpdate).toHaveBeenCalledTimes(0);
    const input = getDatePickerInput();
    await userEvent.click(input);

    expect(getDatePickerDialog()).toBeInTheDocument();

    const dateCell = getDatePickerDateCell('15');
    await userEvent.click(dateCell);

    expect(onUpdate).toHaveBeenCalledTimes(1);
    expect(input).toHaveValue('1970-01-15');
    expect(onUpdate).toHaveBeenCalledWith(new Date('1970-01-15'));
  });
});

describe('RangePicker with React Hook Form', () => {
  it('should update dates on input value change', async () => {
    const onUpdate = jest.fn();
    render(
      <RangePickerRHFExample
        componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_599"
        onUpdate={onUpdate}
      />,
    );

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
    render(
      <RangePickerRHFExample
        componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_645"
        onUpdate={onUpdate}
      />,
    );

    expect(onUpdate).toHaveBeenCalledTimes(0);
    const rangePickerInputs = getRangePickerInputs();
    const fromInput = rangePickerInputs[0];
    await userEvent.click(fromInput);

    expect(getDatePickerDialog()).toBeInTheDocument();

    const fromDateCell = getDatePickerDateCell('15');
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

    const toDateCell = getDatePickerDateCell('20');
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
    render(
      <RangePickerRHFControlExample
        componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_697"
        onUpdate={onUpdate}
      />,
    );

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
    render(
      <RangePickerRHFControlExample
        componentId="codegen_design-system_src_development_datepicker_datepicker.test.tsx_728"
        onUpdate={onUpdate}
      />,
    );

    expect(onUpdate).toHaveBeenCalledTimes(0);
    const rangePickerInputs = getRangePickerInputs();
    const fromInput = rangePickerInputs[0];
    await userEvent.click(fromInput);

    expect(getDatePickerDialog()).toBeInTheDocument();

    const fromDateCell = getDatePickerDateCell('15');
    await userEvent.click(fromDateCell);

    expect(onUpdate).toHaveBeenCalledTimes(1);
    expect(fromInput).toHaveValue('1970-01-15');
    expect(onUpdate).toHaveBeenCalledWith(new Date('1970-01-15'));

    const toInput = rangePickerInputs[1];

    expect(getDatePickerDialog()).toBeInTheDocument();

    const toDateCell = getDatePickerDateCell('20');
    await userEvent.click(toDateCell);

    expect(onUpdate).toHaveBeenCalledTimes(2);
    expect(toInput).toHaveValue('1970-01-20');
    expect(onUpdate).toHaveBeenCalledWith(new Date('1970-01-20'));
  });
});
