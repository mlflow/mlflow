import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { createRef, forwardRef, useImperativeHandle, useReducer, useState } from 'react';
import type { UseFormSetValue } from 'react-hook-form';
import { Controller, useForm } from 'react-hook-form';

import { SimpleSelect, SimpleSelectOption, SimpleSelectOptionGroup } from './SimpleSelect';
import { simpleSelectTestUtils } from '../../test-utils/rtl';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider';

/** waits a little bit to ensure the assertions within the callback never pass */
async function expectNever(callback: () => unknown): Promise<void> {
  await expect(waitFor(callback)).rejects.toEqual(expect.anything());
}

const commonTest = () => {
  simpleSelectTestUtils.expectSelectedOptionFromTriggerToBe("Bar's Label");
  simpleSelectTestUtils.toggleSelect();
  simpleSelectTestUtils.expectOptionsLengthToBe(3);
  simpleSelectTestUtils.expectSelectedOptionToBe("Bar's Label");

  simpleSelectTestUtils.selectOption("Foo's Label");
  simpleSelectTestUtils.expectSelectedOptionFromTriggerToBe("Foo's Label");

  simpleSelectTestUtils.expectSelectToBeClosed();
  simpleSelectTestUtils.toggleSelect();
  simpleSelectTestUtils.expectSelectedOptionToBe("Foo's Label");
};

describe('SimpleSelect', () => {
  test('display label updates when uncontrolled + handles re-renders correctly', () => {
    function Test() {
      // Force a re-render.
      const [, anythingThatCouldCauseARerender] = useReducer((s) => s + 1, 0);

      return (
        <SimpleSelect
          componentId="codegen_design-system_src_design-system_select_simpleselect.test.tsx_36"
          defaultValue="bar"
          label="hello"
          onChange={() => anythingThatCouldCauseARerender()}
        >
          <SimpleSelectOption value="foo">Foo's Label</SimpleSelectOption>
          <SimpleSelectOption value="bar">Bar's Label</SimpleSelectOption>
          <SimpleSelectOption value="baz">Baz's Label</SimpleSelectOption>
        </SimpleSelect>
      );
    }

    render(<Test />);

    expect(screen.getByRole('combobox').textContent).toBe("Bar's Label");

    fireEvent.click(screen.getByRole('combobox'));

    expect(screen.queryAllByRole('option')).toHaveLength(3);
    expect(screen.getByRole('option', { name: "Foo's Label", selected: false })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: "Bar's Label", selected: true })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: "Baz's Label", selected: false })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('option', { name: "Foo's Label" }));

    expect(screen.getByRole('combobox').textContent).toBe("Foo's Label");
    expect(screen.queryAllByRole('option')).toHaveLength(0);
  });

  test('works uncontrolled', () => {
    render(
      <SimpleSelect
        componentId="codegen_design-system_src_design-system_select_simpleselect.test.tsx_63"
        id="simple-select"
        placeholder="Choose an option"
        value="bar"
      >
        <SimpleSelectOption value="foo">Foo's Label</SimpleSelectOption>
        <SimpleSelectOption value="bar">Bar's Label</SimpleSelectOption>
        <SimpleSelectOption value="baz">Baz's Label</SimpleSelectOption>
      </SimpleSelect>,
    );

    commonTest();
    expect(true).toBe(true); // Add an assertion to satisfy the lint requirement
  });

  test('works uncontrolled with defaultValue', () => {
    render(
      <SimpleSelect
        componentId="codegen_design-system_src_design-system_select_simpleselect.test.tsx_76"
        id="simple-select"
        placeholder="Choose an option"
        defaultValue="bar"
      >
        <SimpleSelectOption value="foo">Foo's Label</SimpleSelectOption>
        <SimpleSelectOption value="bar">Bar's Label</SimpleSelectOption>
        <SimpleSelectOption value="baz">Baz's Label</SimpleSelectOption>
      </SimpleSelect>,
    );

    expect(screen.getByRole('combobox').textContent).toBe("Bar's Label");

    simpleSelectTestUtils.toggleSelect();

    expect(screen.queryAllByRole('option')).toHaveLength(3);
    expect(screen.getByRole('option', { name: "Foo's Label", selected: false })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: "Bar's Label", selected: true })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: "Baz's Label", selected: false })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('option', { name: "Foo's Label" }));

    expect(screen.getByRole('combobox').textContent).toBe("Foo's Label");
    expect(screen.queryAllByRole('option')).toHaveLength(0);
  });

  test('works controlled', () => {
    const onChange = jest.fn();

    const SimpleSelectWrapper = () => {
      const [selectedValue, setSelectedValue] = useState('bar');

      return (
        <SimpleSelect
          componentId="codegen_design-system_src_design-system_select_simpleselect.test.tsx_105"
          id="simple-select"
          placeholder="Choose an option"
          value={selectedValue}
          onChange={(e) => {
            setSelectedValue(e.target.value);
            onChange(e);
          }}
        >
          <SimpleSelectOption value="foo">Foo's Label</SimpleSelectOption>
          <SimpleSelectOption value="bar">Bar's Label</SimpleSelectOption>
          <SimpleSelectOption value="baz">Baz's Label</SimpleSelectOption>
        </SimpleSelect>
      );
    };

    render(<SimpleSelectWrapper />);

    commonTest();
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        target: expect.objectContaining({
          value: 'foo',
        }),
      }),
    );
  });

  test('works uncontrolled with groups', () => {
    render(
      <SimpleSelect
        componentId="codegen_design-system_src_design-system_select_simpleselect.test.tsx_135"
        id="simple-select"
        placeholder="Choose an option"
        value="bar"
      >
        <SimpleSelectOptionGroup label="Group 1">
          <SimpleSelectOption value="foo">Foo's Label</SimpleSelectOption>
          <SimpleSelectOption value="bar">Bar's Label</SimpleSelectOption>
        </SimpleSelectOptionGroup>
        <SimpleSelectOptionGroup label="Group 2">
          <SimpleSelectOption value="baz">Baz's Label</SimpleSelectOption>
        </SimpleSelectOptionGroup>
      </SimpleSelect>,
    );

    commonTest();
    expect(true).toBe(true); // Add an assertion to satisfy the lint requirement
  });

  test('works controlled with groups', () => {
    const onChange = jest.fn();

    const SimpleSelectWrapper = () => {
      const [selectedValue, setSelectedValue] = useState('bar');

      return (
        <SimpleSelect
          componentId="codegen_design-system_src_design-system_select_simpleselect.test.tsx_157"
          id="simple-select"
          placeholder="Choose an option"
          value={selectedValue}
          onChange={(e) => {
            setSelectedValue(e.target.value);
            onChange(e);
          }}
        >
          <SimpleSelectOptionGroup label="Group 1">
            <SimpleSelectOption value="foo">Foo's Label</SimpleSelectOption>
            <SimpleSelectOption value="bar">Bar's Label</SimpleSelectOption>
          </SimpleSelectOptionGroup>
          <SimpleSelectOptionGroup label="Group 2">
            <SimpleSelectOption value="baz">Baz's Label</SimpleSelectOption>
          </SimpleSelectOptionGroup>
        </SimpleSelect>
      );
    };

    render(<SimpleSelectWrapper />);

    commonTest();
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        target: expect.objectContaining({
          value: 'foo',
        }),
      }),
    );
  });

  test('works uncontrolled with string and React node children', () => {
    const StringComponent = () => <>Foo's Label</>;

    render(
      <SimpleSelect
        componentId="codegen_design-system_src_design-system_select_simpleselect.test.tsx_193"
        id="simple-select"
        placeholder="Choose an option"
        value="bar"
      >
        <SimpleSelectOption value="foo">
          <StringComponent />
        </SimpleSelectOption>
        <SimpleSelectOption value="bar">Bar's Label</SimpleSelectOption>
      </SimpleSelect>,
    );

    // Open the dropdown
    simpleSelectTestUtils.toggleSelect();

    expect(screen.getByRole('option', { name: "Foo's Label" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('option', { name: "Foo's Label" }));

    expect(screen.getByRole('combobox').textContent).toBe("Foo's Label");
  });

  test('works controlled with string and React node children', () => {
    const onChange = jest.fn();
    const StringComponent = () => <>Bar's Label</>;

    const SimpleSelectWrapper = () => {
      const [selectedValue, setSelectedValue] = useState('foo');

      return (
        <SimpleSelect
          componentId="codegen_design-system_src_design-system_select_simpleselect.test.tsx_219"
          id="simple-select"
          placeholder="Choose an option"
          value={selectedValue}
          onChange={(e) => {
            setSelectedValue(e.target.value);
            onChange(e);
          }}
        >
          <SimpleSelectOption value="foo">Foo's Label</SimpleSelectOption>
          <SimpleSelectOption value="bar">
            <StringComponent />
          </SimpleSelectOption>
        </SimpleSelect>
      );
    };

    render(<SimpleSelectWrapper />);

    // Open the dropdown
    simpleSelectTestUtils.toggleSelect();

    expect(screen.getByRole('option', { name: "Foo's Label", selected: true })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('option', { name: "Bar's Label" }));

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        target: expect.objectContaining({
          value: 'bar',
        }),
      }),
    );
    expect(screen.getByRole('combobox').textContent).toBe("Bar's Label");
  });

  test('clears the selected value when the clear button is clicked', () => {
    render(
      <SimpleSelect
        componentId="codegen_design-system_src_design-system_select_simpleselect.test.tsx_257"
        id="simple-select"
        placeholder="Choose an option"
        allowClear
      >
        <SimpleSelectOption value="foo">Foo's Label</SimpleSelectOption>
        <SimpleSelectOption value="bar">Bar's Label</SimpleSelectOption>
        <SimpleSelectOption value="baz">Baz's Label</SimpleSelectOption>
      </SimpleSelect>,
    );

    // Open the dropdown
    simpleSelectTestUtils.toggleSelect();

    // Select an option
    fireEvent.click(screen.getByRole('option', { name: "Bar's Label" }));

    // Check if the selected option is displayed
    expect(screen.getByRole('combobox')).toHaveTextContent("Bar's Label");

    // Click the clear button
    fireEvent.click(screen.getByRole('button', { name: /clear/i }));

    // Check if the placeholder is displayed after clearing
    expect(screen.getByRole('combobox')).toHaveTextContent('Choose an option');
  });

  test('should not call onChange when initially mounted', async () => {
    const onChange = jest.fn();

    render(
      <SimpleSelect
        componentId="codegen_design-system_src_design-system_select_simpleselect.test.tsx_284"
        onChange={onChange}
        value="bar"
        label="test"
      >
        <SimpleSelectOption value="foo">Foo's Label</SimpleSelectOption>
        <SimpleSelectOption value="bar">Bar's Label</SimpleSelectOption>
        <SimpleSelectOption value="baz">Baz's Label</SimpleSelectOption>
      </SimpleSelect>,
    );

    await expectNever(() => {
      expect(onChange).toHaveBeenCalled();
    });
  });

  test('should not cause an infinite render loop if state is an object', async () => {
    const ComplexStateComponent = () => {
      const [value, setValue] = useState({ value: 'pie' });

      return (
        <>
          <SimpleSelect
            componentId="codegen_design-system_src_design-system_select_simpleselect.test.tsx_302"
            id="infinite-render-select"
            placeholder="Choose an option"
            value={value.value}
            onChange={(e) => setValue({ value: e.target.value })}
          >
            <SimpleSelectOption value="pie">
              <>Pie</>
            </SimpleSelectOption>
            <SimpleSelectOption value="bar">Bar</SimpleSelectOption>
            <SimpleSelectOption value="line">Line</SimpleSelectOption>
            <SimpleSelectOption value="bubble">
              <>Bubble</>
            </SimpleSelectOption>
            <SimpleSelectOption value="column">Column</SimpleSelectOption>
          </SimpleSelect>
        </>
      );
    };

    expect(() => {
      render(<ComplexStateComponent />);
    }).not.toThrow();
  });

  test('closes dropdown when selected value is re-selected', () => {
    render(
      <SimpleSelect
        componentId="codegen_design-system_src_design-system_select_simpleselect.test.tsx_329"
        id="simple-select"
        value="foo"
        placeholder="Choose an option"
        allowClear
        label="test"
      >
        <SimpleSelectOption value="foo">Foo's Label</SimpleSelectOption>
        <SimpleSelectOption value="bar">Bar's Label</SimpleSelectOption>
        <SimpleSelectOption value="baz">Baz's Label</SimpleSelectOption>
      </SimpleSelect>,
    );

    expect(screen.getByRole('combobox')).toHaveTextContent("Foo's Label");
    simpleSelectTestUtils.toggleSelect();
    fireEvent.click(screen.getByRole('option', { name: "Foo's Label" }));
    // check if the dropdown is closed
    expect(screen.queryAllByRole('option')).toHaveLength(0);
    expect(screen.getByRole('combobox')).toHaveTextContent("Foo's Label");
  });

  test('triggers the onOpenChange callback when the dropdown is opened or closed', () => {
    const onOpenChange = jest.fn();

    const SimpleSelectWrapper = () => {
      return (
        <SimpleSelect
          componentId="codegen_design-system_src_design-system_select_simpleselect.test.tsx_349"
          placeholder="Choose an option"
          onOpenChange={onOpenChange}
          label="test"
        >
          <SimpleSelectOption value="foo">Foo's Label</SimpleSelectOption>
          <SimpleSelectOption value="bar">Bar's Label</SimpleSelectOption>
        </SimpleSelect>
      );
    };

    render(<SimpleSelectWrapper />);

    // Initial render; should be false.
    expect(onOpenChange).toHaveBeenLastCalledWith(false);

    // Open the dropdown
    simpleSelectTestUtils.toggleSelect();
    expect(onOpenChange).toHaveBeenLastCalledWith(true);

    // trigger onOpenChange on click trigger again
    simpleSelectTestUtils.toggleSelect();
    expect(onOpenChange).toHaveBeenLastCalledWith(false);

    // trigger onOpenChange on chose new value
    simpleSelectTestUtils.toggleSelect();
    expect(onOpenChange).toHaveBeenLastCalledWith(true);

    fireEvent.click(screen.getByRole('option', { name: "Bar's Label" }));
    expect(onOpenChange).toHaveBeenLastCalledWith(false);

    expect(onOpenChange).toHaveBeenCalledTimes(5);
  });

  describe('Analytics Events', () => {
    const eventCallback = jest.fn();

    const UncontrolledComponent = ({
      valueHasNoPii,
      defaultValue,
      allowClear,
    }: {
      valueHasNoPii?: boolean;
      defaultValue?: string;
      allowClear?: boolean;
    }) => (
      <DesignSystemEventProvider callback={eventCallback}>
        <SimpleSelect
          componentId="simple_select_test"
          valueHasNoPii={valueHasNoPii}
          id="simple-select"
          placeholder="Choose an option"
          defaultValue={defaultValue}
          allowClear={allowClear}
        >
          <SimpleSelectOption value="foo">Foo's Label</SimpleSelectOption>
          <SimpleSelectOption value="bar">Bar's Label</SimpleSelectOption>
          <SimpleSelectOption value="baz">Baz's Label</SimpleSelectOption>
        </SimpleSelect>
      </DesignSystemEventProvider>
    );

    const ControlledComponent = ({
      valueHasNoPii,
      defaultValue,
      allowClear,
    }: {
      valueHasNoPii?: boolean;
      defaultValue?: string;
      allowClear?: boolean;
    }) => {
      const [selectedValue, setSelectedValue] = useState(defaultValue);

      return (
        <DesignSystemEventProvider callback={eventCallback}>
          <SimpleSelect
            componentId="simple_select_test"
            valueHasNoPii={valueHasNoPii}
            id="simple-select"
            placeholder="Choose an option"
            value={selectedValue}
            onChange={(e) => {
              setSelectedValue(e.target.value);
            }}
            allowClear={allowClear}
          >
            <SimpleSelectOption value="foo">Foo's Label</SimpleSelectOption>
            <SimpleSelectOption value="bar">Bar's Label</SimpleSelectOption>
            <SimpleSelectOption value="baz">Baz's Label</SimpleSelectOption>
          </SimpleSelect>
        </DesignSystemEventProvider>
      );
    };

    describe.each([
      { Component: UncontrolledComponent, title: 'uncontrolled' },
      { Component: ControlledComponent, title: 'controlled' },
    ])('$title simple select analytics events', ({ Component }) => {
      test('emits value change events with value', () => {
        render(<Component valueHasNoPii />);
        expect(eventCallback).not.toHaveBeenCalled();

        simpleSelectTestUtils.toggleSelect();
        expect(eventCallback).not.toHaveBeenCalled();
        fireEvent.click(screen.getByRole('option', { name: "Foo's Label" }));
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
          eventType: 'onValueChange',
          componentId: 'simple_select_test',
          componentType: 'simple_select',
          shouldStartInteraction: false,
          value: 'foo',
        });

        simpleSelectTestUtils.toggleSelect();
        fireEvent.click(screen.getByRole('option', { name: "Bar's Label" }));
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenCalledWith({
          eventType: 'onValueChange',
          componentId: 'simple_select_test',
          componentType: 'simple_select',
          shouldStartInteraction: false,
          value: 'bar',
        });
      });

      test('emits value change events without value when valueHasNoPii is not set', () => {
        render(<Component />);
        expect(eventCallback).not.toHaveBeenCalled();

        simpleSelectTestUtils.toggleSelect();
        expect(eventCallback).not.toHaveBeenCalled();
        fireEvent.click(screen.getByRole('option', { name: "Foo's Label" }));
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
          eventType: 'onValueChange',
          componentId: 'simple_select_test',
          componentType: 'simple_select',
          shouldStartInteraction: false,
          value: undefined,
        });
      });

      test('emits a value change event when the value is cleared', () => {
        render(<Component valueHasNoPii defaultValue="foo" allowClear />);
        expect(eventCallback).not.toHaveBeenCalled();

        fireEvent.click(screen.getByRole('button', { name: /clear/i }));
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
          eventType: 'onValueChange',
          componentId: 'simple_select_test',
          componentType: 'simple_select',
          shouldStartInteraction: false,
          value: '',
        });
      });

      test('does not emit a value change event for the default value', () => {
        render(<Component defaultValue="foo" />);
        expect(eventCallback).not.toHaveBeenCalled();
      });
    });
  });
});

interface TestRHFExamplesRef {
  setValue: UseFormSetValue<TestRHFExampleFormValues>;
}

interface TestRHFExampleFormValues {
  fruit: string;
  vegetable: string;
}

interface TestRHFExampleFormProps {
  allowClear?: boolean;
  defaultValues?: TestRHFExampleFormValues;
  onSubmit?: (data: TestRHFExampleFormValues) => void;
  rules?: { required: string };
}

const TestRHFControlExampleForm = forwardRef<TestRHFExamplesRef, TestRHFExampleFormProps>(
  ({ defaultValues, onSubmit = () => {}, rules, allowClear }: TestRHFExampleFormProps, ref) => {
    const { control, handleSubmit, setValue } = useForm<TestRHFExampleFormValues>({ defaultValues });

    useImperativeHandle(ref, () => ({
      setValue,
    }));

    return (
      <form onSubmit={handleSubmit(onSubmit)}>
        <Controller
          name="fruit"
          control={control}
          rules={rules}
          render={({ field, fieldState: { error } }) => (
            <>
              <SimpleSelect
                {...field}
                componentId="fruit_simple_select_test"
                valueHasNoPii
                placeholder="Select a fruit"
                label="Select a fruit"
                allowClear={allowClear}
              >
                <SimpleSelectOption value="apple">Apple</SimpleSelectOption>
                <SimpleSelectOption value="banana">Banana</SimpleSelectOption>
                <SimpleSelectOption value="orange">Orange</SimpleSelectOption>
              </SimpleSelect>
              {error && <span>{error.message}</span>}
            </>
          )}
        />
        <Controller
          name="vegetable"
          control={control}
          rules={rules}
          render={({ field, fieldState: { error } }) => (
            <>
              <SimpleSelect
                {...field}
                componentId="vegetable_simple_select_test"
                valueHasNoPii
                placeholder="Select a vegetable"
                label="Select a vegetable"
                allowClear={allowClear}
              >
                <SimpleSelectOptionGroup label="Leafy Greens">
                  <SimpleSelectOption value="spinach">Spinach</SimpleSelectOption>
                  <SimpleSelectOption value="kale">Kale</SimpleSelectOption>
                </SimpleSelectOptionGroup>
                <SimpleSelectOptionGroup label="Root Vegetables">
                  <SimpleSelectOption value="carrot">Carrot</SimpleSelectOption>
                  <SimpleSelectOption value="potato">Potato</SimpleSelectOption>
                </SimpleSelectOptionGroup>
              </SimpleSelect>
              {error && <span>{error.message}</span>}
            </>
          )}
        />
        <button type="submit">Submit</button>
      </form>
    );
  },
);

const TestRHFRegisterExampleForm = forwardRef<TestRHFExamplesRef, TestRHFExampleFormProps>(
  ({ defaultValues, onSubmit = () => {}, rules, allowClear }: TestRHFExampleFormProps, ref) => {
    const {
      register,
      handleSubmit,
      formState: { errors },
      setValue,
    } = useForm<TestRHFExampleFormValues>({ defaultValues });

    useImperativeHandle(ref, () => ({
      setValue,
    }));

    return (
      <form onSubmit={handleSubmit(onSubmit)}>
        <SimpleSelect
          {...register('fruit', rules)}
          componentId="fruit_simple_select_test"
          valueHasNoPii
          placeholder="Select a fruit"
          label="Select a fruit"
          allowClear={allowClear}
        >
          <SimpleSelectOption value="apple">Apple</SimpleSelectOption>
          <SimpleSelectOption value="banana">Banana</SimpleSelectOption>
          <SimpleSelectOption value="orange">Orange</SimpleSelectOption>
        </SimpleSelect>

        {errors.fruit && <span>{errors.fruit.message}</span>}

        <SimpleSelect
          {...register('vegetable', rules)}
          componentId="vegetable_simple_select_test"
          valueHasNoPii
          placeholder="Select a vegetable"
          label="Select a vegetable"
          allowClear={allowClear}
        >
          <SimpleSelectOptionGroup label="Leafy Greens">
            <SimpleSelectOption value="spinach">Spinach</SimpleSelectOption>
            <SimpleSelectOption value="kale">Kale</SimpleSelectOption>
          </SimpleSelectOptionGroup>
          <SimpleSelectOptionGroup label="Root Vegetables">
            <SimpleSelectOption value="carrot">Carrot</SimpleSelectOption>
            <SimpleSelectOption value="potato">Potato</SimpleSelectOption>
          </SimpleSelectOptionGroup>
        </SimpleSelect>
        {errors.vegetable && <span>{errors.vegetable.message}</span>}

        <button type="submit">Submit</button>
      </form>
    );
  },
);

[
  {
    TestFixture: TestRHFControlExampleForm,
    title: 'works with ReactHookForm control Syntax',
    testGroupId: 'control',
  },
  {
    TestFixture: TestRHFRegisterExampleForm,
    title: 'works with ReactHookForm register Syntax',
    testGroupId: 'register',
  },
].forEach(({ TestFixture, title, testGroupId }) => {
  // eslint-disable-next-line jest/valid-title -- Jest requires it to be a literal string... but we know it's a string.
  describe(title, () => {
    // eslint-disable-next-line jest/expect-expect
    test('integrates with React Hook Form', async () => {
      render(<TestFixture />);
      simpleSelectTestUtils.toggleSelect(/Select a fruit/);

      simpleSelectTestUtils.selectOption('Banana');
      simpleSelectTestUtils.expectSelectedOptionFromTriggerToBe('Banana', /Select a fruit/);
    });

    test('updates form values when selection changes', async () => {
      const onSubmit = jest.fn();
      render(<TestFixture defaultValues={{ fruit: '', vegetable: '' }} onSubmit={onSubmit} />);

      simpleSelectTestUtils.toggleSelect(/Select a fruit/);
      simpleSelectTestUtils.selectOption('Orange');
      fireEvent.submit(screen.getByRole('button', { name: 'Submit' }));

      await waitFor(() => {
        expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ fruit: 'orange' }), expect.anything());
        // Register syntax will return an empty string (matching real element) whereas control syntax will return undefined.
        expect(onSubmit.mock.calls[0][0]).toHaveProperty('vegetable');
        expect(['', undefined]).toContain(onSubmit.mock.calls[0][0].vegetable);
      });
    });

    // eslint-disable-next-line jest/expect-expect
    test('displays initial form value', () => {
      render(<TestFixture defaultValues={{ fruit: 'apple', vegetable: 'carrot' }} />);
      simpleSelectTestUtils.expectSelectedOptionFromTriggerToBe('Apple', /Select a fruit/);
    });

    test('displays initial form value with option groups', () => {
      const { getByRole } = render(
        <TestFixture
          defaultValues={{
            fruit: 'banana',
            vegetable: 'carrot',
          }}
        />,
      );

      expect(getByRole('combobox', { name: /Select a fruit/ })).toHaveTextContent('Banana');
      expect(getByRole('combobox', { name: /Select a vegetable/ })).toHaveTextContent('Carrot');
    });

    test('validates required field', async () => {
      render(<TestFixture rules={{ required: 'Please select a fruit' }} />);

      fireEvent.submit(screen.getByRole('button', { name: 'Submit' }));

      await waitFor(() => {
        expect(screen.getAllByText('Please select a fruit')).toHaveLength(2);
      });
    });

    test('works with allowClear', () => {
      render(<TestFixture rules={{ required: 'Please select a fruit' }} />);

      fireEvent.click(screen.getByRole('combobox', { name: /Select a fruit/ }));
      fireEvent.click(screen.getByRole('option', { name: 'Banana' }));

      expect(screen.getByRole('combobox', { name: /Select a fruit/ }).textContent).toBe('Banana');

      fireEvent.click(screen.getByRole('button', { name: /clear/i }));

      expect(screen.getByRole('combobox', { name: /Select a fruit/ }).textContent).toBe('Select a fruit');
    });

    test('works as expected with setValue (must be controlled)', () => {
      const ref = createRef<TestRHFExamplesRef>();

      render(
        <TestFixture
          ref={ref}
          defaultValues={{
            fruit: 'apple',
            vegetable: 'carrot',
          }}
        />,
      );

      expect(screen.getByRole('combobox', { name: /Select a fruit/ }).textContent).toBe('Apple');
      expect(screen.getByRole('combobox', { name: /Select a vegetable/ }).textContent).toBe('Carrot');

      act(() => {
        ref.current?.setValue('fruit', 'banana');
      });

      expect(screen.getByRole('combobox', { name: /Select a fruit/ }).textContent).toBe(
        testGroupId === 'control' ? 'Banana' : 'Apple',
      );

      act(() => {
        ref.current?.setValue('vegetable', 'kale');
      });

      expect(screen.getByRole('combobox', { name: /Select a vegetable/ }).textContent).toBe(
        testGroupId === 'control' ? 'Kale' : 'Carrot',
      );
    });

    test('emits value change events', () => {
      const eventCallback = jest.fn();
      render(
        <DesignSystemEventProvider callback={eventCallback}>
          <TestFixture />
        </DesignSystemEventProvider>,
      );

      simpleSelectTestUtils.toggleSelect(/Select a fruit/);
      simpleSelectTestUtils.selectOption('Banana');
      expect(eventCallback).toHaveBeenCalledWith({
        eventType: 'onValueChange',
        componentId: 'fruit_simple_select_test',
        componentType: 'simple_select',
        shouldStartInteraction: false,
        value: 'banana',
      });
    });
  });
});
