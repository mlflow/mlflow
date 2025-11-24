import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "@emotion/react/jsx-runtime";
import { expect, describe, test, jest, beforeEach } from '@jest/globals';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { createRef, forwardRef, useImperativeHandle, useReducer, useState } from 'react';
import { Controller, useForm } from 'react-hook-form';
import { SimpleSelect, SimpleSelectOption, SimpleSelectOptionGroup } from './SimpleSelect';
import { simpleSelectTestUtils } from '../../test-utils/rtl';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
import { setupSafexTesting } from '../utils/safex';
/** waits a little bit to ensure the assertions within the callback never pass */
async function expectNever(callback) {
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
            return (_jsxs(SimpleSelect, { componentId: "codegen_design-system_src_design-system_select_simpleselect.test.tsx_36", defaultValue: "bar", label: "hello", onChange: () => anythingThatCouldCauseARerender(), children: [_jsx(SimpleSelectOption, { value: "foo", children: "Foo's Label" }), _jsx(SimpleSelectOption, { value: "bar", children: "Bar's Label" }), _jsx(SimpleSelectOption, { value: "baz", children: "Baz's Label" })] }));
        }
        render(_jsx(Test, {}));
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
        render(_jsxs(SimpleSelect, { componentId: "codegen_design-system_src_design-system_select_simpleselect.test.tsx_63", id: "simple-select", placeholder: "Choose an option", value: "bar", children: [_jsx(SimpleSelectOption, { value: "foo", children: "Foo's Label" }), _jsx(SimpleSelectOption, { value: "bar", children: "Bar's Label" }), _jsx(SimpleSelectOption, { value: "baz", children: "Baz's Label" })] }));
        commonTest();
        expect(true).toBe(true); // Add an assertion to satisfy the lint requirement
    });
    test('works uncontrolled with defaultValue', () => {
        render(_jsxs(SimpleSelect, { componentId: "codegen_design-system_src_design-system_select_simpleselect.test.tsx_76", id: "simple-select", placeholder: "Choose an option", defaultValue: "bar", children: [_jsx(SimpleSelectOption, { value: "foo", children: "Foo's Label" }), _jsx(SimpleSelectOption, { value: "bar", children: "Bar's Label" }), _jsx(SimpleSelectOption, { value: "baz", children: "Baz's Label" })] }));
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
            return (_jsxs(SimpleSelect, { componentId: "codegen_design-system_src_design-system_select_simpleselect.test.tsx_105", id: "simple-select", placeholder: "Choose an option", value: selectedValue, onChange: (e) => {
                    setSelectedValue(e.target.value);
                    onChange(e);
                }, children: [_jsx(SimpleSelectOption, { value: "foo", children: "Foo's Label" }), _jsx(SimpleSelectOption, { value: "bar", children: "Bar's Label" }), _jsx(SimpleSelectOption, { value: "baz", children: "Baz's Label" })] }));
        };
        render(_jsx(SimpleSelectWrapper, {}));
        commonTest();
        expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
            target: expect.objectContaining({
                value: 'foo',
            }),
        }));
    });
    test('works uncontrolled with groups', () => {
        render(_jsxs(SimpleSelect, { componentId: "codegen_design-system_src_design-system_select_simpleselect.test.tsx_135", id: "simple-select", placeholder: "Choose an option", value: "bar", children: [_jsxs(SimpleSelectOptionGroup, { label: "Group 1", children: [_jsx(SimpleSelectOption, { value: "foo", children: "Foo's Label" }), _jsx(SimpleSelectOption, { value: "bar", children: "Bar's Label" })] }), _jsx(SimpleSelectOptionGroup, { label: "Group 2", children: _jsx(SimpleSelectOption, { value: "baz", children: "Baz's Label" }) })] }));
        commonTest();
        expect(true).toBe(true); // Add an assertion to satisfy the lint requirement
    });
    test('works controlled with groups', () => {
        const onChange = jest.fn();
        const SimpleSelectWrapper = () => {
            const [selectedValue, setSelectedValue] = useState('bar');
            return (_jsxs(SimpleSelect, { componentId: "codegen_design-system_src_design-system_select_simpleselect.test.tsx_157", id: "simple-select", placeholder: "Choose an option", value: selectedValue, onChange: (e) => {
                    setSelectedValue(e.target.value);
                    onChange(e);
                }, children: [_jsxs(SimpleSelectOptionGroup, { label: "Group 1", children: [_jsx(SimpleSelectOption, { value: "foo", children: "Foo's Label" }), _jsx(SimpleSelectOption, { value: "bar", children: "Bar's Label" })] }), _jsx(SimpleSelectOptionGroup, { label: "Group 2", children: _jsx(SimpleSelectOption, { value: "baz", children: "Baz's Label" }) })] }));
        };
        render(_jsx(SimpleSelectWrapper, {}));
        commonTest();
        expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
            target: expect.objectContaining({
                value: 'foo',
            }),
        }));
    });
    test('works uncontrolled with string and React node children', () => {
        const StringComponent = () => _jsx(_Fragment, { children: "Foo's Label" });
        render(_jsxs(SimpleSelect, { componentId: "codegen_design-system_src_design-system_select_simpleselect.test.tsx_193", id: "simple-select", placeholder: "Choose an option", value: "bar", children: [_jsx(SimpleSelectOption, { value: "foo", children: _jsx(StringComponent, {}) }), _jsx(SimpleSelectOption, { value: "bar", children: "Bar's Label" })] }));
        // Open the dropdown
        simpleSelectTestUtils.toggleSelect();
        expect(screen.getByRole('option', { name: "Foo's Label" })).toBeInTheDocument();
        fireEvent.click(screen.getByRole('option', { name: "Foo's Label" }));
        expect(screen.getByRole('combobox').textContent).toBe("Foo's Label");
    });
    test('works controlled with string and React node children', () => {
        const onChange = jest.fn();
        const StringComponent = () => _jsx(_Fragment, { children: "Bar's Label" });
        const SimpleSelectWrapper = () => {
            const [selectedValue, setSelectedValue] = useState('foo');
            return (_jsxs(SimpleSelect, { componentId: "codegen_design-system_src_design-system_select_simpleselect.test.tsx_219", id: "simple-select", placeholder: "Choose an option", value: selectedValue, onChange: (e) => {
                    setSelectedValue(e.target.value);
                    onChange(e);
                }, children: [_jsx(SimpleSelectOption, { value: "foo", children: "Foo's Label" }), _jsx(SimpleSelectOption, { value: "bar", children: _jsx(StringComponent, {}) })] }));
        };
        render(_jsx(SimpleSelectWrapper, {}));
        // Open the dropdown
        simpleSelectTestUtils.toggleSelect();
        expect(screen.getByRole('option', { name: "Foo's Label", selected: true })).toBeInTheDocument();
        fireEvent.click(screen.getByRole('option', { name: "Bar's Label" }));
        expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
            target: expect.objectContaining({
                value: 'bar',
            }),
        }));
        expect(screen.getByRole('combobox').textContent).toBe("Bar's Label");
    });
    test('clears the selected value when the clear button is clicked', () => {
        render(_jsxs(SimpleSelect, { componentId: "codegen_design-system_src_design-system_select_simpleselect.test.tsx_257", id: "simple-select", placeholder: "Choose an option", allowClear: true, children: [_jsx(SimpleSelectOption, { value: "foo", children: "Foo's Label" }), _jsx(SimpleSelectOption, { value: "bar", children: "Bar's Label" }), _jsx(SimpleSelectOption, { value: "baz", children: "Baz's Label" })] }));
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
        render(_jsxs(SimpleSelect, { componentId: "codegen_design-system_src_design-system_select_simpleselect.test.tsx_284", onChange: onChange, value: "bar", label: "test", children: [_jsx(SimpleSelectOption, { value: "foo", children: "Foo's Label" }), _jsx(SimpleSelectOption, { value: "bar", children: "Bar's Label" }), _jsx(SimpleSelectOption, { value: "baz", children: "Baz's Label" })] }));
        await expectNever(() => {
            expect(onChange).toHaveBeenCalled();
        });
    });
    test('should not cause an infinite render loop if state is an object', async () => {
        const ComplexStateComponent = () => {
            const [value, setValue] = useState({ value: 'pie' });
            return (_jsx(_Fragment, { children: _jsxs(SimpleSelect, { componentId: "codegen_design-system_src_design-system_select_simpleselect.test.tsx_302", id: "infinite-render-select", placeholder: "Choose an option", value: value.value, onChange: (e) => setValue({ value: e.target.value }), children: [_jsx(SimpleSelectOption, { value: "pie", children: _jsx(_Fragment, { children: "Pie" }) }), _jsx(SimpleSelectOption, { value: "bar", children: "Bar" }), _jsx(SimpleSelectOption, { value: "line", children: "Line" }), _jsx(SimpleSelectOption, { value: "bubble", children: _jsx(_Fragment, { children: "Bubble" }) }), _jsx(SimpleSelectOption, { value: "column", children: "Column" })] }) }));
        };
        expect(() => {
            render(_jsx(ComplexStateComponent, {}));
        }).not.toThrow();
    });
    test('closes dropdown when selected value is re-selected', () => {
        render(_jsxs(SimpleSelect, { componentId: "codegen_design-system_src_design-system_select_simpleselect.test.tsx_329", id: "simple-select", value: "foo", placeholder: "Choose an option", allowClear: true, label: "test", children: [_jsx(SimpleSelectOption, { value: "foo", children: "Foo's Label" }), _jsx(SimpleSelectOption, { value: "bar", children: "Bar's Label" }), _jsx(SimpleSelectOption, { value: "baz", children: "Baz's Label" })] }));
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
            return (_jsxs(SimpleSelect, { componentId: "codegen_design-system_src_design-system_select_simpleselect.test.tsx_349", placeholder: "Choose an option", onOpenChange: onOpenChange, label: "test", children: [_jsx(SimpleSelectOption, { value: "foo", children: "Foo's Label" }), _jsx(SimpleSelectOption, { value: "bar", children: "Bar's Label" })] }));
        };
        render(_jsx(SimpleSelectWrapper, {}));
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
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        const { setSafex } = setupSafexTesting();
        const UncontrolledComponent = ({ valueHasNoPii, defaultValue, allowClear, }) => (_jsx(DesignSystemEventProviderForTest, { children: _jsxs(SimpleSelect, { componentId: "simple_select_test", valueHasNoPii: valueHasNoPii, id: "simple-select", placeholder: "Choose an option", defaultValue: defaultValue, allowClear: allowClear, children: [_jsx(SimpleSelectOption, { value: "foo", children: "Foo's Label" }), _jsx(SimpleSelectOption, { value: "bar", children: "Bar's Label" }), _jsx(SimpleSelectOption, { value: "baz", children: "Baz's Label" })] }) }));
        const ControlledComponent = ({ valueHasNoPii, defaultValue, allowClear, }) => {
            const [selectedValue, setSelectedValue] = useState(defaultValue);
            return (_jsx(DesignSystemEventProviderForTest, { children: _jsxs(SimpleSelect, { componentId: "simple_select_test", valueHasNoPii: valueHasNoPii, id: "simple-select", placeholder: "Choose an option", value: selectedValue, onChange: (e) => {
                        setSelectedValue(e.target.value);
                    }, allowClear: allowClear, children: [_jsx(SimpleSelectOption, { value: "foo", children: "Foo's Label" }), _jsx(SimpleSelectOption, { value: "bar", children: "Bar's Label" }), _jsx(SimpleSelectOption, { value: "baz", children: "Baz's Label" })] }) }));
        };
        beforeEach(() => {
            setSafex({
                'databricks.fe.observability.defaultComponentView.simpleSelect': true,
            });
            // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
            window.IntersectionObserver = undefined;
        });
        describe.each([
            { Component: UncontrolledComponent, title: 'uncontrolled' },
            { Component: ControlledComponent, title: 'controlled' },
        ])('$title simple select analytics events', ({ Component }) => {
            test('emits value change events with value', () => {
                render(_jsx(Component, { valueHasNoPii: true }));
                expect(eventCallback).toHaveBeenCalledTimes(1);
                expect(eventCallback).toHaveBeenNthCalledWith(1, {
                    eventType: 'onView',
                    componentId: 'simple_select_test',
                    componentType: 'simple_select',
                    shouldStartInteraction: false,
                });
                simpleSelectTestUtils.toggleSelect();
                expect(eventCallback).toHaveBeenCalledTimes(1);
                fireEvent.click(screen.getByRole('option', { name: "Foo's Label" }));
                expect(eventCallback).toHaveBeenCalledTimes(2);
                expect(eventCallback).toHaveBeenNthCalledWith(2, {
                    eventType: 'onValueChange',
                    componentId: 'simple_select_test',
                    componentType: 'simple_select',
                    shouldStartInteraction: false,
                    value: 'foo',
                });
                simpleSelectTestUtils.toggleSelect();
                fireEvent.click(screen.getByRole('option', { name: "Bar's Label" }));
                expect(eventCallback).toHaveBeenCalledTimes(3);
                expect(eventCallback).toHaveBeenNthCalledWith(3, {
                    eventType: 'onValueChange',
                    componentId: 'simple_select_test',
                    componentType: 'simple_select',
                    shouldStartInteraction: false,
                    value: 'bar',
                });
            });
            test('emits value change events without value when valueHasNoPii is not set', () => {
                render(_jsx(Component, {}));
                expect(eventCallback).toHaveBeenCalledTimes(1);
                expect(eventCallback).toHaveBeenNthCalledWith(1, {
                    eventType: 'onView',
                    componentId: 'simple_select_test',
                    componentType: 'simple_select',
                    shouldStartInteraction: false,
                });
                simpleSelectTestUtils.toggleSelect();
                expect(eventCallback).toHaveBeenCalledTimes(1);
                fireEvent.click(screen.getByRole('option', { name: "Foo's Label" }));
                expect(eventCallback).toHaveBeenCalledTimes(2);
                expect(eventCallback).toHaveBeenNthCalledWith(2, {
                    eventType: 'onValueChange',
                    componentId: 'simple_select_test',
                    componentType: 'simple_select',
                    shouldStartInteraction: false,
                    value: undefined,
                });
            });
            test('emits a value change event when the value is cleared', () => {
                render(_jsx(Component, { valueHasNoPii: true, defaultValue: "foo", allowClear: true }));
                expect(eventCallback).toHaveBeenCalledTimes(1);
                expect(eventCallback).toHaveBeenNthCalledWith(1, {
                    eventType: 'onView',
                    componentId: 'simple_select_test',
                    componentType: 'simple_select',
                    shouldStartInteraction: false,
                    value: 'foo',
                });
                fireEvent.click(screen.getByRole('button', { name: /clear/i }));
                expect(eventCallback).toHaveBeenCalledTimes(2);
                expect(eventCallback).toHaveBeenNthCalledWith(2, {
                    eventType: 'onValueChange',
                    componentId: 'simple_select_test',
                    componentType: 'simple_select',
                    shouldStartInteraction: false,
                    value: '',
                });
            });
            test('does not emit a value change event for the default value', () => {
                render(_jsx(Component, { defaultValue: "foo" }));
                expect(eventCallback).toHaveBeenCalledTimes(1);
                expect(eventCallback).toHaveBeenNthCalledWith(1, {
                    eventType: 'onView',
                    componentId: 'simple_select_test',
                    componentType: 'simple_select',
                    shouldStartInteraction: false,
                });
            });
        });
    });
});
const TestRHFControlExampleForm = forwardRef(({ defaultValues, onSubmit = () => { }, rules, allowClear }, ref) => {
    const { control, handleSubmit, setValue } = useForm({ defaultValues });
    useImperativeHandle(ref, () => ({
        setValue,
    }));
    return (_jsxs("form", { onSubmit: handleSubmit(onSubmit), children: [_jsx(Controller, { name: "fruit", control: control, rules: rules, render: ({ field, fieldState: { error } }) => (_jsxs(_Fragment, { children: [_jsxs(SimpleSelect, { ...field, componentId: "fruit_simple_select_test", valueHasNoPii: true, placeholder: "Select a fruit", label: "Select a fruit", allowClear: allowClear, children: [_jsx(SimpleSelectOption, { value: "apple", children: "Apple" }), _jsx(SimpleSelectOption, { value: "banana", children: "Banana" }), _jsx(SimpleSelectOption, { value: "orange", children: "Orange" })] }), error && _jsx("span", { children: error.message })] })) }), _jsx(Controller, { name: "vegetable", control: control, rules: rules, render: ({ field, fieldState: { error } }) => (_jsxs(_Fragment, { children: [_jsxs(SimpleSelect, { ...field, componentId: "vegetable_simple_select_test", valueHasNoPii: true, placeholder: "Select a vegetable", label: "Select a vegetable", allowClear: allowClear, children: [_jsxs(SimpleSelectOptionGroup, { label: "Leafy Greens", children: [_jsx(SimpleSelectOption, { value: "spinach", children: "Spinach" }), _jsx(SimpleSelectOption, { value: "kale", children: "Kale" })] }), _jsxs(SimpleSelectOptionGroup, { label: "Root Vegetables", children: [_jsx(SimpleSelectOption, { value: "carrot", children: "Carrot" }), _jsx(SimpleSelectOption, { value: "potato", children: "Potato" })] })] }), error && _jsx("span", { children: error.message })] })) }), _jsx("button", { type: "submit", children: "Submit" })] }));
});
const TestRHFRegisterExampleForm = forwardRef(({ defaultValues, onSubmit = () => { }, rules, allowClear }, ref) => {
    const { register, handleSubmit, formState: { errors }, setValue, } = useForm({ defaultValues });
    useImperativeHandle(ref, () => ({
        setValue,
    }));
    return (_jsxs("form", { onSubmit: handleSubmit(onSubmit), children: [_jsxs(SimpleSelect, { ...register('fruit', rules), componentId: "fruit_simple_select_test", valueHasNoPii: true, placeholder: "Select a fruit", label: "Select a fruit", allowClear: allowClear, children: [_jsx(SimpleSelectOption, { value: "apple", children: "Apple" }), _jsx(SimpleSelectOption, { value: "banana", children: "Banana" }), _jsx(SimpleSelectOption, { value: "orange", children: "Orange" })] }), errors.fruit && _jsx("span", { children: errors.fruit.message }), _jsxs(SimpleSelect, { ...register('vegetable', rules), componentId: "vegetable_simple_select_test", valueHasNoPii: true, placeholder: "Select a vegetable", label: "Select a vegetable", allowClear: allowClear, children: [_jsxs(SimpleSelectOptionGroup, { label: "Leafy Greens", children: [_jsx(SimpleSelectOption, { value: "spinach", children: "Spinach" }), _jsx(SimpleSelectOption, { value: "kale", children: "Kale" })] }), _jsxs(SimpleSelectOptionGroup, { label: "Root Vegetables", children: [_jsx(SimpleSelectOption, { value: "carrot", children: "Carrot" }), _jsx(SimpleSelectOption, { value: "potato", children: "Potato" })] })] }), errors.vegetable && _jsx("span", { children: errors.vegetable.message }), _jsx("button", { type: "submit", children: "Submit" })] }));
});
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
        const { setSafex } = setupSafexTesting();
        beforeEach(() => {
            setSafex({
                'databricks.fe.observability.defaultComponentView.simpleSelect': true,
            });
            // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
            window.IntersectionObserver = undefined;
        });
        // eslint-disable-next-line jest/expect-expect
        test('integrates with React Hook Form', async () => {
            render(_jsx(TestFixture, {}));
            simpleSelectTestUtils.toggleSelect(/Select a fruit/);
            simpleSelectTestUtils.selectOption('Banana');
            simpleSelectTestUtils.expectSelectedOptionFromTriggerToBe('Banana', /Select a fruit/);
        });
        test('updates form values when selection changes', async () => {
            const onSubmit = jest.fn();
            render(_jsx(TestFixture, { defaultValues: { fruit: '', vegetable: '' }, onSubmit: onSubmit }));
            simpleSelectTestUtils.toggleSelect(/Select a fruit/);
            simpleSelectTestUtils.selectOption('Orange');
            fireEvent.submit(screen.getByRole('button', { name: 'Submit' }));
            await waitFor(() => {
                expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ fruit: 'orange' }), expect.anything());
                // Register syntax will return an empty string (matching real element) whereas control syntax will return undefined.
                expect(onSubmit.mock.calls[0][0]).toHaveProperty('vegetable');
                // @ts-expect-error TODO(FEINF-1796)
                expect(['', undefined]).toContain(onSubmit.mock.calls[0][0].vegetable);
            });
        });
        // eslint-disable-next-line jest/expect-expect
        test('displays initial form value', () => {
            render(_jsx(TestFixture, { defaultValues: { fruit: 'apple', vegetable: 'carrot' } }));
            simpleSelectTestUtils.expectSelectedOptionFromTriggerToBe('Apple', /Select a fruit/);
        });
        test('displays initial form value with option groups', () => {
            const { getByRole } = render(_jsx(TestFixture, { defaultValues: {
                    fruit: 'banana',
                    vegetable: 'carrot',
                } }));
            expect(getByRole('combobox', { name: /Select a fruit/ })).toHaveTextContent('Banana');
            expect(getByRole('combobox', { name: /Select a vegetable/ })).toHaveTextContent('Carrot');
        });
        test('validates required field', async () => {
            render(_jsx(TestFixture, { rules: { required: 'Please select a fruit' } }));
            fireEvent.submit(screen.getByRole('button', { name: 'Submit' }));
            await waitFor(() => {
                expect(screen.getAllByText('Please select a fruit')).toHaveLength(2);
            });
        });
        test('works with allowClear', () => {
            render(_jsx(TestFixture, { rules: { required: 'Please select a fruit' } }));
            fireEvent.click(screen.getByRole('combobox', { name: /Select a fruit/ }));
            fireEvent.click(screen.getByRole('option', { name: 'Banana' }));
            expect(screen.getByRole('combobox', { name: /Select a fruit/ }).textContent).toBe('Banana');
            fireEvent.click(screen.getByRole('button', { name: /clear/i }));
            expect(screen.getByRole('combobox', { name: /Select a fruit/ }).textContent).toBe('Select a fruit');
        });
        test('works as expected with setValue (must be controlled)', () => {
            const ref = createRef();
            render(_jsx(TestFixture, { ref: ref, defaultValues: {
                    fruit: 'apple',
                    vegetable: 'carrot',
                } }));
            expect(screen.getByRole('combobox', { name: /Select a fruit/ }).textContent).toBe('Apple');
            expect(screen.getByRole('combobox', { name: /Select a vegetable/ }).textContent).toBe('Carrot');
            act(() => {
                ref.current?.setValue('fruit', 'banana');
            });
            expect(screen.getByRole('combobox', { name: /Select a fruit/ }).textContent).toBe(testGroupId === 'control' ? 'Banana' : 'Apple');
            act(() => {
                ref.current?.setValue('vegetable', 'kale');
            });
            expect(screen.getByRole('combobox', { name: /Select a vegetable/ }).textContent).toBe(testGroupId === 'control' ? 'Kale' : 'Carrot');
        });
        test('emits value change events', () => {
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(TestFixture, {}) }));
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onView',
                componentId: 'fruit_simple_select_test',
                componentType: 'simple_select',
                shouldStartInteraction: false,
            });
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
//# sourceMappingURL=SimpleSelect.test.js.map