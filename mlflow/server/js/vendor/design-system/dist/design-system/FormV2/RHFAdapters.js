import { createElement as _createElement } from "@emotion/react";
import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { useMergeRefs } from '@floating-ui/react';
import React, { Children, useCallback, useEffect, useRef, useState } from 'react';
import { useController } from 'react-hook-form';
import { Checkbox } from '../Checkbox';
import { DialogCombobox, DialogComboboxContent, DialogComboboxOptionList, DialogComboboxTrigger, } from '../DialogCombobox';
import { useDesignSystemTheme } from '../Hooks';
import { Input } from '../Input';
import { LegacySelect } from '../LegacySelect';
import { Radio } from '../Radio';
import { Select, SelectContent, SelectOption, SelectTrigger } from '../Select';
import { Switch } from '../Switch';
import { TypeaheadComboboxInput, TypeaheadComboboxMenu, TypeaheadComboboxMultiSelectInput, useComboboxState, useMultipleSelectionState, } from '../TypeaheadCombobox';
import { TypeaheadComboboxRoot } from '../TypeaheadCombobox/TypeaheadComboboxRoot';
function RHFControlledInput({ name, control, rules, inputRef, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules,
    });
    const mergedRef = useMergeRefs([field.ref, inputRef]);
    return (_jsx(Input, { ...restProps, ...field, ref: mergedRef, value: field.value, defaultValue: restProps.defaultValue }));
}
function RHFControlledPasswordInput({ name, control, rules, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules,
    });
    return _jsx(Input.Password, { ...restProps, ...field, value: field.value, defaultValue: restProps.defaultValue });
}
function RHFControlledTextArea({ name, control, rules, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules,
    });
    return _jsx(Input.TextArea, { ...restProps, ...field, value: field.value, defaultValue: restProps.defaultValue });
}
/**
 * @deprecated Use `RHFControlledSelect` instead.
 */
function RHFControlledLegacySelect({ name, control, rules, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules,
    });
    return (_jsx(LegacySelect, { ...restProps, ...field, value: field.value, defaultValue: field.value }));
}
/**
 * @deprecated This component is no longer necessary as `SimpleSelect` can be used uncontrolled by RHF.
 * Please consult the Forms Guide on go/dubois.
 */
function RHFControlledSelect({ name, control, rules, options, validationState, children, width, triggerProps, contentProps, optionProps, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules,
    });
    const [selectedValueLabel, setSelectedValueLabel] = useState(field.value ? (field.value.label ? field.value.label : field.value) : '');
    const handleOnChange = (option) => {
        field.onChange(typeof option === 'object' ? option.value : option);
    };
    useEffect(() => {
        if (!field.value) {
            return;
        }
        // Find the appropriate label for the selected value
        if (!options?.length && children) {
            const renderedChildren = children({ onChange: handleOnChange });
            const child = (Array.isArray(renderedChildren) ? renderedChildren : Children.toArray(renderedChildren.props.children)).find((child) => React.isValidElement(child) && child.props.value === field.value);
            if (child) {
                if (child.props?.children !== field.value) {
                    setSelectedValueLabel(child.props.children);
                }
                else {
                    setSelectedValueLabel(field.value);
                }
            }
        }
        else if (options?.length) {
            const option = options.find((option) => option.value === field.value);
            setSelectedValueLabel(option?.label);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [field.value]);
    return (_jsxs(Select, { ...restProps, value: field.value, children: [_jsx(SelectTrigger, { ...triggerProps, width: width, onBlur: field.onBlur, validationState: validationState, ref: field.ref, children: selectedValueLabel }), _jsx(SelectContent, { ...contentProps, side: "bottom", children: options && options.length > 0
                    ? options.map((option) => (_createElement(SelectOption, { ...optionProps, key: option.value, value: option.value, onChange: handleOnChange }, option.label)))
                    : // SelectOption out of the box gives users control over state and in this case RHF is controlling state
                        // We expose onChange through a children renderer function to let users pass this down to SelectOption
                        children?.({
                            onChange: handleOnChange,
                        }) })] }));
}
function RHFControlledDialogCombobox({ name, control, rules, children, allowClear, validationState, placeholder, width, triggerProps, contentProps, optionListProps, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules,
    });
    const [valueMap, setValueMap] = useState({});
    const updateValueMap = useCallback((updatedValue) => {
        if (updatedValue) {
            if (Array.isArray(updatedValue)) {
                setValueMap(updatedValue.reduce((acc, value) => {
                    acc[value] = true;
                    return acc;
                }, {}));
            }
            else {
                setValueMap({ [updatedValue]: true });
            }
        }
        else {
            setValueMap({});
        }
    }, []);
    useEffect(() => {
        updateValueMap(field.value);
    }, [field.value, updateValueMap]);
    const handleOnChangeSingleSelect = (option) => {
        let updatedValue = field.value;
        if (field.value === option) {
            updatedValue = undefined;
        }
        else {
            updatedValue = option;
        }
        field.onChange(updatedValue);
        updateValueMap(updatedValue);
    };
    const handleOnChangeMultiSelect = (option) => {
        let updatedValue;
        if (field.value?.includes(option)) {
            updatedValue = field.value.filter((value) => value !== option);
        }
        else if (!field.value) {
            updatedValue = [option];
        }
        else {
            updatedValue = [...field.value, option];
        }
        field.onChange(updatedValue);
        updateValueMap(updatedValue);
    };
    const handleOnChange = (option) => {
        if (restProps.multiSelect) {
            handleOnChangeMultiSelect(option);
        }
        else {
            handleOnChangeSingleSelect(option);
        }
    };
    const isChecked = (option) => {
        return valueMap[option];
    };
    const handleOnClear = () => {
        field.onChange(Array.isArray(field.value) ? [] : '');
        setValueMap({});
    };
    return (_jsxs(DialogCombobox, { ...restProps, value: field.value ? (Array.isArray(field.value) ? field.value : [field.value]) : undefined, children: [_jsx(DialogComboboxTrigger, { ...triggerProps, onBlur: field.onBlur, allowClear: allowClear, validationState: validationState, onClear: handleOnClear, withInlineLabel: false, placeholder: placeholder, width: width, ref: field.ref }), _jsx(DialogComboboxContent, { ...contentProps, side: "bottom", width: width, children: _jsx(DialogComboboxOptionList, { ...optionListProps, children: children?.({
                        onChange: handleOnChange,
                        value: field.value,
                        isChecked,
                    }) }) })] }));
}
function RHFControlledTypeaheadCombobox({ name, control, rules, allItems, itemToString, matcher, allowNewValue, children, validationState, inputProps, menuProps, onInputChange, componentId, analyticsEvents, valueHasNoPii, preventUnsetOnBlur = false, ...props }) {
    const { field } = useController({
        name,
        control,
        rules,
    });
    const [items, setItems] = useState(allItems);
    const comboboxState = useComboboxState({
        allItems,
        items,
        setItems,
        itemToString,
        matcher,
        allowNewValue,
        formValue: field.value,
        formOnChange: field.onChange,
        formOnBlur: field.onBlur,
        componentId,
        analyticsEvents,
        valueHasNoPii,
        preventUnsetOnBlur,
    });
    const lastEmmitedInputValue = useRef(inputProps?.value);
    useEffect(() => {
        setItems(allItems);
    }, [allItems]);
    useEffect(() => {
        if (onInputChange && lastEmmitedInputValue.current !== comboboxState.inputValue) {
            onInputChange(comboboxState.inputValue);
            lastEmmitedInputValue.current = comboboxState.inputValue;
        }
    }, [comboboxState.inputValue, onInputChange]);
    return (_jsxs(TypeaheadComboboxRoot, { ...props, comboboxState: comboboxState, children: [_jsx(TypeaheadComboboxInput, { ...inputProps, validationState: validationState, formOnChange: field.onChange, comboboxState: comboboxState, ref: field.ref }), _jsx(TypeaheadComboboxMenu, { ...menuProps, comboboxState: comboboxState, children: children({
                    comboboxState,
                    items,
                }) })] }));
}
function RHFControlledMultiSelectTypeaheadCombobox({ name, control, rules, allItems, itemToString, matcher, children, validationState, inputProps, menuProps, onInputChange, componentId, analyticsEvents, valueHasNoPii, ...props }) {
    const { field } = useController({
        name,
        control,
        rules,
    });
    const [inputValue, setInputValue] = useState('');
    const [selectedItems, setSelectedItems] = useState(field.value || []);
    useEffect(() => {
        setSelectedItems(field.value || []);
    }, [field.value]);
    const items = React.useMemo(() => allItems.filter((item) => matcher(item, inputValue)), [inputValue, matcher, allItems]);
    const handleItemUpdate = (item) => {
        field.onChange(item);
        setSelectedItems(item);
    };
    const comboboxState = useComboboxState({
        allItems,
        items,
        setInputValue,
        matcher,
        itemToString,
        multiSelect: true,
        selectedItems,
        setSelectedItems: handleItemUpdate,
        formValue: field.value,
        formOnChange: field.onChange,
        formOnBlur: field.onBlur,
        componentId,
        analyticsEvents,
        valueHasNoPii,
    });
    const multipleSelectionState = useMultipleSelectionState(selectedItems, handleItemUpdate, comboboxState);
    const lastEmmitedInputValue = useRef(inputProps?.value);
    useEffect(() => {
        if (onInputChange && lastEmmitedInputValue.current !== comboboxState.inputValue) {
            onInputChange(comboboxState.inputValue);
            lastEmmitedInputValue.current = comboboxState.inputValue;
        }
    }, [comboboxState.inputValue, onInputChange]);
    return (_jsxs(TypeaheadComboboxRoot, { ...props, comboboxState: comboboxState, children: [_jsx(TypeaheadComboboxMultiSelectInput, { ...inputProps, multipleSelectionState: multipleSelectionState, selectedItems: selectedItems, setSelectedItems: handleItemUpdate, getSelectedItemLabel: itemToString, comboboxState: comboboxState, validationState: validationState, ref: field.ref }), _jsx(TypeaheadComboboxMenu, { ...menuProps, comboboxState: comboboxState, children: children({
                    comboboxState,
                    items,
                    selectedItems,
                }) })] }));
}
function RHFControlledCheckboxGroup({ name, control, rules, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules,
    });
    return _jsx(Checkbox.Group, { ...restProps, ...field, value: field.value });
}
function RHFControlledCheckbox({ name, control, rules, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules,
    });
    const { theme } = useDesignSystemTheme();
    return (_jsx("div", { css: { marginTop: theme.spacing.sm }, children: _jsx(Checkbox, { ...restProps, ...field, isChecked: field.value }) }));
}
function RHFControlledRadioGroup({ name, control, rules, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules,
    });
    return _jsx(Radio.Group, { ...restProps, ...field });
}
function RHFControlledSwitch({ name, control, rules, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules,
    });
    return _jsx(Switch, { ...restProps, ...field, checked: field.value });
}
export const RHFControlledComponents = {
    Input: RHFControlledInput,
    Password: RHFControlledPasswordInput,
    TextArea: RHFControlledTextArea,
    LegacySelect: RHFControlledLegacySelect,
    Select: RHFControlledSelect,
    DialogCombobox: RHFControlledDialogCombobox,
    Checkbox: RHFControlledCheckbox,
    CheckboxGroup: RHFControlledCheckboxGroup,
    RadioGroup: RHFControlledRadioGroup,
    TypeaheadCombobox: RHFControlledTypeaheadCombobox,
    MultiSelectTypeaheadCombobox: RHFControlledMultiSelectTypeaheadCombobox,
    Switch: RHFControlledSwitch,
};
//# sourceMappingURL=RHFAdapters.js.map