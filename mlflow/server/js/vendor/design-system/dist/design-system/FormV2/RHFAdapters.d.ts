import type { CheckboxGroupProps } from 'antd/lib/checkbox';
import type { UseComboboxReturnValue } from 'downshift';
import type { HTMLAttributes } from 'react';
import React from 'react';
import type { Control, FieldPath, FieldValues, UseControllerProps } from 'react-hook-form';
import type { CheckboxProps } from '../Checkbox';
import type { DialogComboboxContentProps, DialogComboboxOptionListProps, DialogComboboxProps, DialogComboboxTriggerProps } from '../DialogCombobox';
import type { InputProps, PasswordProps, TextAreaProps } from '../Input';
import type { SelectProps } from '../LegacySelect';
import type { RadioGroupProps } from '../Radio';
import type { SelectV2ContentProps, SelectV2OptionProps, SelectV2Props, SelectV2TriggerProps } from '../SelectV2';
import type { SwitchProps } from '../Switch';
import type { TypeaheadComboboxInputProps } from '../TypeaheadCombobox/TypeaheadComboboxInput';
import type { TypeaheadComboboxMenuProps } from '../TypeaheadCombobox/TypeaheadComboboxMenu';
import type { TypeaheadComboboxMultiSelectInputProps } from '../TypeaheadCombobox/TypeaheadComboboxMultiSelectInput';
import type { TypeaheadComboboxRootProps } from '../TypeaheadCombobox/TypeaheadComboboxRoot';
import type { ValidationState } from '../types';
type OmittedOriginalProps = 'name' | 'onChange' | 'onBlur' | 'value' | 'defaultValue' | 'checked';
interface RHFControlledInputProps<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>> extends Omit<InputProps, OmittedOriginalProps>, UseControllerProps<TFieldValues, TName> {
    control: Control<TFieldValues>;
}
declare function RHFControlledInput<TFieldValues extends FieldValues>({ name, control, rules, ...restProps }: React.PropsWithChildren<RHFControlledInputProps<TFieldValues>>): import("@emotion/react/jsx-runtime").JSX.Element;
interface RHFControlledPasswordInputProps<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>> extends Omit<PasswordProps, OmittedOriginalProps>, UseControllerProps<TFieldValues, TName> {
    control: Control<TFieldValues>;
}
declare function RHFControlledPasswordInput<TFieldValues extends FieldValues>({ name, control, rules, ...restProps }: React.PropsWithChildren<RHFControlledPasswordInputProps<TFieldValues>>): import("@emotion/react/jsx-runtime").JSX.Element;
interface RHFControlledTextAreaProps<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>> extends Omit<TextAreaProps, OmittedOriginalProps>, UseControllerProps<TFieldValues, TName> {
    control: Control<TFieldValues>;
}
declare function RHFControlledTextArea<TFieldValues extends FieldValues>({ name, control, rules, ...restProps }: React.PropsWithChildren<RHFControlledTextAreaProps<TFieldValues>>): import("@emotion/react/jsx-runtime").JSX.Element;
interface RHFControlledSelectProps<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>> extends Omit<SelectProps<unknown>, OmittedOriginalProps>, UseControllerProps<TFieldValues, TName> {
    control: Control<TFieldValues>;
}
declare function RHFControlledSelect<TFieldValues extends FieldValues>({ name, control, rules, ...restProps }: React.PropsWithChildren<RHFControlledSelectProps<TFieldValues>>): import("@emotion/react/jsx-runtime").JSX.Element;
interface RHFControlledSelectV2Props<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>> extends Omit<SelectV2Props, OmittedOriginalProps>, UseControllerProps<TFieldValues, TName> {
    control: Control<TFieldValues>;
    options?: {
        label: string;
        value: string;
    }[];
    validationState?: ValidationState;
    width?: string | number;
    children?: ({ onChange }: {
        onChange: (value: string) => void;
    }) => React.ReactNode;
    triggerProps?: Pick<SelectV2TriggerProps, 'minWidth' | 'maxWidth' | 'disabled' | 'style' | 'className'>;
    contentProps?: Pick<SelectV2ContentProps, 'maxHeight' | 'minHeight' | 'loading' | 'style' | 'className' | 'width'>;
    optionProps?: Pick<SelectV2OptionProps, 'disabled' | 'disabledReason' | 'style' | 'className'>;
}
declare function RHFControlledSelectV2<TFieldValues extends FieldValues>({ name, control, rules, options, validationState, children, width, triggerProps, contentProps, optionProps, ...restProps }: React.PropsWithChildren<RHFControlledSelectV2Props<TFieldValues>> & HTMLAttributes<HTMLDivElement>): import("@emotion/react/jsx-runtime").JSX.Element;
interface RHFControlledDialogComboboxProps<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>> extends Omit<DialogComboboxProps, OmittedOriginalProps>, UseControllerProps<TFieldValues, TName> {
    control: Control<TFieldValues>;
    validationState?: ValidationState;
    allowClear?: boolean;
    width?: string | number;
    children?: ({ onChange, value, }: {
        onChange: (value: string) => void;
        value: string | string[];
        isChecked: (value: string) => boolean;
    }) => React.ReactNode;
    triggerProps?: Pick<DialogComboboxTriggerProps, 'minWidth' | 'maxWidth' | 'disabled' | 'style' | 'className'>;
    contentProps?: Pick<DialogComboboxContentProps, 'maxHeight' | 'minHeight' | 'style' | 'className'>;
    optionListProps?: Pick<DialogComboboxOptionListProps, 'loading' | 'withProgressiveLoading' | 'style' | 'className'>;
}
declare function RHFControlledDialogCombobox<TFieldValues extends FieldValues>({ name, control, rules, children, allowClear, validationState, placeholder, width, triggerProps, contentProps, optionListProps, ...restProps }: React.PropsWithChildren<RHFControlledDialogComboboxProps<TFieldValues>> & HTMLAttributes<HTMLDivElement>): import("@emotion/react/jsx-runtime").JSX.Element;
interface RHFControlledTypeaheadComboboxProps<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>> extends Omit<TypeaheadComboboxRootProps<TFieldValues>, OmittedOriginalProps | 'comboboxState' | 'multiSelect'>, UseControllerProps<TFieldValues, TName> {
    control: Control<any, TFieldValues>;
    name: any;
    children: ({ comboboxState, items, }: {
        comboboxState: UseComboboxReturnValue<TFieldValues>;
        items: TFieldValues[];
    }) => React.ReactNode;
    validationState?: ValidationState;
    allItems: TFieldValues[];
    itemToString: (item: TFieldValues) => string;
    matcher: (item: TFieldValues, query: string) => boolean;
    allowNewValue?: boolean;
    inputProps?: Partial<Omit<TypeaheadComboboxInputProps<TFieldValues>, 'comboboxState' | 'rhfOnChange' | 'id' | 'validationState'>>;
    menuProps?: Partial<Omit<TypeaheadComboboxMenuProps<TFieldValues>, 'comboboxState'>>;
}
declare function RHFControlledTypeaheadCombobox<TFieldValues extends FieldValues>({ name, control, rules, allItems, itemToString, matcher, allowNewValue, children, validationState, inputProps, menuProps, ...props }: React.PropsWithChildren<RHFControlledTypeaheadComboboxProps<TFieldValues>> & HTMLAttributes<HTMLDivElement>): import("@emotion/react/jsx-runtime").JSX.Element;
interface RHFControlledMultiSelectTypeaheadComboboxProps<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>> extends Omit<TypeaheadComboboxRootProps<TFieldValues>, OmittedOriginalProps | 'comboboxState' | 'multiSelect'>, UseControllerProps<TFieldValues, TName> {
    control: Control<any, TFieldValues>;
    name: any;
    children: ({ comboboxState, items, selectedItems, }: {
        comboboxState: UseComboboxReturnValue<TFieldValues>;
        items: TFieldValues[];
        selectedItems: TFieldValues[];
    }) => React.ReactNode;
    validationState?: ValidationState;
    allItems: TFieldValues[];
    itemToString: (item: TFieldValues) => string;
    matcher: (item: TFieldValues, query: string) => boolean;
    inputProps?: Partial<Omit<TypeaheadComboboxMultiSelectInputProps<TFieldValues>, 'comboboxState' | 'rhfOnChange' | 'id' | 'validationState'>>;
    menuProps?: Partial<Omit<TypeaheadComboboxMenuProps<TFieldValues>, 'comboboxState'>>;
}
declare function RHFControlledMultiSelectTypeaheadCombobox<TFieldValues extends FieldValues>({ name, control, rules, allItems, itemToString, matcher, children, validationState, inputProps, menuProps, ...props }: React.PropsWithChildren<RHFControlledMultiSelectTypeaheadComboboxProps<TFieldValues>> & HTMLAttributes<HTMLDivElement>): import("@emotion/react/jsx-runtime").JSX.Element;
interface RHFControlledCheckboxGroupProps<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>> extends Omit<CheckboxGroupProps, OmittedOriginalProps>, UseControllerProps<TFieldValues, TName> {
    control: Control<TFieldValues>;
}
declare function RHFControlledCheckboxGroup<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>>({ name, control, rules, ...restProps }: React.PropsWithChildren<RHFControlledCheckboxGroupProps<TFieldValues, TName>>): import("@emotion/react/jsx-runtime").JSX.Element;
interface RHFControlledCheckboxProps<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>> extends Omit<CheckboxProps, OmittedOriginalProps>, UseControllerProps<TFieldValues, TName> {
    control: Control<TFieldValues>;
}
declare function RHFControlledCheckbox<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>>({ name, control, rules, ...restProps }: React.PropsWithChildren<RHFControlledCheckboxProps<TFieldValues, TName>>): import("@emotion/react/jsx-runtime").JSX.Element;
interface RHFControlledRadioGroupProps<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>> extends Omit<RadioGroupProps, OmittedOriginalProps>, UseControllerProps<TFieldValues, TName> {
    control: Control<TFieldValues>;
}
declare function RHFControlledRadioGroup<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>>({ name, control, rules, ...restProps }: React.PropsWithChildren<RHFControlledRadioGroupProps<TFieldValues, TName>>): import("@emotion/react/jsx-runtime").JSX.Element;
interface RHFControlledSwitchProps<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>> extends Omit<SwitchProps, OmittedOriginalProps>, UseControllerProps<TFieldValues, TName> {
    control: Control<TFieldValues>;
}
declare function RHFControlledSwitch<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>>({ name, control, rules, ...restProps }: React.PropsWithChildren<RHFControlledSwitchProps<TFieldValues, TName>>): import("@emotion/react/jsx-runtime").JSX.Element;
export declare const RHFControlledComponents: {
    Input: typeof RHFControlledInput;
    Password: typeof RHFControlledPasswordInput;
    TextArea: typeof RHFControlledTextArea;
    Select: typeof RHFControlledSelect;
    SelectV2: typeof RHFControlledSelectV2;
    DialogCombobox: typeof RHFControlledDialogCombobox;
    Checkbox: typeof RHFControlledCheckbox;
    CheckboxGroup: typeof RHFControlledCheckboxGroup;
    RadioGroup: typeof RHFControlledRadioGroup;
    TypeaheadCombobox: typeof RHFControlledTypeaheadCombobox;
    MultiSelectTypeaheadCombobox: typeof RHFControlledMultiSelectTypeaheadCombobox;
    Switch: typeof RHFControlledSwitch;
};
export {};
//# sourceMappingURL=RHFAdapters.d.ts.map