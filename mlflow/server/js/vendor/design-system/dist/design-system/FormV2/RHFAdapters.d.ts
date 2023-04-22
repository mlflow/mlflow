/// <reference types="react" />
import type { CheckboxGroupProps } from 'antd/lib/checkbox';
import type { Control, FieldPath, FieldValues, UseControllerProps } from 'react-hook-form';
import type { CheckboxProps } from '../Checkbox';
import type { InputProps, PasswordProps, TextAreaProps } from '../Input';
import type { RadioGroupProps } from '../Radio';
import type { SelectProps } from '../Select';
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
export declare const RHFControlledComponents: {
    Input: typeof RHFControlledInput;
    Password: typeof RHFControlledPasswordInput;
    TextArea: typeof RHFControlledTextArea;
    Select: typeof RHFControlledSelect;
    Checkbox: typeof RHFControlledCheckbox;
    CheckboxGroup: typeof RHFControlledCheckboxGroup;
    RadioGroup: typeof RHFControlledRadioGroup;
};
export {};
//# sourceMappingURL=RHFAdapters.d.ts.map