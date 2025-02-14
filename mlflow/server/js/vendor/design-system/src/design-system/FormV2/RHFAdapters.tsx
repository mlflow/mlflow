import type { CheckboxGroupProps } from 'antd/lib/checkbox';
import type { CheckboxValueType } from 'antd/lib/checkbox/Group';
import type { UseComboboxReturnValue } from 'downshift';
import type { HTMLAttributes, ReactElement } from 'react';
import React, { Children, useCallback, useEffect, useRef, useState } from 'react';
import type { Control, FieldPath, FieldValues, UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';

import type { CheckboxProps } from '../Checkbox';
import { Checkbox } from '../Checkbox';
import type { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type {
  ConditionalOptionalLabel,
  DialogComboboxContentProps,
  DialogComboboxOptionListProps,
  DialogComboboxProps,
  DialogComboboxTriggerProps,
} from '../DialogCombobox';
import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxTrigger,
} from '../DialogCombobox';
import { useDesignSystemTheme } from '../Hooks';
import type { InputProps, PasswordProps, TextAreaProps } from '../Input';
import { Input } from '../Input';
import type { LegacySelectProps, LegacySelectValue } from '../LegacySelect';
import { LegacySelect } from '../LegacySelect';
import type { RadioGroupProps } from '../Radio';
import { Radio } from '../Radio';
import type { SelectContentProps, SelectOptionProps, SelectProps, SelectTriggerProps } from '../Select';
import { Select, SelectContent, SelectOption, SelectTrigger } from '../Select';
import type { SwitchProps } from '../Switch';
import { Switch } from '../Switch';
import {
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMultiSelectInput,
  useComboboxState,
  useMultipleSelectionState,
} from '../TypeaheadCombobox';
import type { TypeaheadComboboxInputProps } from '../TypeaheadCombobox/TypeaheadComboboxInput';
import type { TypeaheadComboboxMenuProps } from '../TypeaheadCombobox/TypeaheadComboboxMenu';
import type { TypeaheadComboboxMultiSelectInputProps } from '../TypeaheadCombobox/TypeaheadComboboxMultiSelectInput';
import type { TypeaheadComboboxRootProps } from '../TypeaheadCombobox/TypeaheadComboboxRoot';
import { TypeaheadComboboxRoot } from '../TypeaheadCombobox/TypeaheadComboboxRoot';
import type { AnalyticsEventValueChangeNoPiiFlagProps, ValidationState } from '../types';

// These props are omitted, as `useController` handles them.
type OmittedOriginalProps = 'name' | 'onChange' | 'onBlur' | 'value' | 'defaultValue' | 'checked';

interface RHFControlledInputProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
> extends Omit<InputProps, OmittedOriginalProps>,
    UseControllerProps<TFieldValues, TName> {
  control: Control<TFieldValues>;
}

type AntInputValueType = string | ReadonlyArray<string> | number | undefined;

function RHFControlledInput<TFieldValues extends FieldValues>({
  name,
  control,
  rules,
  ...restProps
}: React.PropsWithChildren<RHFControlledInputProps<TFieldValues>>) {
  const { field } = useController({
    name: name,
    control: control,
    rules: rules,
  });

  return (
    <Input
      {...restProps}
      {...field}
      value={field.value as AntInputValueType}
      defaultValue={restProps.defaultValue as AntInputValueType}
    />
  );
}

interface RHFControlledPasswordInputProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
> extends Omit<PasswordProps, OmittedOriginalProps>,
    UseControllerProps<TFieldValues, TName> {
  control: Control<TFieldValues>;
}

function RHFControlledPasswordInput<TFieldValues extends FieldValues>({
  name,
  control,
  rules,
  ...restProps
}: React.PropsWithChildren<RHFControlledPasswordInputProps<TFieldValues>>) {
  const { field } = useController({
    name: name,
    control: control,
    rules: rules,
  });

  return <Input.Password {...restProps} {...field} value={field.value} defaultValue={restProps.defaultValue} />;
}

interface RHFControlledTextAreaProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
> extends Omit<TextAreaProps, OmittedOriginalProps>,
    UseControllerProps<TFieldValues, TName> {
  control: Control<TFieldValues>;
}

function RHFControlledTextArea<TFieldValues extends FieldValues>({
  name,
  control,
  rules,
  ...restProps
}: React.PropsWithChildren<RHFControlledTextAreaProps<TFieldValues>>) {
  const { field } = useController({
    name: name,
    control: control,
    rules: rules,
  });

  return <Input.TextArea {...restProps} {...field} value={field.value} defaultValue={restProps.defaultValue} />;
}

interface RHFControlledLegacySelectProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
> extends Omit<LegacySelectProps<unknown>, OmittedOriginalProps>,
    UseControllerProps<TFieldValues, TName> {
  control: Control<TFieldValues>;
}

/**
 * @deprecated Use `RHFControlledSelect` instead.
 */
function RHFControlledLegacySelect<TFieldValues extends FieldValues>({
  name,
  control,
  rules,
  ...restProps
}: React.PropsWithChildren<RHFControlledLegacySelectProps<TFieldValues>>) {
  const { field } = useController({
    name: name,
    control: control,
    rules: rules,
  });

  return (
    <LegacySelect
      {...restProps}
      {...field}
      value={field.value as LegacySelectValue}
      defaultValue={field.value as LegacySelectValue | undefined}
    />
  );
}

interface RHFControlledSelectProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
> extends Omit<SelectProps, OmittedOriginalProps>,
    UseControllerProps<TFieldValues, TName> {
  control: Control<TFieldValues>;
  options?: {
    label: string;
    value: string;
  }[];
  validationState?: ValidationState;
  width?: string | number;
  children?: ({ onChange }: { onChange: (value: string) => void }) => React.ReactNode;
  triggerProps?: Pick<SelectTriggerProps, 'minWidth' | 'maxWidth' | 'disabled' | 'style' | 'className'>;
  contentProps?: Pick<SelectContentProps, 'maxHeight' | 'minHeight' | 'loading' | 'style' | 'className' | 'width'>;
  optionProps?: Pick<SelectOptionProps, 'disabled' | 'disabledReason' | 'style' | 'className'>;
}

/**
 * @deprecated This component is no longer necessary as `SimpleSelect` can be used uncontrolled by RHF.
 * Please consult the Forms Guide on go/dubois.
 */
function RHFControlledSelect<TFieldValues extends FieldValues>({
  name,
  control,
  rules,
  options,
  validationState,
  children,
  width,
  triggerProps,
  contentProps,
  optionProps,
  ...restProps
}: React.PropsWithChildren<RHFControlledSelectProps<TFieldValues>> &
  HTMLAttributes<HTMLDivElement> &
  ConditionalOptionalLabel) {
  const { field } = useController({
    name: name,
    control: control,
    rules: rules,
  });

  const [selectedValueLabel, setSelectedValueLabel] = useState<string | undefined>(
    field.value ? (field.value.label ? field.value.label : field.value) : '',
  );

  const handleOnChange = (option: string | { label: string; value: string }) => {
    field.onChange(typeof option === 'object' ? option.value : option);
  };

  useEffect(() => {
    if (!field.value) {
      return;
    }

    // Find the appropriate label for the selected value
    if (!options?.length && children) {
      const renderedChildren = children({ onChange: handleOnChange }) as ReactElement;
      const child = (
        Array.isArray(renderedChildren) ? renderedChildren : Children.toArray(renderedChildren.props.children)
      ).find((child) => React.isValidElement(child) && (child as ReactElement).props.value === field.value);
      if (child) {
        if ((child as ReactElement).props?.children !== field.value) {
          setSelectedValueLabel((child as ReactElement).props.children as string);
        } else {
          setSelectedValueLabel(field.value);
        }
      }
    } else if (options?.length) {
      const option = options.find((option) => option.value === field.value);
      setSelectedValueLabel(option?.label);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [field.value]);

  return (
    <Select {...restProps} value={field.value}>
      <SelectTrigger
        {...triggerProps}
        width={width}
        onBlur={field.onBlur}
        validationState={validationState}
        ref={field.ref}
      >
        {selectedValueLabel}
      </SelectTrigger>
      <SelectContent {...contentProps} side="bottom">
        {options && options.length > 0
          ? options.map((option) => (
              <SelectOption {...optionProps} key={option.value} value={option.value} onChange={handleOnChange}>
                {option.label}
              </SelectOption>
            ))
          : // SelectOption out of the box gives users control over state and in this case RHF is controlling state
            // We expose onChange through a children renderer function to let users pass this down to SelectOption
            children?.({
              onChange: handleOnChange,
            })}
      </SelectContent>
    </Select>
  );
}

interface RHFControlledDialogComboboxProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
> extends Omit<DialogComboboxProps, OmittedOriginalProps>,
    UseControllerProps<TFieldValues, TName> {
  control: Control<TFieldValues>;
  validationState?: ValidationState;
  allowClear?: boolean;
  width?: string | number;
  children?: ({
    onChange,
    value,
  }: {
    onChange: (value: string) => void;
    value: string | string[];
    isChecked: (value: string) => boolean;
  }) => React.ReactNode;
  triggerProps?: Pick<
    DialogComboboxTriggerProps,
    'minWidth' | 'maxWidth' | 'disabled' | 'style' | 'className' | 'renderDisplayedValue'
  > & { 'data-testid'?: string };
  contentProps?: Pick<DialogComboboxContentProps, 'maxHeight' | 'minHeight' | 'style' | 'className'>;
  optionListProps?: Pick<DialogComboboxOptionListProps, 'loading' | 'withProgressiveLoading' | 'style' | 'className'>;
}

function RHFControlledDialogCombobox<TFieldValues extends FieldValues>({
  name,
  control,
  rules,
  children,
  allowClear,
  validationState,
  placeholder,
  width,
  triggerProps,
  contentProps,
  optionListProps,
  ...restProps
}: React.PropsWithChildren<RHFControlledDialogComboboxProps<TFieldValues>> &
  HTMLAttributes<HTMLDivElement> &
  ConditionalOptionalLabel) {
  const { field } = useController({
    name: name,
    control: control,
    rules: rules,
  });

  const [valueMap, setValueMap] = useState<Record<string, boolean>>({});

  const updateValueMap = useCallback((updatedValue?: string | string[]) => {
    if (updatedValue) {
      if (Array.isArray(updatedValue)) {
        setValueMap(
          updatedValue.reduce((acc: Record<string, boolean>, value: string) => {
            acc[value] = true;
            return acc;
          }, {}),
        );
      } else {
        setValueMap({ [updatedValue]: true });
      }
    } else {
      setValueMap({});
    }
  }, []);

  useEffect(() => {
    updateValueMap(field.value);
  }, [field.value, updateValueMap]);

  const handleOnChangeSingleSelect = (option: string) => {
    let updatedValue: string | string[] | undefined = field.value;

    if (field.value === option) {
      updatedValue = undefined;
    } else {
      updatedValue = option;
    }

    field.onChange(updatedValue);
    updateValueMap(updatedValue);
  };

  const handleOnChangeMultiSelect = (option: string) => {
    let updatedValue: string | string[] | undefined;

    if (field.value?.includes(option)) {
      updatedValue = field.value.filter((value: string) => value !== option);
    } else if (!field.value) {
      updatedValue = [option];
    } else {
      updatedValue = [...(field.value as string[]), option];
    }

    field.onChange(updatedValue);
    updateValueMap(updatedValue);
  };

  const handleOnChange = (option: string) => {
    if (restProps.multiSelect) {
      handleOnChangeMultiSelect(option);
    } else {
      handleOnChangeSingleSelect(option);
    }
  };

  const isChecked = (option: string) => {
    return valueMap[option];
  };

  const handleOnClear = () => {
    field.onChange(Array.isArray(field.value) ? [] : '');
    setValueMap({});
  };

  return (
    <DialogCombobox
      {...restProps}
      value={field.value ? (Array.isArray(field.value) ? field.value : [field.value]) : undefined}
    >
      <DialogComboboxTrigger
        {...triggerProps}
        onBlur={field.onBlur}
        allowClear={allowClear}
        validationState={validationState}
        onClear={handleOnClear}
        withInlineLabel={false}
        placeholder={placeholder}
        width={width}
        ref={field.ref}
      />
      <DialogComboboxContent {...contentProps} side="bottom" width={width}>
        <DialogComboboxOptionList {...optionListProps}>
          {children?.({
            onChange: handleOnChange,
            value: field.value,
            isChecked,
          })}
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
}

interface RHFControlledTypeaheadComboboxProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
> extends Omit<
      TypeaheadComboboxRootProps<TFieldValues>,
      OmittedOriginalProps | 'comboboxState' | 'multiSelect' | 'componentId'
    >,
    UseControllerProps<TFieldValues, TName>,
    AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  control: Control<TFieldValues, any>;
  name: TName;
  children: ({
    comboboxState,
    items,
  }: {
    comboboxState: UseComboboxReturnValue<TFieldValues[TName]>;
    items: Required<TFieldValues>[TName][];
  }) => React.ReactNode;
  validationState?: ValidationState;
  allItems: Required<TFieldValues>[TName][];
  itemToString: (item: Required<TFieldValues>[TName]) => string;
  matcher: (item: Required<TFieldValues>[TName], query: string) => boolean;
  /* Whether to allow the input value to not match any options in the list. For this to work, the item type (T) must be a string.
   * Defaults to false.
   */
  allowNewValue?: boolean;
  inputProps?: Partial<
    Omit<TypeaheadComboboxInputProps<TFieldValues[TName]>, 'comboboxState' | 'rhfOnChange' | 'validationState'>
  >;
  menuProps?: Partial<Omit<TypeaheadComboboxMenuProps<TFieldValues[TName]>, 'comboboxState'>>;
  onInputChange?: (inputValue: string) => void;
}

function RHFControlledTypeaheadCombobox<TFieldValues extends FieldValues, TName extends FieldPath<TFieldValues>>({
  name,
  control,
  rules,
  allItems,
  itemToString,
  matcher,
  allowNewValue,
  children,
  validationState,
  inputProps,
  menuProps,
  onInputChange,
  componentId,
  analyticsEvents,
  valueHasNoPii,
  ...props
}: React.PropsWithChildren<RHFControlledTypeaheadComboboxProps<TFieldValues, TName>> & HTMLAttributes<HTMLDivElement>) {
  const { field } = useController<TFieldValues, TName>({
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

  return (
    <TypeaheadComboboxRoot {...props} comboboxState={comboboxState}>
      <TypeaheadComboboxInput
        {...inputProps}
        validationState={validationState}
        formOnChange={field.onChange}
        comboboxState={comboboxState}
        ref={field.ref}
      />
      <TypeaheadComboboxMenu {...menuProps} comboboxState={comboboxState}>
        {children({
          comboboxState,
          items,
        })}
      </TypeaheadComboboxMenu>
    </TypeaheadComboboxRoot>
  );
}

interface RHFControlledMultiSelectTypeaheadComboboxProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
> extends Omit<
      TypeaheadComboboxRootProps<TFieldValues>,
      OmittedOriginalProps | 'comboboxState' | 'multiSelect' | 'componentId'
    >,
    UseControllerProps<TFieldValues, TName>,
    AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  control: Control<any, TFieldValues>;
  // Required override because the full object is returned but TName implies the value is a property of the object
  name: any;
  children: ({
    comboboxState,
    items,
    selectedItems,
  }: {
    comboboxState: UseComboboxReturnValue<TFieldValues>;
    items: TFieldValues[];
    selectedItems: TFieldValues[];
  }) => React.ReactNode;
  validationState?: ValidationState;
  allItems: TFieldValues[];
  itemToString: (item: TFieldValues) => string;
  matcher: (item: TFieldValues, query: string) => boolean;
  inputProps?: Partial<
    Omit<
      TypeaheadComboboxMultiSelectInputProps<TFieldValues>,
      'comboboxState' | 'rhfOnChange' | 'id' | 'validationState'
    >
  >;
  menuProps?: Partial<Omit<TypeaheadComboboxMenuProps<TFieldValues>, 'comboboxState'>>;
  onInputChange?: (inputValue: string) => void;
}

function RHFControlledMultiSelectTypeaheadCombobox<TFieldValues extends FieldValues>({
  name,
  control,
  rules,
  allItems,
  itemToString,
  matcher,
  children,
  validationState,
  inputProps,
  menuProps,
  onInputChange,
  componentId,
  analyticsEvents,
  valueHasNoPii,
  ...props
}: React.PropsWithChildren<RHFControlledMultiSelectTypeaheadComboboxProps<TFieldValues>> &
  HTMLAttributes<HTMLDivElement>) {
  const { field } = useController({
    name,
    control,
    rules,
  });
  const [inputValue, setInputValue] = useState('');
  const [selectedItems, setSelectedItems] = useState<TFieldValues[]>(field.value || []);

  useEffect(() => {
    setSelectedItems(field.value || []);
  }, [field.value]);

  const items = React.useMemo(
    () => allItems.filter((item) => matcher(item, inputValue)),
    [inputValue, matcher, allItems],
  );

  const handleItemUpdate = (item: any) => {
    field.onChange(item);
    setSelectedItems(item);
  };

  const comboboxState = useComboboxState<TFieldValues>({
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

  return (
    <TypeaheadComboboxRoot {...props} comboboxState={comboboxState}>
      <TypeaheadComboboxMultiSelectInput
        {...inputProps}
        multipleSelectionState={multipleSelectionState}
        selectedItems={selectedItems}
        setSelectedItems={handleItemUpdate}
        getSelectedItemLabel={itemToString}
        comboboxState={comboboxState}
        validationState={validationState}
        ref={field.ref}
      />
      <TypeaheadComboboxMenu {...menuProps} comboboxState={comboboxState}>
        {children({
          comboboxState,
          items,
          selectedItems,
        })}
      </TypeaheadComboboxMenu>
    </TypeaheadComboboxRoot>
  );
}

interface RHFControlledCheckboxGroupProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
> extends Omit<CheckboxGroupProps, OmittedOriginalProps>,
    UseControllerProps<TFieldValues, TName> {
  control: Control<TFieldValues>;
}

function RHFControlledCheckboxGroup<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
>({
  name,
  control,
  rules,
  ...restProps
}: React.PropsWithChildren<RHFControlledCheckboxGroupProps<TFieldValues, TName>>) {
  const { field } = useController({
    name: name,
    control: control,
    rules: rules,
  });

  return <Checkbox.Group {...restProps} {...field} value={field.value as CheckboxValueType[] | undefined} />;
}

interface RHFControlledCheckboxProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
> extends Omit<CheckboxProps, OmittedOriginalProps>,
    UseControllerProps<TFieldValues, TName> {
  control: Control<TFieldValues>;
}

function RHFControlledCheckbox<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
>({ name, control, rules, ...restProps }: React.PropsWithChildren<RHFControlledCheckboxProps<TFieldValues, TName>>) {
  const { field } = useController({
    name: name,
    control: control,
    rules: rules,
  });
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ marginTop: theme.spacing.sm }}>
      <Checkbox {...restProps} {...field} isChecked={field.value} />
    </div>
  );
}

interface RHFControlledRadioGroupProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
> extends Omit<RadioGroupProps, OmittedOriginalProps>,
    UseControllerProps<TFieldValues, TName> {
  control: Control<TFieldValues>;
}

function RHFControlledRadioGroup<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
>({ name, control, rules, ...restProps }: React.PropsWithChildren<RHFControlledRadioGroupProps<TFieldValues, TName>>) {
  const { field } = useController({
    name: name,
    control: control,
    rules: rules,
  });

  return <Radio.Group {...restProps} {...field} />;
}

interface RHFControlledSwitchProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
> extends Omit<SwitchProps, OmittedOriginalProps>,
    UseControllerProps<TFieldValues, TName> {
  control: Control<TFieldValues>;
}

function RHFControlledSwitch<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
>({ name, control, rules, ...restProps }: React.PropsWithChildren<RHFControlledSwitchProps<TFieldValues, TName>>) {
  const { field } = useController({
    name: name,
    control: control,
    rules: rules,
  });

  return <Switch {...restProps} {...field} checked={field.value} />;
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
